import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urlsplit

from benchmark.baseline import SingleAgentBaseline
from benchmark.evaluator import BenchmarkEvaluator
from config import settings
from config.settings import BENCHMARK_DIR, LLM_PROVIDER, validate_settings
from core.orchestrator import MedhaOrchestrator
from storage.workspace import cleanup_stale_workspaces
from utils.privacy import storage_mode_context
from utils.logger import get_logger
from utils.retry import SAFE_PROVIDER_ERROR_CODES

logger = get_logger("main")

BANNER = """
======================================================================
                              MEDHA AI
======================================================================
               Multi-Agent Evidence-Driven Intelligence
         Problem Statement 02 - Chain of Command (SEED 2026)
======================================================================
"""

# PowerShell 5.1 decodes redirected native output with the OEM code page, garbling typographic Unicode.
_TYPOGRAPHIC_TO_ASCII = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u2026": "...",
    "\u2264": "<=", "\u2265": ">=", "\u2192": "->", "\u2022": "*",
})


def _console_text(text: Any) -> str:
    return str(text or "").translate(_TYPOGRAPHIC_TO_ASCII)


_DIRECT_ANSWER_HEADING = re.compile(r"^#{1,6}\s*direct answer\s*:?\s*$", re.IGNORECASE)
_SAFE_DIAGNOSTIC_LABEL = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_SAFE_REFERENCE_IDENTIFIER = re.compile(
    r"^(?:claim|ev|evidence|chunk|src|source|ref|msg|message|doc|file|benchmark)[_:.-]"
    r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
)
_SAFE_DIAGNOSTIC_STAGES = frozenset({
    "baseline",
    "claim_extraction",
    "deliberation",
    "general",
    "planning",
    "repair",
    "specialist",
    "specialist_execution",
    "synthesis",
    "verification",
})
_SENSITIVE_REFERENCE_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)bearer\s+\S+"),
)


def _direct_answer(report: str) -> Optional[str]:
    """Return the body of the report's 'Direct Answer' section, or None if absent."""
    lines = (report or "").splitlines()
    for index, line in enumerate(lines):
        if _DIRECT_ANSWER_HEADING.match(line.strip()):
            body: List[str] = []
            for following in lines[index + 1:]:
                stripped = following.strip()
                if stripped.startswith("#") or stripped == "---":
                    break
                body.append(following.rstrip())
            text = "\n".join(body).strip()
            return text or None
    return None


def _answer_for_console(report: str, full: bool) -> str:
    if full:
        return _console_text(report)
    return _console_text(_direct_answer(report) or report)


def _safe_diagnostic_label(value: Any) -> str:
    text = str(value or "")
    return text if _SAFE_DIAGNOSTIC_LABEL.fullmatch(text) else "unknown"


def _safe_reference_identifier(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value or len(value) > 256 or any(char.isspace() for char in value):
        return None
    lowered = value.lower()
    if any(secret_marker in lowered for secret_marker in (
        "api_key", "apikey", "token=", "password", "secret", "bearer", "authorization",
    )):
        return None
    if any(pattern.search(value) for pattern in _SENSITIVE_REFERENCE_PATTERNS):
        return None
    if _SAFE_REFERENCE_IDENTIFIER.fullmatch(value):
        return value
    parsed = urlsplit(value)
    if (
        parsed.scheme in {"https", "local"}
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
        and (parsed.scheme == "local" or bool(parsed.hostname))
    ):
        return value
    return None


def _print_deliberation_diagnostics(summary: dict) -> None:
    def token_value(value: Any) -> str:
        return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else "N/A"

    reason = _safe_diagnostic_label(summary.get("termination_reason"))
    print(
        "Deliberation: "
        f"termination={reason}, claims_considered={summary.get('claims_considered', 0)}, "
        f"triggers={summary.get('trigger_count', 0)}, "
        f"suppressed={summary.get('suppressed_trigger_count', 0)}, "
        f"coalesced_duplicates={summary.get('coalesced_duplicate_trigger_count', 0)}, "
        f"challenges={summary.get('challenge_count', 0)}, "
        f"exchanges={summary.get('exchange_count', 0)}, "
        f"completed_exchanges={summary.get('completed_exchange_count', 0)}, "
        f"completed_rounds={summary.get('completed_round_count', 0)}, "
        f"messages={summary.get('total_messages', 0)}, "
        f"cumulative_tokens={summary.get('total_tokens', 'N/A')}, "
        f"token_source={_safe_diagnostic_label(summary.get('token_usage_source'))}"
    )
    for round_record in summary.get("rounds", []):
        print(
            f"  Deliberation round {round_record.get('round_number', '?')}: "
            f"exchanges={round_record.get('exchange_count', 0)}, "
            f"messages={round_record.get('message_count', 0)}, "
            f"incremental_tokens={round_record.get('incremental_tokens', 'N/A')}, "
            f"cumulative_tokens={round_record.get('cumulative_tokens', 'N/A')}, "
            f"completed={'yes' if round_record.get('completed') else 'no'}, "
            f"termination={_safe_diagnostic_label(round_record.get('termination_reason'))}"
        )
    for exchange in summary.get("exchanges", []):
        print(
            f"  Deliberation exchange: round={exchange.get('round_number', '?')}, "
            f"type={_safe_diagnostic_label(exchange.get('trigger_type'))}, "
            f"claim_ref={_safe_diagnostic_label(exchange.get('claim_reference'))}, "
            f"related_claim_refs={','.join(filter(None, (_safe_diagnostic_label(ref) for ref in exchange.get('related_claim_references', [])))) or 'none'}, "
            f"source_refs={','.join(filter(None, (_safe_reference_identifier(ref) for ref in exchange.get('source_references', [])))) or 'redacted/none'}, "
            f"tracker_delta={token_value(exchange.get('tracker_accounted_token_delta'))}, "
            f"estimated_delta={token_value(exchange.get('tracker_estimated_token_delta'))}, "
            f"reserved_input={token_value(exchange.get('estimated_input_tokens'))}, "
            f"reserved_output={token_value(exchange.get('reserved_output_tokens'))}, "
            f"provider_usage=({token_value(exchange.get('provider_prompt_tokens'))}/"
            f"{token_value(exchange.get('provider_candidate_tokens'))}/"
            f"{token_value(exchange.get('provider_total_tokens'))})"
        )
        for attempt in exchange.get("attempts", []):
            code = attempt.get("provider_error_code")
            code = code if code in SAFE_PROVIDER_ERROR_CODES else "N/A"
            status_code = attempt.get("http_status_code")
            status_code = status_code if isinstance(status_code, int) and 100 <= status_code <= 599 else "N/A"
            print(
                f"    Deliberation attempt {attempt.get('attempt_number', '?')}: "
                f"retry={attempt.get('retry_number', 0)}/"
                f"{attempt.get('max_retry_attempts', '?')}, "
                f"status={_safe_diagnostic_label(attempt.get('status'))}, "
                f"reserved=({token_value(attempt.get('reserved_input_tokens'))}/"
                f"{token_value(attempt.get('reserved_output_tokens'))}/"
                f"{token_value(attempt.get('reserved_total_tokens'))}), "
                f"provider_usage=({token_value(attempt.get('provider_prompt_tokens'))}/"
                f"{token_value(attempt.get('provider_candidate_tokens'))}/"
                f"{token_value(attempt.get('provider_total_tokens'))}), "
                f"usage_source={_safe_diagnostic_label(attempt.get('usage_source'))}, "
                f"exception={_safe_diagnostic_label(attempt.get('error_category'))}, "
                f"http_status={status_code}, provider_code={code}, "
                f"retry_classification={_safe_diagnostic_label(attempt.get('retry_classification'))}"
            )


def _print_execution_diagnostics(snapshot: dict) -> None:
    """Print payload-free telemetry before the execution workspace is cleaned up."""
    if snapshot.get("diagnostic_phase") == "after_cleanup":
        mode = _safe_diagnostic_label(snapshot.get("cleanup_mode"))
        verified = snapshot.get("cleanup_verified")
        removed = snapshot.get("cleanup_removed")
        print(
            f"Post-interruption cleanup: mode={mode}, "
            f"verified={'yes' if verified is True else 'no' if verified is False else 'unknown'}, "
            f"removed={'yes' if removed is True else 'no' if removed is False else 'unknown'}"
        )
        return
    print("\n--- Execution Diagnostics (before workspace cleanup) ---")
    print(f"Pre-cleanup status: {_safe_diagnostic_label(snapshot.get('status'))}")
    reasons = [
        _safe_diagnostic_label(reason)
        for reason in snapshot.get("status_reasons", [])
        if _SAFE_DIAGNOSTIC_LABEL.fullmatch(str(reason or ""))
    ]
    print(f"Status reason(s): {', '.join(reasons) if reasons else 'none recorded'}")

    usage = snapshot.get("token_usage") or {}

    def token_count(name: str) -> str:
        value = usage.get(name)
        return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else "N/A"

    print(
        "Tokens (existing tracker totals): "
        f"input={token_count('prompt_tokens')}, "
        f"output={token_count('candidate_tokens')}, "
        f"total={token_count('total_tokens')}, "
        f"accounted={token_count('accounted_total_tokens')}, "
        f"source={_safe_diagnostic_label(usage.get('usage_source'))}"
    )

    records = snapshot.get("provider_call_records") or []
    counts = {"SUCCEEDED": 0, "FAILED": 0, "TIMED_OUT": 0, "CANCELLED": 0, "IN_PROGRESS": 0}
    for record in records:
        status_value = getattr(record.status, "value", record.status)
        if status_value in counts:
            counts[status_value] += 1
    retry_count = sum(
        1 for record in records
        if isinstance(getattr(record, "retry_number", None), int) and record.retry_number > 0
    )
    dropped = snapshot.get("provider_call_records_dropped", 0)
    print(
        f"Provider calls: total={len(records)}, succeeded={counts['SUCCEEDED']}, "
        f"failed={counts['FAILED']}, timed_out={counts['TIMED_OUT']}, "
        f"cancelled={counts['CANCELLED']}, retried_attempts={retry_count}, "
        f"dropped_records={dropped}"
    )

    for record in records:
        status_value = getattr(record.status, "value", record.status)
        is_retry = isinstance(record.retry_number, int) and record.retry_number > 0
        if status_value == "SUCCEEDED" and not is_retry:
            continue
        exception_class = _safe_diagnostic_label(record.error_category) if record.error_category else "none"
        provider_code = (
            record.provider_error_code
            if record.provider_error_code in SAFE_PROVIDER_ERROR_CODES
            else "N/A"
        )
        status_code = record.http_status_code
        status_code_text = str(status_code) if isinstance(status_code, int) and 100 <= status_code <= 599 else "N/A"
        retry_limit = record.max_retry_attempts if isinstance(record.max_retry_attempts, int) else "?"
        retry_classification = _safe_diagnostic_label(record.retry_classification) if record.retry_classification else "N/A"
        stage = record.stage if record.stage in _SAFE_DIAGNOSTIC_STAGES else "unknown"
        print(
            f"  Provider attempt: stage={stage}, status={_safe_diagnostic_label(status_value)}, "
            f"attempt={record.attempt_number}, retry_attempt={record.retry_number + 1}/{retry_limit}, "
            f"exception={exception_class}, http_status={status_code_text}, "
            f"provider_code={provider_code}, retry_classification={retry_classification}"
        )
    deliberation = snapshot.get("deliberation")
    if deliberation is not None:
        _print_deliberation_diagnostics(deliberation)


def _format_rate(val: Any) -> str:
    """Format rate as percentage string, safely handling both fraction (0.0-1.0) and percentage (0.0-100.0)."""
    if val is None:
        return "N/A (no claims reviewed)"
    try:
        f = float(val or 0.0)
        pct = f * 100.0 if (0.0 < f <= 1.0) else f
        return f"{pct:.1f}%"
    except (ValueError, TypeError):
        return "N/A"


async def run_interactive_session(full_report: bool = False, persistent: bool = False):
    """Run interactive terminal session for MEDHA."""
    print(BANNER)
    try:
        validate_settings()
        print(f"[OK] Configuration and {LLM_PROVIDER.upper()} API credentials verified.")
    except Exception as e:
        print(f"[ERROR] Configuration error: {e}")
        sys.exit(1)

    storage_mode = "persistent" if persistent else settings.MEDHA_STORAGE_MODE
    orchestrator = MedhaOrchestrator(storage_mode=storage_mode)

    while True:
        try:
            print("\nEnter your task (or type 'exit' to quit):")
            user_input = input("> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\nShutting down MEDHA AI. Goodbye!")
                break

            print("\n----------------------------------------------------------------------")
            print("[1/7] Analyzing task & constructing dynamic specialist plan...")
            print("[2/7] Spawning dynamic specialist agents & executing parallel subtasks...")
            print("[3/7] Routing inter-agent messages & sharing evidence on message bus...")
            print("[4/7] Extracting discrete verifiable claims into Evidence Ledger...")
            print("[5/7] Conducting independent adversarial audit on each claim...")
            print("[6/7] Routing rejected claims for targeted precision repair...")
            print("[7/7] Synthesizing evidence-grounded final answer...")
            print("----------------------------------------------------------------------\n")

            start_time = time.perf_counter()
            result = await orchestrator.run_task(
                user_input,
                diagnostics_callback=_print_execution_diagnostics,
            )
            duration = round(time.perf_counter() - start_time, 2)

            print("======================================================================")
            print("                         FINAL SYNTHESIS                              ")
            print("======================================================================\n")
            print(_answer_for_console(result.final_answer, full_report))
            print("\n======================================================================")
            print("                       EXECUTION SUMMARY                              ")
            print("======================================================================")
            print(f"Execution ID:          {result.execution_id}")
            print(f"Execution Status:      {result.status}")
            print(f"Total Duration:        {duration}s")

            rep = result.verification_report
            print(f"Specialist Agents:     {len(result.generated_agents)}")
            for agent in result.generated_agents:
                print(f"  - {agent['role']} ({agent['status']})")

            print(f"\nClaims Audited:        {rep.get('total_claims', 0)}")
            print(f"Claims Supported:      {len(rep.get('accepted_claims', []))}")
            print(f"Claims Rejected:       {len(rep.get('rejected_claims', []))}")
            print(f"Claims Unverifiable:   {len(rep.get('unverifiable_claims', []))}")
            print(f"Claim Support Rate:    {_format_rate(rep.get('support_rate', 0.0))}")
            print(f"Repair Cycles:         {len(result.repair_history)}")
            print(f"Repair Success Rate:   {_format_rate(rep.get('repair_success_rate', 0.0))}")

            if result.execution_status_reasons:
                reasons = ", ".join(_safe_diagnostic_label(reason) for reason in result.execution_status_reasons)
                print(f"Execution status reason(s): {reasons}")

            if result.unresolved_questions:
                print("\nUnresolved Open Risks / Caveats:")
                for q in result.unresolved_questions:
                    print(f"  * {q}")

            print("======================================================================\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\nOperation interrupted by user. Gracefully shutting down...")
            break
        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")


async def run_single_task(
    task_text: str,
    full_report: bool = False,
    documents: Optional[List[str]] = None,
    source_urls: Optional[List[str]] = None,
    persistent: bool = False,
):
    """Run a single task passed via command line."""
    validate_settings()
    storage_mode = "persistent" if persistent else settings.MEDHA_STORAGE_MODE
    orchestrator = MedhaOrchestrator(storage_mode=storage_mode)
    print("Executing MEDHA task...\n")
    result = await orchestrator.run_task(
        task_text,
        documents=documents,
        source_urls=source_urls,
        storage_mode=storage_mode,
        diagnostics_callback=_print_execution_diagnostics,
    )
    print(_answer_for_console(result.final_answer, full_report))
    print("\n--- Summary ---")
    print(f"Execution ID: {result.execution_id} | Duration: {result.duration_seconds}s | Status: {result.status}")
    print(f"Support Rate: {_format_rate(result.verification_report.get('support_rate', 0.0))}")
    if result.execution_status_reasons:
        reasons = ", ".join(_safe_diagnostic_label(reason) for reason in result.execution_status_reasons)
        print(f"Execution status reason(s): {reasons}")
    if not full_report and result.storage_mode == "persistent":
        print(f"Full report: logs/execution_{result.execution_id}.json (or rerun with --full)")
    elif not full_report:
        if result.cleanup_verified:
            print("Ephemeral Mode: workspace cleanup verified after result delivery.")
        else:
            print(f"Ephemeral Mode cleanup was not verified: {result.cleanup_error or 'unknown cleanup error'}")


async def run_benchmark_cli(
    max_cases: Optional[int] = None,
    repetitions: int = 1,
    experiment_id: Optional[str] = None,
    output_directory: Optional[Path] = None,
    persistent_authorized: bool = False,
):
    """Run the explicit four-arm benchmark experiment."""
    validate_settings()
    from config import settings

    if settings.VERIFICATION_MODE != "strict":
        print("Benchmark requires VERIFICATION_MODE=strict; set it in .env and rerun.")
        sys.exit(2)
    if not persistent_authorized:
        print("Benchmark exports require explicit persistence authorization; rerun with --persistent.")
        sys.exit(2)
    print("\nStarting MEDHA four-arm benchmark experiment...")
    evaluator = BenchmarkEvaluator()
    with storage_mode_context("persistent"):
        result = await evaluator.run_experiment(
            max_cases=max_cases,
            repetitions=repetitions,
            experiment_id=experiment_id,
            output_directory=output_directory,
        )
    print(json.dumps({
        "experiment_id": result.experiment.experiment_id,
        "aggregate_results": result.aggregate_results,
        "actual_experiment_cost": result.actual_experiment_cost,
        "exports": result.export_paths,
    }, indent=2))


def main():
    """CLI Entry point with flag parsing."""
    # Windows consoles default to cp1252, which cannot print many model-generated characters.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="MEDHA: Dynamic Multi-Agent Evidence-Driven AI System"
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        help="Execute a specific task directly without entering interactive mode",
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        action="store_true",
        help="Run the four-arm frozen-plan benchmark experiment",
    )
    parser.add_argument("--benchmark-cases", type=int, help="Limit the number of dataset cases")
    parser.add_argument("--benchmark-repetitions", type=int, default=1, help="Positive repetitions per case")
    parser.add_argument("--experiment-id", type=str, help="Safe explicit ID used in benchmark export filenames")
    parser.add_argument("--benchmark-output", type=Path, help="Output directory for experiment JSON and CSV")
    parser.add_argument(
        "--baseline",
        type=str,
        help="Run a task through the single-agent baseline directly",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Start the MEDHA MCP server over stdio",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print the full verification report instead of only the direct answer",
    )
    parser.add_argument(
        "--persistent",
        action="store_true",
        help="Explicitly retain execution records (required for benchmark exports)",
    )
    parser.add_argument("--document", action="append", dest="documents", help="Local PDF/DOCX/text source; may be repeated")
    parser.add_argument("--source-url", action="append", dest="source_urls", help="Approved HTTPS source URL; may be repeated")

    args = parser.parse_args()

    if args.benchmark_repetitions < 1:
        parser.error("--benchmark-repetitions must be positive")
    if args.benchmark_cases is not None and args.benchmark_cases < 1:
        parser.error("--benchmark-cases must be positive when supplied")
    if args.experiment_id and (
        not args.experiment_id[0].isalnum()
        or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in args.experiment_id)
        or ".." in args.experiment_id
        or len(args.experiment_id) > 128
    ):
        parser.error("--experiment-id must be a safe filename component (letters, digits, dot, underscore, hyphen)")
    if args.benchmark_output is not None:
        if ".." in args.benchmark_output.parts:
            parser.error("--benchmark-output cannot contain parent-directory traversal")
        if not args.benchmark_output.is_absolute():
            args.benchmark_output = BENCHMARK_DIR / args.benchmark_output
        try:
            args.benchmark_output.resolve().relative_to(BENCHMARK_DIR.resolve())
        except ValueError:
            parser.error("--benchmark-output must remain inside the benchmark output directory")
    if not args.benchmark and any((args.benchmark_cases is not None, args.experiment_id, args.benchmark_output is not None, args.benchmark_repetitions != 1)):
        parser.error("benchmark options require --benchmark")
    if (args.documents or args.source_urls) and not args.task:
        parser.error("--document and --source-url require --task")
    if args.mcp and args.persistent:
        parser.error("For MCP Persistent Mode, set MEDHA_STORAGE_MODE=persistent before starting the server.")

    startup_mode = "persistent" if args.persistent else settings.MEDHA_STORAGE_MODE
    cleanup_report = cleanup_stale_workspaces(startup_mode)
    if cleanup_report["failed"]:
        logger.warning("Startup cleanup could not remove %d expired execution workspace(s).", cleanup_report["failed"])

    if args.mcp:
        from mcp_server.server import main as mcp_main
        mcp_main()
    elif args.benchmark:
        asyncio.run(run_benchmark_cli(
            max_cases=args.benchmark_cases,
            repetitions=args.benchmark_repetitions,
            experiment_id=args.experiment_id,
            output_directory=args.benchmark_output,
            persistent_authorized=args.persistent,
        ))
    elif args.baseline:
        validate_settings()
        baseline = SingleAgentBaseline(storage_mode="persistent" if args.persistent else settings.MEDHA_STORAGE_MODE)
        res = asyncio.run(baseline.run(args.baseline))
        print(_console_text(res.answer))
    elif args.task:
        asyncio.run(run_single_task(
            args.task,
            full_report=args.full,
            documents=args.documents,
            source_urls=args.source_urls,
            persistent=args.persistent,
        ))
    else:
        asyncio.run(run_interactive_session(full_report=args.full, persistent=args.persistent))


if __name__ == "__main__":
    main()
