import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

from benchmark.baseline import SingleAgentBaseline
from benchmark.evaluator import BenchmarkEvaluator
from config.settings import BENCHMARK_DIR, LLM_PROVIDER, validate_settings
from core.orchestrator import MedhaOrchestrator
from utils.logger import get_logger

logger = get_logger("main")

BANNER = """
======================================================================
                              MEDHA AI
======================================================================
               Multi-Agent Evidence-Driven Intelligence
         Problem Statement 02 - Chain of Command (SEED 2026)
======================================================================
"""


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


async def run_interactive_session():
    """Run interactive terminal session for MEDHA."""
    print(BANNER)
    try:
        validate_settings()
        print(f"[OK] Configuration and {LLM_PROVIDER.upper()} API credentials verified.")
    except Exception as e:
        print(f"[ERROR] Configuration error: {e}")
        sys.exit(1)

    orchestrator = MedhaOrchestrator()

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
            result = await orchestrator.run_task(user_input)
            duration = round(time.perf_counter() - start_time, 2)

            print("======================================================================")
            print("                         FINAL SYNTHESIS                              ")
            print("======================================================================\n")
            print(result.final_answer)
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

            tokens = result.token_usage
            if tokens:
                print(f"\nToken Consumption:     {tokens.get('total_tokens', 0):,} total")
                print(f"  Prompt Tokens:       {tokens.get('prompt_tokens', 0):,}")
                print(f"  Candidate Tokens:    {tokens.get('candidate_tokens', 0):,}")
                print(f"  API Requests:        {tokens.get('successful_requests', 0)}")

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


async def run_single_task(task_text: str):
    """Run a single task passed via command line."""
    validate_settings()
    orchestrator = MedhaOrchestrator()
    print(f"Executing MEDHA for task: '{task_text}'\n")
    result = await orchestrator.run_task(task_text)
    print(result.final_answer)
    print("\n--- Summary ---")
    print(f"Execution ID: {result.execution_id} | Duration: {result.duration_seconds}s | Status: {result.status}")
    print(f"Support Rate: {_format_rate(result.verification_report.get('support_rate', 0.0))}")


async def run_benchmark_cli(
    max_cases: Optional[int] = None,
    repetitions: int = 1,
    experiment_id: Optional[str] = None,
    output_directory: Optional[Path] = None,
):
    """Run the explicit four-arm benchmark experiment."""
    validate_settings()
    print("\nStarting MEDHA four-arm benchmark experiment...")
    evaluator = BenchmarkEvaluator()
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

    if args.mcp:
        from mcp_server.server import main as mcp_main
        mcp_main()
    elif args.benchmark:
        asyncio.run(run_benchmark_cli(
            max_cases=args.benchmark_cases,
            repetitions=args.benchmark_repetitions,
            experiment_id=args.experiment_id,
            output_directory=args.benchmark_output,
        ))
    elif args.baseline:
        validate_settings()
        baseline = SingleAgentBaseline()
        res = asyncio.run(baseline.run(args.baseline))
        print(res.answer)
    elif args.task:
        asyncio.run(run_single_task(args.task))
    else:
        asyncio.run(run_interactive_session())


if __name__ == "__main__":
    main()
