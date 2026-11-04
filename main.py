import argparse
import asyncio
import sys
import time
from typing import Any, Optional

from benchmark.baseline import SingleAgentBaseline
from benchmark.evaluator import BenchmarkEvaluator
from config.settings import LLM_PROVIDER, validate_settings
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


async def run_benchmark_cli():
    """Run comparative benchmark against single-agent baseline."""
    validate_settings()
    print("\nStarting MEDHA Quantitative Benchmark against Single-Agent Baseline...")
    evaluator = BenchmarkEvaluator()
    data = await evaluator.run_benchmark()
    summary_md = BenchmarkEvaluator.generate_markdown_summary(data)
    print("\n" + summary_md)
    print("\nDetailed results saved to benchmark/benchmark_report.json and benchmark/benchmark_results.csv")


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
        help="Run the quantitative benchmark suite comparing MEDHA to single-agent baseline",
    )
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

    if args.mcp:
        from mcp_server.server import main as mcp_main
        mcp_main()
    elif args.benchmark:
        asyncio.run(run_benchmark_cli())
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
