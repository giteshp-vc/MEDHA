import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.baseline import SingleAgentBaseline
from benchmark.medha_runner import MedhaBenchmarkRunner
from benchmark.metrics import (
    SystemMetrics,
    TaskEvaluationResult,
    compute_system_metrics,
)
from config.settings import BENCHMARK_DIR
from utils.logger import get_logger

logger = get_logger("benchmark_evaluator")


def _format_pct(val: Any) -> str:
    """Safely format rate as percentage, handling both fraction (0.0-1.0) and percentage (0.0-100.0)."""
    if val is None:
        return "N/A"
    try:
        f = float(val or 0.0)
        pct = f * 100.0 if (0.0 < f <= 1.0) else f
        return f"{pct:.1f}%"
    except (ValueError, TypeError):
        return "N/A"


class BenchmarkEvaluator:
    """Evaluates and compares Mode A (Single-Agent Baseline) vs Mode B (MEDHA Multi-Agent)."""

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        baseline_runner: Optional[SingleAgentBaseline] = None,
        medha_runner: Optional[MedhaBenchmarkRunner] = None,
    ):
        self.dataset_path = dataset_path or (BENCHMARK_DIR / "benchmark_dataset.json")
        self.baseline_runner = baseline_runner or SingleAgentBaseline()
        self.medha_runner = medha_runner or MedhaBenchmarkRunner()

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load benchmark dataset test cases."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found at {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_answer(
        self,
        task_id: str,
        domain: str,
        mode: str,
        prompt: str,
        answer: str,
        duration: float,
        token_usage: Dict[str, Any],
        reference_criteria: List[str],
        known_pitfalls: List[str],
        verification_stats: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> TaskEvaluationResult:
        """Evaluate a generated answer against reference criteria and known pitfalls."""
        matched: List[str] = []
        missed: List[str] = []
        found_pitfalls: List[str] = []

        answer_lower = answer.lower()

        # Evaluate coverage of reference criteria using concept extraction
        for criterion in reference_criteria:
            words = [w.strip(".,;:()[]\"'").lower() for w in criterion.split() if len(w) > 4]
            if words and sum(1 for w in words if w in answer_lower) >= max(1, len(words) // 2):
                matched.append(criterion)
            else:
                missed.append(criterion)

        # Check for known pitfalls
        for pitfall in known_pitfalls:
            p_words = [w.strip(".,;:()[]\"'").lower() for w in pitfall.split() if len(w) > 4]
            if p_words and sum(1 for w in p_words if w in answer_lower) >= max(2, len(p_words) * 0.7):
                found_pitfalls.append(pitfall)

        total_crit = len(reference_criteria)
        score = round(len(matched) / total_crit, 3) if total_crit > 0 else 0.0

        return TaskEvaluationResult(
            task_id=task_id,
            domain=domain,
            mode=mode,
            prompt=prompt,
            answer=answer,
            success=(error is None and len(answer) > 0),
            duration_seconds=duration,
            token_usage=token_usage,
            matched_criteria=matched,
            missed_criteria=missed,
            identified_pitfalls=found_pitfalls,
            criteria_coverage_score=score,
            verification_stats=verification_stats or {},
            error=error,
        )

    async def run_benchmark(
        self,
        max_cases: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute the benchmark suite on both Mode A and Mode B and return comparative results."""
        cases = self.load_dataset()
        if max_cases:
            cases = cases[:max_cases]

        logger.info(f"Starting benchmark evaluation across {len(cases)} test cases...")

        baseline_results: List[TaskEvaluationResult] = []
        medha_results: List[TaskEvaluationResult] = []

        for idx, case in enumerate(cases, 1):
            task_id = case["id"]
            domain = case["domain"]
            prompt = case["prompt"]
            ref_crit = case.get("reference_criteria", [])
            pitfalls = case.get("known_pitfalls", [])

            logger.info(f"[{idx}/{len(cases)}] Benchmarking: {task_id} ({domain})")

            # --- Run Mode A: Single-Agent Baseline ---
            logger.info(f"Running Mode A (Baseline) for {task_id}...")
            b_res = await self.baseline_runner.run(prompt)
            eval_b = self.evaluate_answer(
                task_id=task_id,
                domain=domain,
                mode="BASELINE",
                prompt=prompt,
                answer=b_res.answer,
                duration=b_res.duration_seconds,
                token_usage=b_res.token_usage,
                reference_criteria=ref_crit,
                known_pitfalls=pitfalls,
                error=b_res.error,
            )
            baseline_results.append(eval_b)

            # --- Run Mode B: MEDHA Multi-Agent ---
            logger.info(f"Running Mode B (MEDHA) for {task_id}...")
            m_res = await self.medha_runner.run(prompt)
            v_rep = m_res.verification_report or {}
            v_stats = {
                "total_claims": v_rep.get("total_claims", len(m_res.claims)),
                "accepted_claims_count": len(v_rep.get("accepted_claims", [])),
                "rejected_claims_count": len(v_rep.get("rejected_claims", [])),
                "support_rate": v_rep.get("support_rate"),
                "unsupported_rate": v_rep.get("unsupported_rate"),
                "repair_success_rate": v_rep.get("repair_success_rate", 0.0),
            }
            eval_m = self.evaluate_answer(
                task_id=task_id,
                domain=domain,
                mode="MEDHA",
                prompt=prompt,
                answer=m_res.final_answer,
                duration=m_res.duration_seconds,
                token_usage=m_res.token_usage,
                reference_criteria=ref_crit,
                known_pitfalls=pitfalls,
                verification_stats=v_stats,
                error=m_res.error,
            )
            medha_results.append(eval_m)

        # Compute aggregate metrics
        baseline_metrics = compute_system_metrics("Single-Agent Baseline (Mode A)", baseline_results)
        medha_metrics = compute_system_metrics("MEDHA Multi-Agent (Mode B)", medha_results)

        comparison_data = {
            "baseline_metrics": baseline_metrics.model_dump(),
            "medha_metrics": medha_metrics.model_dump(),
            "baseline_tasks": [r.model_dump() for r in baseline_results],
            "medha_tasks": [r.model_dump() for r in medha_results],
        }

        # Save artifacts
        self.save_json_report(comparison_data)
        self.save_csv_report(baseline_results, medha_results)

        return comparison_data

    def save_json_report(self, data: Dict[str, Any], filepath: Optional[Path] = None) -> Path:
        """Export benchmark comparison report to JSON."""
        target = filepath or (BENCHMARK_DIR / "benchmark_report.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved benchmark JSON report to {target}")
        return target

    def save_csv_report(
        self,
        baseline_results: List[TaskEvaluationResult],
        medha_results: List[TaskEvaluationResult],
        filepath: Optional[Path] = None,
    ) -> Path:
        """Export side-by-side benchmark task runs to CSV."""
        target = filepath or (BENCHMARK_DIR / "benchmark_results.csv")
        target.parent.mkdir(parents=True, exist_ok=True)

        headers = [
            "task_id",
            "domain",
            "mode",
            "success",
            "duration_seconds",
            "total_tokens",
            "criteria_coverage_score",
            "matched_criteria_count",
            "missed_criteria_count",
            "identified_pitfalls_count",
            "claim_support_rate",
        ]

        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for r in baseline_results + medha_results:
                writer.writerow([
                    r.task_id,
                    r.domain,
                    r.mode,
                    r.success,
                    r.duration_seconds,
                    r.token_usage.get("total_tokens", 0),
                    r.criteria_coverage_score,
                    len(r.matched_criteria),
                    len(r.missed_criteria),
                    len(r.identified_pitfalls),
                    r.verification_stats.get("support_rate", "N/A"),
                ])

        logger.info(f"Saved benchmark CSV report to {target}")
        return target

    @staticmethod
    def generate_markdown_summary(data: Dict[str, Any]) -> str:
        """Generate human-readable side-by-side comparative markdown summary."""
        bm = data["baseline_metrics"]
        mm = data["medha_metrics"]

        md = [
            "# MEDHA vs Single-Agent Baseline Quantitative Benchmark\n\n",
            "## Summary Metrics Comparison\n\n",
            "| Metric | Single-Agent Baseline (Mode A) | MEDHA Multi-Agent (Mode B) | Impact / Delta |\n",
            "| :--- | :--- | :--- | :--- |\n",
            f"| **Task Completion Rate** | {bm['task_completion_rate']*100:.1f}% | {mm['task_completion_rate']*100:.1f}% | {round((mm['task_completion_rate'] - bm['task_completion_rate'])*100, 1):+0.1f}% |\n",
            f"| **Reference Criteria Coverage** | {bm['reference_criteria_coverage']*100:.1f}% | {mm['reference_criteria_coverage']*100:.1f}% | {round((mm['reference_criteria_coverage'] - bm['reference_criteria_coverage'])*100, 1):+0.1f}% |\n",
            f"| **Claim Support Rate** | N/A (unverified) | {_format_pct(mm['claim_support_rate'])} | Audited & Traceable |\n",
            f"| **Unsupported Claim Rate** | N/A (unverified) | {_format_pct(mm['unsupported_claim_rate'])} | Filtered / Repaired |\n",
            f"| **Reviewer Rejection Rate** | 0.0% (no review) | {_format_pct(mm['reviewer_rejection_rate'])} | Active Scrutiny |\n",
            f"| **Repair Success Rate** | N/A | {_format_pct(mm['repair_success_rate'])} | Precision Fixes |\n",
            f"| **Average Latency** | {bm['avg_latency_seconds']:.2f}s | {mm['avg_latency_seconds']:.2f}s | {round(mm['avg_latency_seconds'] - bm['avg_latency_seconds'], 2):+0.2f}s |\n",
            f"| **Avg Tokens per Task** | {bm['avg_tokens_per_task']:.0f} | {mm['avg_tokens_per_task']:.0f} | {round(mm['avg_tokens_per_task'] - bm['avg_tokens_per_task'], 0):+0.0f} |\n",
            f"| **Estimated Cost (USD)** | ${bm['estimated_cost_usd']:.4f} | ${mm['estimated_cost_usd']:.4f} | +${round(mm['estimated_cost_usd'] - bm['estimated_cost_usd'], 4):.4f} |\n",
        ]
        return "".join(md)
