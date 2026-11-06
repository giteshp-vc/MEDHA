import csv
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.baseline import SingleAgentBaseline
from benchmark.medha_runner import MedhaBenchmarkRunner
from benchmark.metrics import (
    BenchmarkConfiguration,
    SystemMetrics,
    TaskEvaluationResult,
    compute_system_metrics,
)
from config.settings import BENCHMARK_DIR
from utils.logger import get_logger

logger = get_logger("benchmark_evaluator")


class BenchmarkConfigurationMismatchError(ValueError):
    """Raised when baseline and MEDHA are not configured for a comparable run."""

    def __init__(self, differences: Dict[str, Any]):
        self.differences = differences
        super().__init__(
            "Benchmark configurations are incompatible; refusing comparison: "
            + json.dumps(differences, sort_keys=True)
        )


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
        pricing: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.dataset_path = dataset_path or (BENCHMARK_DIR / "benchmark_dataset.json")
        self.baseline_runner = baseline_runner or SingleAgentBaseline()
        self.medha_runner = medha_runner or MedhaBenchmarkRunner()
        self.pricing = pricing or {}

    async def run_experiment(
        self,
        max_cases: Optional[int] = None,
        repetitions: int = 1,
        experiment_id: Optional[str] = None,
        output_directory: Optional[Path] = None,
    ):
        """Run the reproducible four-arm frozen-plan benchmark experiment."""
        from benchmark.four_arm import FourArmBenchmark

        return await FourArmBenchmark(self).run(
            max_cases=max_cases,
            repetitions=repetitions,
            experiment_id=experiment_id,
            output_directory=output_directory,
        )

    @staticmethod
    def _runner_configuration(runner: Any) -> BenchmarkConfiguration:
        raw = {}
        method = getattr(runner, "benchmark_configuration", None)
        if callable(method):
            candidate = method()
            if isinstance(candidate, dict):
                raw = candidate
        return BenchmarkConfiguration.model_validate(raw or {})

    def _check_configurations(self) -> Dict[str, Any]:
        baseline = self._runner_configuration(self.baseline_runner)
        medha = self._runner_configuration(self.medha_runner)
        differences = baseline.compatibility_differences(medha)
        if differences:
            raise BenchmarkConfigurationMismatchError(differences)
        return {"BASELINE": baseline.model_dump(), "MEDHA": medha.model_dump(), "differences": differences}

    def _cost_for(self, configuration: Dict[str, Any], token_usage: Dict[str, Any]) -> Dict[str, Any]:
        provider = configuration.get("provider", "unknown")
        model = configuration.get("model", "unknown")
        price = self.pricing.get(f"{provider}:{model}") or self.pricing.get(provider)
        usage_source = str(token_usage.get("usage_source", "unknown")).lower()
        if price is None:
            return {
                "amount_usd": None,
                "status": "unknown",
                "reason": "No provider/model pricing was configured.",
                "usage_source": usage_source,
            }
        rates = (price.get("input_per_million_usd"), price.get("output_per_million_usd"))
        if any(
            not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or not math.isfinite(rate)
            or rate < 0
            for rate in rates
        ):
            return {
                "amount_usd": None,
                "status": "unknown",
                "reason": "Explicit non-negative input and output pricing is incomplete or invalid.",
                "usage_source": usage_source,
            }
        if usage_source in {"estimated", "mixed"}:
            return {
                "amount_usd": None,
                "status": usage_source,
                "reason": "Token usage does not provide a complete provider-reported input/output split.",
                "usage_source": usage_source,
            }
        if usage_source != "provider":
            return {
                "amount_usd": None,
                "status": "unknown",
                "reason": "Token usage provenance is unknown.",
                "usage_source": usage_source,
            }
        input_tokens = token_usage.get("prompt_tokens")
        output_tokens = token_usage.get("candidate_tokens")
        if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                   for value in (input_tokens, output_tokens)):
            return {
                "amount_usd": None,
                "status": "unknown",
                "reason": "Provider-reported input/output token usage is unavailable.",
                "usage_source": usage_source,
            }
        amount = (
            input_tokens * rates[0]
            + output_tokens * rates[1]
        ) / 1_000_000.0
        return {
            "amount_usd": round(amount, 8),
            "status": "actual",
            "reason": "Explicit provider/model pricing applied to provider-reported usage.",
            "usage_source": usage_source,
        }

    @staticmethod
    def _answer_metrics(
        matched: List[str],
        total_criteria: int,
        pitfalls: List[str],
        total_pitfalls: int,
        answer: str,
        success: bool,
        verification_stats: Dict[str, Any],
        human_scores: Optional[Dict[str, float]] = None,
    ) -> tuple[Dict[str, Optional[float]], List[str]]:
        lower = answer.lower()
        reasoning = ["completeness uses the shared reference-criteria evaluator"]
        scores: Dict[str, Optional[float]] = {
            "answer_completeness": round(len(matched) / total_criteria, 3) if total_criteria else None,
            "task_completion": 1.0 if success else 0.0,
            "pitfall_avoidance": round(1 - len(pitfalls) / total_pitfalls, 3) if total_pitfalls else None,
            "uncertainty_handling": 1.0 if any(
                marker in lower for marker in ("assumption", "uncertain", "depends", "unknown", "limitation")
            ) else 0.0,
            "evidence_quality": 1.0 if any(
                marker in lower for marker in ("evidence", "source", "citation", "benchmark", "reference")
            ) else 0.0,
            "factual_correctness": None,
            "unsupported_claim_rate": verification_stats.get("unsupported_claim_rate"),
        }
        reasoning.extend([
            "pitfall avoidance, uncertainty handling, and evidence quality are automated proxies",
            "factual correctness requires blinded human or independent expert scoring unless supplied",
        ])
        if human_scores:
            for name, value in human_scores.items():
                if name in scores:
                    scores[name] = value
            reasoning.append("human scores were supplied separately from MEDHA's internal reviewer")
        return scores, reasoning

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
        experiment_id: str = "",
        trial_id: str = "",
        trial_index: int = 0,
        execution_order: Optional[List[str]] = None,
        timestamp: str = "",
        configuration: Optional[Dict[str, Any]] = None,
        human_scores: Optional[Dict[str, float]] = None,
        completion_status: str = "UNKNOWN",
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
        completion_status = str(completion_status or "UNKNOWN").upper()
        if error is not None and completion_status == "SUCCESS":
            completion_status = "FAILED"
        elif completion_status == "SUCCESS" and not answer.strip():
            completion_status = "EMPTY_OUTPUT"
        success = completion_status == "SUCCESS"
        answer_metrics, evaluator_reasoning = self._answer_metrics(
            matched, total_crit, found_pitfalls, len(known_pitfalls), answer, success, verification_stats or {}, human_scores
        )
        configuration = configuration or {}

        return TaskEvaluationResult(
            task_id=task_id,
            domain=domain,
            mode=mode,
            prompt=prompt,
            answer=answer,
            success=success,
            completion_status=completion_status,
            quality_eligible=success,
            duration_seconds=duration,
            token_usage=token_usage,
            matched_criteria=matched,
            missed_criteria=missed,
            identified_pitfalls=found_pitfalls,
            criteria_coverage_score=score,
            verification_stats=verification_stats or {},
            error=error,
            experiment_id=experiment_id,
            trial_id=trial_id,
            trial_index=trial_index,
            execution_order=execution_order or [],
            timestamp=timestamp,
            configuration=configuration,
            answer_metrics=answer_metrics,
            evaluator_reasoning=evaluator_reasoning,
            cost=self._cost_for(configuration, token_usage),
            scoring_evidence={
                "matched_criteria": matched,
                "missed_criteria": missed,
                "identified_pitfalls": found_pitfalls,
                "human_scores_supplied": bool(human_scores),
            },
        )

    async def run_benchmark(
        self,
        max_cases: Optional[int] = None,
        repetitions: int = 1,
        experiment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the benchmark suite on both Mode A and Mode B and return comparative results."""
        cases = self.load_dataset()
        if max_cases:
            cases = cases[:max_cases]

        if repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        configurations = self._check_configurations()
        experiment_id = experiment_id or f"exp_{uuid.uuid4().hex[:12]}"
        logger.info(
            f"Starting benchmark experiment {experiment_id} across {len(cases)} test cases "
            f"with {repetitions} counterbalanced repetition(s)..."
        )

        baseline_results: List[TaskEvaluationResult] = []
        medha_results: List[TaskEvaluationResult] = []

        trial_results: List[TaskEvaluationResult] = []
        trial_records: List[Dict[str, Any]] = []
        for idx, case in enumerate(cases, 1):
            task_id = case["id"]
            domain = case["domain"]
            prompt = case["prompt"]
            ref_crit = case.get("reference_criteria", [])
            pitfalls = case.get("known_pitfalls", [])

            for repetition in range(repetitions):
                order = ["BASELINE", "MEDHA"] if repetition % 2 == 0 else ["MEDHA", "BASELINE"]
                trial_id = f"{experiment_id}_{task_id}_{repetition + 1}"
                trial_timestamp = datetime.now(timezone.utc).isoformat()
                trial_records.append({"trial_id": trial_id, "task_id": task_id, "trial_index": repetition + 1, "execution_order": order, "timestamp": trial_timestamp})
                for mode in order:
                    runner = self.baseline_runner if mode == "BASELINE" else self.medha_runner
                    try:
                        raw = await runner.run(prompt)
                        if mode == "BASELINE":
                            answer = raw.answer
                            duration = raw.duration_seconds
                            token_usage = raw.token_usage
                            error = raw.error
                            verification_stats = {}
                            completion_status = (
                                "SUCCESS"
                                if raw.success is True and error is None and answer.strip()
                                else "FAILED" if raw.success is not True or error is not None
                                else "EMPTY_OUTPUT"
                            )
                        else:
                            answer = raw.final_answer
                            duration = raw.duration_seconds
                            token_usage = raw.token_usage
                            error = raw.error
                            reported_status = str(getattr(raw.status, "value", raw.status)).upper()
                            completion_status = (
                                reported_status
                                if reported_status in {
                                    "SUCCESS",
                                    "PARTIAL_FAILURE",
                                    "FAILED",
                                    "CANCELLED",
                                    "INTERRUPTED",
                                }
                                else "UNKNOWN"
                            )
                            if completion_status == "SUCCESS" and not answer.strip():
                                completion_status = "EMPTY_OUTPUT"
                            v_rep = raw.verification_report or {}
                            verification_stats = {
                                "total_claims": v_rep.get("total_claims", len(raw.claims)),
                                "accepted_claims_count": len(v_rep.get("accepted_claims", [])),
                                "rejected_claims_count": len(v_rep.get("rejected_claims", [])),
                                "support_rate": v_rep.get("support_rate"),
                                "unsupported_rate": v_rep.get("unsupported_rate"),
                                "unsupported_claim_rate": v_rep.get("unsupported_rate"),
                                "repair_success_rate": v_rep.get("repair_success_rate", 0.0),
                            }
                    except Exception as exc:
                        answer, duration, token_usage, error, verification_stats = "", 0.0, {}, f"{type(exc).__name__}: {exc}", {}
                        completion_status = "EXCEPTION"
                    evaluated = self.evaluate_answer(
                        task_id=task_id,
                        domain=domain,
                        mode=mode,
                        prompt=prompt,
                        answer=answer,
                        duration=duration,
                        token_usage=token_usage,
                        reference_criteria=ref_crit,
                        known_pitfalls=pitfalls,
                        verification_stats=verification_stats,
                        error=error,
                        experiment_id=experiment_id,
                        trial_id=trial_id,
                        trial_index=repetition + 1,
                        execution_order=order,
                        timestamp=trial_timestamp,
                        configuration=configurations[mode],
                        human_scores=(case.get("human_scores", {}) or {}).get(mode),
                        completion_status=completion_status,
                    )
                    trial_results.append(evaluated)
                    (baseline_results if mode == "BASELINE" else medha_results).append(evaluated)

        # Compute aggregate metrics
        baseline_metrics = compute_system_metrics("Single-Agent Baseline (Mode A)", baseline_results)
        medha_metrics = compute_system_metrics("MEDHA Multi-Agent (Mode B)", medha_results)

        comparison_data = {
            "experiment_id": experiment_id,
            "repetitions": repetitions,
            "configurations": configurations,
            "trial_records": trial_records,
            "evaluation_protocol": {
                "same_evaluator": True,
                "automated_metrics_are_heuristics": True,
                "human_evaluation_supported": True,
                "statistical_significance_claimed": False,
            },
            "baseline_metrics": baseline_metrics.model_dump(),
            "medha_metrics": medha_metrics.model_dump(),
            "trials": [r.model_dump() for r in trial_results],
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
            "experiment_id",
            "trial_id",
            "trial_index",
            "execution_order",
            "task_id",
            "domain",
            "mode",
            "completion_status",
            "quality_eligible",
            "success",
            "duration_seconds",
            "total_tokens",
            "criteria_coverage_score",
            "matched_criteria_count",
            "missed_criteria_count",
            "identified_pitfalls_count",
            "claim_support_rate",
            "answer_completeness",
            "factual_correctness",
            "unsupported_claim_rate",
            "evidence_quality",
            "uncertainty_handling",
            "task_completion",
            "cost_amount_usd",
            "cost_status",
            "cost_usage_source",
            "answer",
            "error",
        ]

        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for r in baseline_results + medha_results:
                row = [
                    r.experiment_id,
                    r.trial_id,
                    r.trial_index,
                    ">".join(r.execution_order),
                    r.task_id,
                    r.domain,
                    r.mode,
                    r.completion_status,
                    r.quality_eligible,
                    r.success,
                    r.duration_seconds,
                    r.token_usage.get("accounted_total_tokens", r.token_usage.get("total_tokens", 0)),
                    r.criteria_coverage_score,
                    len(r.matched_criteria),
                    len(r.missed_criteria),
                    len(r.identified_pitfalls),
                    r.verification_stats.get("support_rate", "N/A"),
                    r.answer_metrics.get("answer_completeness"),
                    r.answer_metrics.get("factual_correctness"),
                    r.answer_metrics.get("unsupported_claim_rate"),
                    r.answer_metrics.get("evidence_quality"),
                    r.answer_metrics.get("uncertainty_handling"),
                    r.answer_metrics.get("task_completion"),
                    r.cost.get("amount_usd"),
                    r.cost.get("status", "unknown"),
                    r.cost.get("usage_source", "unknown"),
                    r.answer,
                    r.error,
                ]
                writer.writerow(["null" if value is None else value for value in row])

        logger.info(f"Saved benchmark CSV report to {target}")
        return target

    @staticmethod
    def generate_markdown_summary(data: Dict[str, Any]) -> str:
        """Generate human-readable side-by-side comparative markdown summary."""
        bm = data["baseline_metrics"]
        mm = data["medha_metrics"]
        def format_cost(metrics: Dict[str, Any]) -> str:
            amount = metrics.get("estimated_cost_usd")
            status = metrics.get("cost_status", "unknown")
            return f"{status}: ${amount:.4f}" if amount is not None else f"{status}: N/A"

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
            f"| **Cost (USD)** | {format_cost(bm)} | {format_cost(mm)} | Not a significance test |\n",
            f"| **Latency Variance** | {bm.get('latency_variance_seconds', 'N/A')} | {mm.get('latency_variance_seconds', 'N/A')} | Descriptive only |\n",
            f"| **Failed Trials** | {bm.get('failure_count', 0)} | {mm.get('failure_count', 0)} | Preserved in trial records |\n",
        ]
        return "".join(md)
