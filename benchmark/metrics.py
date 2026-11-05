import statistics
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from models.schemas import BaseSchema


class TaskEvaluationResult(BaseSchema):
    """Detailed evaluation score of a single benchmark task run."""

    task_id: str
    domain: str
    mode: str  # "BASELINE" or "MEDHA"
    prompt: str
    answer: str
    success: bool
    completion_status: str = "UNKNOWN"
    quality_eligible: Optional[bool] = None
    duration_seconds: float
    token_usage: Dict[str, Any] = Field(default_factory=dict)
    matched_criteria: List[str] = Field(default_factory=list)
    missed_criteria: List[str] = Field(default_factory=list)
    identified_pitfalls: List[str] = Field(default_factory=list)
    criteria_coverage_score: float = 0.0
    verification_stats: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    experiment_id: str = ""
    trial_id: str = ""
    trial_index: int = 0
    execution_order: List[str] = Field(default_factory=list)
    timestamp: str = ""
    configuration: Dict[str, Any] = Field(default_factory=dict)
    answer_metrics: Dict[str, Optional[float]] = Field(default_factory=dict)
    evaluator_reasoning: List[str] = Field(default_factory=list)
    scoring_evidence: Dict[str, Any] = Field(default_factory=dict)
    cost: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfiguration(BaseSchema):
    """Comparable configuration recorded for each benchmark system."""

    provider: str = "unknown"
    model: str = "unknown"
    model_version: str = "unknown"
    temperature: Optional[float] = None
    input_token_limit: Optional[int] = None
    output_token_limit: Optional[int] = None
    task_token_budget: Optional[int] = None
    call_budget: Optional[int] = None

    def compatibility_differences(self, other: "BenchmarkConfiguration") -> Dict[str, Any]:
        differences = {}
        for field_name in type(self).model_fields:
            left = getattr(self, field_name)
            right = getattr(other, field_name)
            if left != right:
                differences[field_name] = {"left": left, "right": right}
        return differences


class SystemMetrics(BaseSchema):
    """Aggregated quantitative performance metrics for an execution mode."""

    mode: str
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    task_completion_rate: float = 0.0
    failure_rate: float = 0.0
    avg_latency_seconds: float = 0.0
    min_latency_seconds: float = 0.0
    max_latency_seconds: float = 0.0
    total_tokens: int = 0
    avg_tokens_per_task: float = 0.0
    total_api_requests: int = 0
    claim_support_rate: Optional[float] = None
    unsupported_claim_rate: Optional[float] = None
    reviewer_rejection_rate: float = 0.0
    repair_success_rate: float = 0.0
    reference_criteria_coverage: float = 0.0
    estimated_cost_usd: float = 0.0
    cost_status: str = "unknown"
    failure_count: int = 0
    latency_variance_seconds: Optional[float] = None
    answer_metrics: Dict[str, Optional[float]] = Field(default_factory=dict)
    metric_sample_counts: Dict[str, int] = Field(default_factory=dict)
    metric_variances: Dict[str, Optional[float]] = Field(default_factory=dict)


def compute_system_metrics(
    mode_name: str,
    results: List[TaskEvaluationResult],
) -> SystemMetrics:
    """Compute aggregate statistical metrics across task results."""
    total = len(results)
    if total == 0:
        return SystemMetrics(mode=mode_name)

    completed = sum(1 for r in results if r.success)
    failed = total - completed
    latencies = [r.duration_seconds for r in results]
    avg_latency = round(sum(latencies) / total, 3) if latencies else 0.0
    min_latency = round(min(latencies), 3) if latencies else 0.0
    max_latency = round(max(latencies), 3) if latencies else 0.0

    total_tokens = sum(
        r.token_usage.get("accounted_total_tokens", r.token_usage.get("total_tokens", 0))
        for r in results
    )
    avg_tokens = round(total_tokens / total, 1)
    total_reqs = sum(r.token_usage.get("total_requests", 1) for r in results)

    quality_results = [
        result for result in results
        if (
            result.quality_eligible
            if result.quality_eligible is not None
            else result.success
        )
    ]
    coverage_scores = [r.criteria_coverage_score for r in quality_results]
    avg_coverage = round(sum(coverage_scores) / len(coverage_scores), 3) if coverage_scores else 0.0

    # Verification specific rates (relevant for MEDHA)
    all_support_rates = [
        r.verification_stats.get("support_rate")
        for r in quality_results if r.verification_stats and r.verification_stats.get("support_rate") is not None
    ]
    avg_support_rate = round(sum(all_support_rates) / len(all_support_rates), 3) if all_support_rates else None

    all_unsupported_rates = [
        r.verification_stats.get("unsupported_rate")
        for r in quality_results if r.verification_stats and r.verification_stats.get("unsupported_rate") is not None
    ]
    avg_unsupported_rate = round(sum(all_unsupported_rates) / len(all_unsupported_rates), 3) if all_unsupported_rates else None

    all_repair_rates = [
        r.verification_stats.get("repair_success_rate", 0.0)
        for r in quality_results if r.verification_stats
    ]
    avg_repair_success = round(sum(all_repair_rates) / len(all_repair_rates), 3) if all_repair_rates else 0.0

    all_rejections = [
        r.verification_stats.get("rejected_claims_count", 0)
        for r in quality_results if r.verification_stats
    ]
    total_claims = sum(
        r.verification_stats.get("total_claims", 0)
        for r in quality_results if r.verification_stats
    )
    rejection_rate = round(sum(all_rejections) / total_claims, 3) if total_claims > 0 else 0.0

    cost_values = [
        result.cost.get("amount_usd")
        for result in results
        if result.cost.get("amount_usd") is not None
        and result.cost.get("status") in ("actual", "estimated", "mixed")
    ]
    cost_statuses = {result.cost.get("status") for result in results if result.cost.get("status")}
    cost_status = (
        "actual" if cost_statuses == {"actual"}
        else "estimated" if cost_statuses and cost_statuses <= {"actual", "estimated"}
        else "mixed" if "mixed" in cost_statuses
        else "unknown"
    )
    total_cost = round(sum(cost_values), 5) if cost_values else 0.0

    metric_names = {name for result in results for name in result.answer_metrics}
    answer_metrics: Dict[str, Optional[float]] = {}
    metric_sample_counts: Dict[str, int] = {}
    metric_variances: Dict[str, Optional[float]] = {}
    for name in metric_names:
        values = [
            result.answer_metrics[name]
            for result in quality_results
            if result.answer_metrics.get(name) is not None
        ]
        metric_sample_counts[name] = len(values)
        answer_metrics[name] = round(sum(values) / len(values), 3) if values else None
        metric_variances[name] = round(statistics.pvariance(values), 6) if len(values) > 1 else None
    latency_variance = round(statistics.pvariance(latencies), 6) if len(latencies) > 1 else None

    return SystemMetrics(
        mode=mode_name,
        total_tasks=total,
        completed_tasks=completed,
        failed_tasks=failed,
        task_completion_rate=round(completed / total, 3),
        failure_rate=round(failed / total, 3),
        avg_latency_seconds=avg_latency,
        min_latency_seconds=min_latency,
        max_latency_seconds=max_latency,
        total_tokens=total_tokens,
        avg_tokens_per_task=avg_tokens,
        total_api_requests=total_reqs,
        claim_support_rate=avg_support_rate,
        unsupported_claim_rate=avg_unsupported_rate,
        reviewer_rejection_rate=rejection_rate,
        repair_success_rate=avg_repair_success,
        reference_criteria_coverage=avg_coverage,
        estimated_cost_usd=total_cost,
        cost_status=cost_status,
        failure_count=failed,
        latency_variance_seconds=latency_variance,
        answer_metrics=answer_metrics,
        metric_sample_counts=metric_sample_counts,
        metric_variances=metric_variances,
    )

