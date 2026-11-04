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
    duration_seconds: float
    token_usage: Dict[str, Any] = Field(default_factory=dict)
    matched_criteria: List[str] = Field(default_factory=list)
    missed_criteria: List[str] = Field(default_factory=list)
    identified_pitfalls: List[str] = Field(default_factory=list)
    criteria_coverage_score: float = 0.0
    verification_stats: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


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

    total_tokens = sum(r.token_usage.get("total_tokens", 0) for r in results)
    avg_tokens = round(total_tokens / total, 1)
    total_reqs = sum(r.token_usage.get("total_requests", 1) for r in results)

    coverage_scores = [r.criteria_coverage_score for r in results]
    avg_coverage = round(sum(coverage_scores) / total, 3) if coverage_scores else 0.0

    # Verification specific rates (relevant for MEDHA)
    all_support_rates = [
        r.verification_stats.get("support_rate")
        for r in results if r.verification_stats and r.verification_stats.get("support_rate") is not None
    ]
    avg_support_rate = round(sum(all_support_rates) / len(all_support_rates), 3) if all_support_rates else None

    all_unsupported_rates = [
        r.verification_stats.get("unsupported_rate")
        for r in results if r.verification_stats and r.verification_stats.get("unsupported_rate") is not None
    ]
    avg_unsupported_rate = round(sum(all_unsupported_rates) / len(all_unsupported_rates), 3) if all_unsupported_rates else None

    all_repair_rates = [
        r.verification_stats.get("repair_success_rate", 0.0)
        for r in results if r.verification_stats
    ]
    avg_repair_success = round(sum(all_repair_rates) / len(all_repair_rates), 3) if all_repair_rates else 0.0

    all_rejections = [
        r.verification_stats.get("rejected_claims_count", 0)
        for r in results if r.verification_stats
    ]
    total_claims = sum(
        r.verification_stats.get("total_claims", 0)
        for r in results if r.verification_stats
    )
    rejection_rate = round(sum(all_rejections) / total_claims, 3) if total_claims > 0 else 0.0

    # Pricing estimate based on standard Gemini 2.5 Flash rates (~$0.30 per 1M tokens)
    estimated_cost = round((total_tokens / 1_000_000.0) * 0.30, 5)

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
        estimated_cost_usd=estimated_cost,
    )

