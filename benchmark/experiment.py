"""Typed records and deterministic scheduling for four-arm benchmark experiments."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from models.frozen_plan import FrozenTaskPlan
from models.provider_call import AccountingSnapshot, ProviderCallRecord
from models.schemas import BaseSchema

WILLIAMS_SEQUENCES = (
    ("BASELINE", "INDEPENDENT_PARALLEL", "FULL_MEDHA", "DEPENDENCY_AWARE"),
    ("INDEPENDENT_PARALLEL", "DEPENDENCY_AWARE", "BASELINE", "FULL_MEDHA"),
    ("DEPENDENCY_AWARE", "FULL_MEDHA", "INDEPENDENT_PARALLEL", "BASELINE"),
    ("FULL_MEDHA", "BASELINE", "DEPENDENCY_AWARE", "INDEPENDENT_PARALLEL"),
)
SCORING_VERSION = "medha-heuristic-v1"


class BenchmarkConfigurationSnapshot(BaseSchema):
    """Versioned application and provider configuration captured for an experiment."""

    model_config = ConfigDict(frozen=True)

    application_version: Optional[str] = None
    git_commit: Optional[str] = None
    git_worktree_dirty: Optional[bool] = None
    python_version: Optional[str] = None
    dependency_versions: Dict[str, Optional[str]] = Field(default_factory=dict)
    provider_configuration: Dict[str, Any] = Field(default_factory=dict)
    pricing_configuration: Dict[str, Any] = Field(default_factory=dict)
    runtime_configuration: Dict[str, Any] = Field(default_factory=dict)
    effective_provider_calls: List[Dict[str, Any]] = Field(default_factory=list)
    treatment_configuration: Dict[str, Any] = Field(default_factory=dict)
    planner_configuration: Dict[str, Any] = Field(default_factory=dict)
    verification_configuration: Dict[str, Any] = Field(default_factory=dict)
    synthesis_configuration: Dict[str, Any] = Field(default_factory=dict)
    scoring_version: str = SCORING_VERSION
    dataset_hash: Optional[str] = None
    plan_hash: Optional[str] = None
    compatible: bool = False
    compatibility_differences: Dict[str, Any] = Field(default_factory=dict)


class MetricMeasurement(BaseSchema):
    """A score and its explicit denominator/missingness information."""

    value: Optional[float] = None
    eligible_sample_count: int = 0
    attempt_count: int = 0
    missing_count: int = 0
    missingness_reason: Optional[str] = None


class BenchmarkArmResult(BaseSchema):
    """Serializable outcome, metrics, telemetry, and costs for one arm trial."""

    experiment_id: str
    task_block_id: str
    trial_id: str
    execution_id: Optional[str] = None
    arm_id: str
    plan_hash: Optional[str] = None
    task_id: str
    repetition: int
    planned_order: List[str] = Field(default_factory=list)
    actual_order: List[str] = Field(default_factory=list)
    start_timestamp: str
    end_timestamp: str
    duration_seconds: float
    status: str
    success: bool
    quality_eligible: bool
    error_category: Optional[str] = None
    answer: str = ""
    partial_answer: Optional[str] = None
    metrics: Dict[str, MetricMeasurement] = Field(default_factory=dict)
    execution_metrics: Dict[str, Optional[float]] = Field(default_factory=dict)
    verification_metrics: Dict[str, Optional[float]] = Field(default_factory=dict)
    provider_call_records: List[ProviderCallRecord] = Field(default_factory=list)
    accounting: Optional[AccountingSnapshot] = None
    cost: Dict[str, Any] = Field(default_factory=dict)
    plan_preparation_cost: Dict[str, Any] = Field(default_factory=dict)
    trial_execution_cost: Dict[str, Any] = Field(default_factory=dict)
    end_to_end_cost: Dict[str, Any] = Field(default_factory=dict)
    telemetry_complete: bool = False
    dropped_provider_call_records: int = 0
    configuration: Optional[BenchmarkConfigurationSnapshot] = None


class BenchmarkSchedule(BaseSchema):
    task_block_id: str
    repetition: int
    sequence_index: int
    planned_order: List[str]
    actual_order: List[str] = Field(default_factory=list)
    interrupted: bool = False


class BenchmarkTrial(BaseSchema):
    """One unique arm execution attempt within a task-block repetition."""

    trial_id: str
    task_block_id: str
    repetition: int
    arm_id: str
    planned_order: List[str]
    actual_order: List[str]
    result: BenchmarkArmResult


class BenchmarkTaskBlock(BaseSchema):
    task_block_id: str
    task_id: str
    task_text: str
    dataset_case_id: Optional[str] = None
    dataset_hash: Optional[str] = None
    plan_hash: Optional[str] = None
    frozen_plan: Optional[Dict[str, Any]] = None
    plan_preparation_duration_seconds: Optional[float] = None
    plan_preparation_accounting: Optional[AccountingSnapshot] = None
    plan_preparation_error_category: Optional[str] = None
    schedules: List[BenchmarkSchedule] = Field(default_factory=list)
    trials: List[BenchmarkTrial] = Field(default_factory=list)


class BenchmarkExperiment(BaseSchema):
    experiment_id: str
    created_at: str
    repetitions: int
    dataset_hash: Optional[str] = None
    configuration_snapshots: Dict[str, BenchmarkConfigurationSnapshot] = Field(default_factory=dict)
    task_blocks: List[BenchmarkTaskBlock] = Field(default_factory=list)
    schedule_policy: str = "deterministic_williams_4_sequence_cycle"
    schedule_sequences_used: List[int] = Field(default_factory=list)
    schedule_cycle_complete: bool = False
    scoring_policy: Dict[str, Any] = Field(default_factory=lambda: {
        "scoring_version": SCORING_VERSION,
        "answer_completeness": "reference-criteria lexical coverage proxy",
        "pitfall_avoidance": "known-pitfall lexical proxy",
        "uncertainty_handling": "language-marker proxy",
        "evidence_quality": "language-marker proxy, not factual verification",
        "factual_correctness": None,
        "human_scoring": "not supplied; no blinded human score is fabricated",
        "agent_agreement_is_evidence": False,
        "statistical_significance_claimed": False,
    })
    resource_accounting_policy: str = (
        "AccountingSnapshot is authoritative for token usage; provider-call records are used only "
        "for request, retry, failure, duration, and dropped-record telemetry."
    )
    statistical_significance_claimed: bool = False


class BenchmarkEvaluationResult(BaseSchema):
    experiment: BenchmarkExperiment
    aggregate_results: Dict[str, Any] = Field(default_factory=dict)
    actual_experiment_cost: Dict[str, Any] = Field(default_factory=dict)
    export_paths: Dict[str, str] = Field(default_factory=dict)


def balanced_sequence(
    task_block_index: int,
    repetition_index: int,
    repetitions_per_block: int = 1,
) -> tuple[str, ...]:
    """Assign sequence slots in block-major order, without randomization."""
    if task_block_index < 0 or repetition_index < 0 or repetitions_per_block < 1:
        raise ValueError("Schedule indices must be non-negative.")
    slot = task_block_index * repetitions_per_block + repetition_index
    return WILLIAMS_SEQUENCES[slot % len(WILLIAMS_SEQUENCES)]


def canonical_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for JSON-compatible benchmark data."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()