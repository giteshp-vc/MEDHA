from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from models.schemas import BaseSchema
from utils.helpers import generate_id, utcnow_iso


class ProviderCallStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"


class ProviderUsageSource(str, Enum):
    PROVIDER_REPORTED = "PROVIDER_REPORTED"
    LOCALLY_ESTIMATED = "LOCALLY_ESTIMATED"
    MIXED = "MIXED"
    UNAVAILABLE = "UNAVAILABLE"


class AccountingContext(BaseSchema):
    """Optional non-benchmark-specific attribution for a bounded accounting scope."""

    experiment_id: Optional[str] = None
    task_block_id: Optional[str] = None
    trial_id: Optional[str] = None
    treatment_id: Optional[str] = None
    execution_id: Optional[str] = None
    scope_kind: Optional[str] = None


class ProviderCallRecord(BaseSchema):
    """Auditable metadata for one provider transport attempt; excludes request/response content."""

    call_id: str = Field(default_factory=lambda: generate_id("call"))
    experiment_id: Optional[str] = None
    task_block_id: Optional[str] = None
    trial_id: Optional[str] = None
    treatment_id: Optional[str] = None
    execution_id: Optional[str] = None
    attempt_number: int
    retry_number: int = 0
    is_fallback: bool = False
    operator_model_version: Optional[str] = None
    model_version_source: str = "unavailable"
    response_model: Optional[str] = None
    model_identifier_policy: str = "provider_may_resolve_mutable_alias"
    stage: Optional[str] = None
    agent_id: Optional[str] = None
    agent_role: Optional[str] = None
    provider: str
    model: Optional[str] = None
    model_version: Optional[str] = None
    temperature: Optional[float] = None
    input_token_limit: Optional[int] = None
    output_token_limit: Optional[int] = None
    timeout_seconds: Optional[float] = None
    timeout_source: str = "unavailable"
    request_timestamp: str = Field(default_factory=utcnow_iso)
    duration_seconds: Optional[float] = None
    prompt_tokens: Optional[int] = None
    candidate_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_prompt_tokens: Optional[int] = None
    estimated_candidate_tokens: Optional[int] = None
    estimated_total_tokens: Optional[int] = None
    usage_source: ProviderUsageSource = ProviderUsageSource.UNAVAILABLE
    status: ProviderCallStatus = ProviderCallStatus.IN_PROGRESS
    error_category: Optional[str] = None
    sanitized_error: Optional[str] = None


class AccountingSnapshot(BaseSchema):
    """Bounded resource snapshot for plan preparation or one planned execution."""

    context: AccountingContext
    duration_seconds: float
    token_usage: Dict[str, Any] = Field(default_factory=dict)
    provider_calls: List[ProviderCallRecord] = Field(default_factory=list)
    dropped_provider_call_records: int = 0