from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field

from models.schemas import BaseSchema
from utils.helpers import generate_id, utcnow_iso


class DeliberationTriggerType(str, Enum):
    """Deterministic reasons for opening a bounded agent exchange."""

    CONFLICTING_CLAIMS = "CONFLICTING_CLAIMS"
    UNSUPPORTED_EMPIRICAL_CLAIM = "UNSUPPORTED_EMPIRICAL_CLAIM"
    UNSUPPORTED_ASSUMPTION = "UNSUPPORTED_ASSUMPTION"
    CRITICAL_UNRESOLVED_QUESTION = "CRITICAL_UNRESOLVED_QUESTION"
    DEPENDENCY_INFORMATION_GAP = "DEPENDENCY_INFORMATION_GAP"


class DeliberationResponseType(str, Enum):
    """Allowed response intents; none changes claim verification status."""

    CLARIFICATION = "CLARIFICATION"
    EVIDENCE_RESPONSE = "EVIDENCE_RESPONSE"
    AGREEMENT = "AGREEMENT"
    DISAGREEMENT = "DISAGREEMENT"
    HANDOFF = "HANDOFF"


class DeliberationTerminationReason(str, Enum):
    """Reasons a future deliberation coordinator may stop an exchange loop."""

    NO_RELEVANT_TRIGGER = "NO_RELEVANT_TRIGGER"
    RESOLVED = "RESOLVED"
    MAX_ROUNDS_REACHED = "MAX_ROUNDS_REACHED"
    MAX_EXCHANGES_REACHED = "MAX_EXCHANGES_REACHED"
    MESSAGE_BUDGET_EXHAUSTED = "MESSAGE_BUDGET_EXHAUSTED"
    TOKEN_BUDGET_EXHAUSTED = "TOKEN_BUDGET_EXHAUSTED"
    TIMEOUT = "TIMEOUT"
    AGENT_FAILURE = "AGENT_FAILURE"
    TASK_CANCELLED = "TASK_CANCELLED"
    UNRESOLVED_DISAGREEMENT = "UNRESOLVED_DISAGREEMENT"
    UNRESOLVED_QUESTIONS = "UNRESOLVED_QUESTIONS"


class DeliberationTrigger(BaseSchema):
    """A policy decision to request one targeted exchange.

    ``claim_reference`` is temporary before ledger extraction and maps to the
    later claim using ``claim:{execution_id}:{subtask_id}:{index}``.
    """

    trigger_id: str = Field(default_factory=lambda: generate_id("trigger"))
    execution_id: str
    exchange_id: str = Field(default_factory=lambda: generate_id("exchange"))
    correlation_id: str = Field(default_factory=lambda: generate_id("corr"))
    trigger_type: DeliberationTriggerType
    claim_reference: Optional[str] = None
    source_agent_id: str
    target_agent_id: str
    source_subtask_id: str
    target_subtask_id: Optional[str] = None
    reason: str
    target_selection_rationale: str = ""
    originating_response_message_id: Optional[str] = None
    related_claim_text: Optional[str] = None
    reply_to_message_id: Optional[str] = None
    round_number: int = 1
    created_at: str = Field(default_factory=utcnow_iso)


class DeliberationResponsePayload(BaseSchema):
    """Structured response payload for a future coordinator.

    This intentionally has no verification-status field. Responses are context,
    not reviewer decisions or evidence by themselves.
    """

    execution_id: str
    exchange_id: str
    correlation_id: str
    sender_agent_id: str
    recipient_agent_id: str
    response_type: DeliberationResponseType
    content: str
    claim_reference: Optional[str] = None
    new_claims: List[str] = Field(default_factory=list)
    clarifications: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    evidence_references: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    reply_to_message_id: Optional[str] = None
    useful_new_information: bool = False
    disagreement: bool = False


class DeliberationExchange(BaseSchema):
    """Audit record for one trigger and its optional response."""

    exchange_id: str
    execution_id: str
    correlation_id: str
    claim_reference: Optional[str] = None
    trigger_type: DeliberationTriggerType
    round_number: int = 1
    initiator_agent_id: str
    target_agent_id: str
    request_message_id: Optional[str] = None
    response_message_id: Optional[str] = None
    response_type: Optional[DeliberationResponseType] = None
    response_content: Optional[str] = None
    useful_new_information: bool = False
    introduced_claims: List[str] = Field(default_factory=list)
    introduced_claim_references: List[str] = Field(default_factory=list)
    introduced_claim_ids: List[Optional[str]] = Field(default_factory=list)
    clarifications: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    evidence_references: List[str] = Field(default_factory=list)
    evidence_references_verified: bool = False
    unresolved_questions: List[str] = Field(default_factory=list)
    disagreement: bool = False
    diagnostic: Optional[str] = None


class DeliberationRound(BaseSchema):
    """Bounded round summary for a future deliberation coordinator."""

    execution_id: str
    round_number: int
    exchange_ids: List[str] = Field(default_factory=list)
    message_count: int = 0
    token_count: Optional[int] = None
    termination_reason: Optional[DeliberationTerminationReason] = None


class SuppressedDeliberationTrigger(BaseSchema):
    """Audit record for an incremental trigger not scheduled for exchange."""

    execution_id: str
    originating_response_message_id: str
    trigger_type: DeliberationTriggerType
    claim_reference: Optional[str] = None
    source_agent_id: str
    target_agent_id: str
    reason: str
    target_selection_rationale: str = ""


class DeliberationReport(BaseSchema):
    """Serializable deliberation audit report, independent of verification status."""

    execution_id: str
    triggered: bool = False
    triggers: List[DeliberationTrigger] = Field(default_factory=list)
    suppressed_triggers: List[SuppressedDeliberationTrigger] = Field(default_factory=list)
    exchanges: List[DeliberationExchange] = Field(default_factory=list)
    rounds: List[DeliberationRound] = Field(default_factory=list)
    unresolved_disagreements: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    diagnostics: List[str] = Field(default_factory=list)
    termination_reason: Optional[DeliberationTerminationReason] = None
    total_messages: int = 0
    total_tokens: Optional[int] = None


__all__ = [
    "DeliberationExchange",
    "DeliberationReport",
    "DeliberationResponsePayload",
    "DeliberationResponseType",
    "DeliberationRound",
    "DeliberationTerminationReason",
    "DeliberationTrigger",
    "DeliberationTriggerType",
    "SuppressedDeliberationTrigger",
]
