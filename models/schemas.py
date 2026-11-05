from enum import Enum
from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    """Base Pydantic model for MEDHA schemas with standard config."""

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        populate_by_name=True,
    )


class TaskStatus(str, Enum):
    """Status lifecycle of a task or subtask."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"


class AgentStatus(str, Enum):
    """Status lifecycle of an agent."""

    INITIALIZED = "INITIALIZED"
    THINKING = "THINKING"
    EXECUTING = "EXECUTING"
    WAITING = "WAITING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"


class MessageType(str, Enum):
    """Types of inter-agent messages on the message bus."""

    FINDING = "FINDING"
    QUESTION = "QUESTION"
    CHALLENGE = "CHALLENGE"
    RESPONSE = "RESPONSE"
    EVIDENCE = "EVIDENCE"
    CORRECTION = "CORRECTION"
    STATUS = "STATUS"
    EVIDENCE_REQUEST = "EVIDENCE_REQUEST"
    EVIDENCE_RESPONSE = "EVIDENCE_RESPONSE"
    CLARIFICATION = "CLARIFICATION"
    AGREEMENT = "AGREEMENT"
    DISAGREEMENT = "DISAGREEMENT"
    HANDOFF = "HANDOFF"


class ClaimStatus(str, Enum):
    """Verification status of a claim in the evidence ledger."""

    PENDING = "PENDING"
    UNREVIEWED = "UNREVIEWED"
    SUPPORTED = "SUPPORTED"
    REJECTED = "REJECTED"
    NEEDS_REVISION = "NEEDS_REVISION"
    UNVERIFIABLE = "UNVERIFIABLE"


class ClaimKind(str, Enum):
    FACTUAL_ASSERTION = "FACTUAL_ASSERTION"
    PROPOSED_TARGET = "PROPOSED_TARGET"
    EMPIRICAL_RESULT = "EMPIRICAL_RESULT"
    ASSUMPTION = "ASSUMPTION"
    RECOMMENDATION = "RECOMMENDATION"


class VerificationDecision(str, Enum):
    """Decision made by the adversarial reviewer for a claim."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    REQUEST_REVISION = "REQUEST_REVISION"
    UNVERIFIABLE = "UNVERIFIABLE"


class EvidenceSourceType(str, Enum):
    """Source classification for pieces of evidence with clear provenance boundaries."""

    USER_INPUT = "USER_INPUT"  # Legacy/general user prompt text (treated as user statement)
    USER_ASSERTION = "USER_ASSERTION"  # Plain user assertion or subjective claim in prompt
    USER_PROVIDED_ARTIFACT = "USER_PROVIDED_ARTIFACT"  # Uploaded file, benchmark report, schema, log, or dataset
    TOOL_RETRIEVED = "TOOL_RETRIEVED"  # Retrieved via tool/API/DB/search with traceability
    REASONING_INFERENCE = "REASONING_INFERENCE"  # Internal model deduction or specialist derivation
    DOMAIN_KNOWLEDGE = "DOMAIN_KNOWLEDGE"  # Internal LLM pretraining knowledge
    BENCHMARK_DATA = "BENCHMARK_DATA"  # Empirically measured test or benchmark data
