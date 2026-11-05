from typing import Any, Dict, List, Optional
from pydantic import Field

from models.schemas import BaseSchema, ClaimKind, ClaimStatus, EvidenceSourceType, VerificationDecision
from utils.helpers import generate_id, utcnow_iso


class Evidence(BaseSchema):
    """An individual piece of supporting evidence stored in the Evidence Ledger."""

    evidence_id: str = Field(default_factory=lambda: generate_id("ev"))
    source_type: EvidenceSourceType = EvidenceSourceType.DOMAIN_KNOWLEDGE
    source_reference: str = Field(description="User prompt snippet, retrieval citation, or standard reference")
    supporting_content: str = Field(description="Content of the evidence")
    originating_agent: str = Field(description="ID or role of agent providing evidence")
    associated_claim_ids: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow_iso)

    # Provenance and artifact metadata
    artifact_type: Optional[str] = Field(
        default=None,
        description="Type of artifact: benchmark_report, log_file, api_response, doc, code_file, etc.",
    )
    source_identifier: Optional[str] = Field(
        default=None,
        description="Inspectable URI, file path, tool execution ID, or external source ID",
    )
    retrieval_timestamp: Optional[str] = Field(
        default=None,
        description="ISO timestamp when the artifact was retrieved or uploaded",
    )
    evidence_location: Optional[str] = Field(
        default=None,
        description="Location within evidence: line range, section, JSON path, or offset",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured provenance metadata",
    )

    @property
    def is_user_assertion(self) -> bool:
        """Indicate whether this is an unverified user statement or prompt assertion."""
        return self.source_type in (
            EvidenceSourceType.USER_INPUT,
            EvidenceSourceType.USER_ASSERTION,
        )

    @property
    def is_external(self) -> bool:
        """Indicate whether this evidence is a verifiable external artifact rather than an assertion or model reasoning.

        Strict Provenance Rules:
        - USER_PROVIDED_ARTIFACT: Qualifies IF it has supporting content AND an inspectable source reference,
          source identifier, or artifact type (e.g. uploaded benchmark report, schema, log file).
        - TOOL_RETRIEVED: Qualifies IF it has supporting content AND a traceable source reference or identifier.
        - BENCHMARK_DATA: Qualifies IF it has supporting content AND a benchmark source reference or identifier.

        Plain user statements (USER_INPUT, USER_ASSERTION) and model inferences
        (REASONING_INFERENCE, DOMAIN_KNOWLEDGE) do NOT qualify as external evidence.
        """
        if self.source_type in (
            EvidenceSourceType.USER_INPUT,
            EvidenceSourceType.USER_ASSERTION,
            EvidenceSourceType.REASONING_INFERENCE,
            EvidenceSourceType.DOMAIN_KNOWLEDGE,
        ):
            return False

        if not self.supporting_content or not self.supporting_content.strip():
            return False

        if self.source_type == EvidenceSourceType.USER_PROVIDED_ARTIFACT:
            return bool(
                (self.source_reference and self.source_reference.strip())
                or (self.source_identifier and self.source_identifier.strip())
                or (self.artifact_type and self.artifact_type.strip())
            )

        if self.source_type in (
            EvidenceSourceType.TOOL_RETRIEVED,
            EvidenceSourceType.BENCHMARK_DATA,
        ):
            return bool(
                (self.source_reference and self.source_reference.strip())
                or (self.source_identifier and self.source_identifier.strip())
            )

        return False


class Claim(BaseSchema):
    """A factual or analytical assertion extracted from agent outputs for verification."""

    claim_id: str = Field(default_factory=lambda: generate_id("claim"))
    claim_text: str = Field(description="The concrete assertion being evaluated")
    originating_agent: str = Field(description="Agent ID that made the claim")
    originating_role: str = Field(default="", description="Role of the agent that made the claim")
    task_id: str = Field(default="")
    subtask_id: str = Field(default="")
    claim_kind: ClaimKind = ClaimKind.FACTUAL_ASSERTION
    source_provenance: List[Dict[str, str]] = Field(default_factory=list)
    evidence_references: List[str] = Field(default_factory=list, description="IDs of evidence supporting this claim")
    unverified_evidence_references: List[str] = Field(
        default_factory=list,
        description="References mentioned by agents that are not registered or validated evidence.",
    )
    supporting_reasoning: str = Field(default="", description="Logical rationale provided by the agent")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions this claim relies on")
    status: ClaimStatus = Field(default=ClaimStatus.PENDING)
    reviewer_feedback: Optional[str] = None
    verification_history: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow_iso)
    updated_at: str = Field(default_factory=utcnow_iso)


class ClaimEvaluationSchema(BaseSchema):
    """Pydantic schema for individual claim evaluation by Reviewer."""

    claim_id: str
    decision: VerificationDecision
    reasoning: str = Field(description="Adversarial scrutiny of the claim and supporting rationale")
    critique: str = Field(default="", description="Specific flaws, fallacies, or unsubstantiated extrapolations")
    suggested_fix: Optional[str] = Field(default=None, description="Actionable instructions for targeted repair")
    is_supported_by_evidence: bool = Field(default=False, description="Is the claim directly backed by valid evidence or logic?")
    is_evidence_relevant: bool = Field(default=False, description="Is the cited evidence directly applicable?")
    has_contradictions: bool = Field(default=False, description="Does this claim conflict with known facts or other verified claims?")
    assumptions_presented_as_facts: bool = Field(default=False, description="Are unverified assumptions framed as certain truth?")
    logically_valid: bool = Field(default=False, description="Is the internal deduction logically sound?")


class BatchReviewSchema(BaseSchema):
    """Pydantic schema for batched claim reviews."""

    reviews: List[ClaimEvaluationSchema] = Field(
        default_factory=list,
        description="List containing exactly one evaluation for each requested claim ID",
    )


class ReviewFeedback(BaseSchema):
    """Structured evaluation returned by the Adversarial Reviewer for a claim."""

    review_id: str = Field(default_factory=lambda: generate_id("rev"))
    claim_id: str
    decision: VerificationDecision
    reasoning: str = Field(description="Adversarial evaluation rationale")
    critique: str = Field(default="", description="Specific flaws, fallacies, or unsubstantiated points identified")
    suggested_fix: Optional[str] = Field(default=None, description="Guidance for targeted repair")
    is_supported_by_evidence: bool = Field(default=False)
    is_evidence_relevant: bool = Field(default=False)
    has_contradictions: bool = Field(default=False)
    assumptions_presented_as_facts: bool = Field(default=False)
    logically_valid: bool = Field(default=False)
    timestamp: str = Field(default_factory=utcnow_iso)


class RepairAttempt(BaseSchema):
    """Record of a targeted repair cycle for a rejected claim."""

    repair_id: str = Field(default_factory=lambda: generate_id("rep"))
    attempt_number: int
    claim_id: str
    agent_id: str
    role: str
    original_claim_text: str
    rejection_reason: str
    revised_claim_text: Optional[str] = None
    additional_reasoning: str = ""
    new_evidence_references: List[str] = Field(default_factory=list)
    status: ClaimStatus = ClaimStatus.NEEDS_REVISION
    timestamp: str = Field(default_factory=utcnow_iso)


class VerificationReport(BaseSchema):
    """Comprehensive summary of the claim verification and repair lifecycle."""

    total_claims: int = 0
    accepted_claims: List[Claim] = Field(default_factory=list)
    rejected_claims: List[Claim] = Field(default_factory=list)
    needs_revision_claims: List[Claim] = Field(default_factory=list)
    unverifiable_claims: List[Claim] = Field(default_factory=list)
    unreviewed_claims: List[Claim] = Field(default_factory=list)  # Claims never audited due to interruption
    review_decisions: List[ReviewFeedback] = Field(default_factory=list)
    repair_history: List[RepairAttempt] = Field(default_factory=list)
    support_rate: Optional[float] = None  # Percentage over reviewed claims; None when no claims were reviewed
    unsupported_rate: Optional[float] = None  # Percentage over reviewed claims; None when no claims were reviewed
    repair_success_rate: float = 0.0  # Percentage: 0.0 to 100.0
    summary_message: Optional[str] = None
