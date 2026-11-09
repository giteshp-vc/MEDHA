import asyncio
import re
from typing import Any, Dict, List, Optional

from agents.reviewer_agent import GENERAL_KNOWLEDGE_PREFIX, ReviewerAgent
from config import settings
from config.settings import REVIEW_BATCH_SIZE
from core.llm_client import BaseLLMClient, LLMClient
from models.schemas import ClaimKind, ClaimStatus, VerificationDecision
from models.verification_models import Claim, ReviewFeedback
from utils.logger import get_logger
from verification.claim_validator import ClaimValidator
from verification.evidence_ledger import EvidenceLedger

logger = get_logger("adversarial_reviewer")

NUMERIC_CLAIM_PATTERN = re.compile(r"\d")


class AdversarialReviewer:
    """Orchestrates rigorous, independent adversarial auditing across all claims in the Evidence Ledger."""

    def __init__(
        self,
        reviewer_agent: Optional[ReviewerAgent] = None,
        llm_client: Optional[BaseLLMClient] = None,
        batch_size: int = REVIEW_BATCH_SIZE,
    ):
        self.client = llm_client or LLMClient.get_instance()
        self.reviewer_agent = reviewer_agent or ReviewerAgent(llm_client=self.client)
        self.validator = ClaimValidator()
        self.batch_size = max(1, batch_size)

    async def review_claims(
        self,
        ledger: EvidenceLedger,
        task_context: str = "",
        target_claims: Optional[List[Claim]] = None,
        deliberation_context: str = "",
    ) -> List[ReviewFeedback]:
        """Perform adversarial auditing on pending or target claims in the ledger in configurable batches."""
        claims_to_review = (
            target_claims
            if target_claims is not None
            else await ledger.get_all_claims(status=ClaimStatus.PENDING)
        )

        logger.info(
            f"AdversarialReviewer auditing {len(claims_to_review)} claim(s) "
            f"in batches of up to {self.batch_size}..."
        )
        feedback_results: List[ReviewFeedback] = []

        for i in range(0, len(claims_to_review), self.batch_size):
            batch = claims_to_review[i : i + self.batch_size]

            # Gather associated evidence for the claims in this batch
            evidence_by_claim: Dict[str, List[Any]] = {}
            for claim in batch:
                evidence_by_claim[claim.claim_id] = await ledger.get_evidence_for_claim(claim.claim_id)

            # Audit batch in single structured LLM call
            with self.client.get_token_tracker().call_context(
                agent_id=self.reviewer_agent.agent_id,
                agent_role=self.reviewer_agent.role,
                stage="verification",
            ):
                batch_feedbacks = await self.reviewer_agent.evaluate_claims_batch(
                    claims=batch,
                    evidence_by_claim=evidence_by_claim,
                    task_context=task_context,
                    deliberation_context=deliberation_context,
                )

            feedbacks_map = {fb.claim_id: fb for fb in batch_feedbacks}

            for claim in batch:
                feedback = feedbacks_map.get(claim.claim_id)
                if not feedback:
                    # Bounded safety fallback: never accept if feedback was missing
                    feedback = ReviewFeedback(
                        claim_id=claim.claim_id,
                        decision=VerificationDecision.UNVERIFIABLE,
                        reasoning="Evaluation feedback missing from batch audit response.",
                        critique="Verification audit could not be mapped to claim ID.",
                        is_supported_by_evidence=False,
                    )

                # Heuristic validation: flag superlatives or ungrounded absolutes
                ev_items = evidence_by_claim.get(claim.claim_id, [])
                is_structurally_valid, issues = self.validator.validate_claim_structure(claim, evidence_items=ev_items)
                if not is_structurally_valid and feedback.decision == VerificationDecision.ACCEPT:
                    feedback.decision = VerificationDecision.REQUEST_REVISION
                    feedback.reasoning += f" [Heuristic Flag: {'; '.join(issues)}]"
                    feedback.suggested_fix = (
                        (feedback.suggested_fix or "")
                        + " Remove absolute unverified superlatives or provide empirical proof."
                    )

                if feedback.decision == VerificationDecision.ACCEPT and claim.claim_kind in (
                    ClaimKind.PROPOSED_TARGET,
                    ClaimKind.ASSUMPTION,
                    ClaimKind.RECOMMENDATION,
                ):
                    feedback.decision = VerificationDecision.REQUEST_REVISION
                    feedback.reasoning += (
                        f" [Heuristic Flag: {claim.claim_kind} cannot be accepted as an established factual result.]"
                    )
                    feedback.suggested_fix = (
                        (feedback.suggested_fix or "")
                        + " Keep this explicitly labeled as a target, assumption, or recommendation."
                    )

                if (
                    feedback.decision == VerificationDecision.ACCEPT
                    and settings.VERIFICATION_MODE == "general"
                    and not ev_items
                    and NUMERIC_CLAIM_PATTERN.search(claim.claim_text)
                ):
                    feedback.decision = VerificationDecision.REQUEST_REVISION
                    feedback.reasoning += (
                        " [Heuristic Flag: numeric or quantitative claims need registered evidence even in general mode.]"
                    )
                    feedback.suggested_fix = (
                        (feedback.suggested_fix or "")
                        + " Remove unsupported figures or provide a source for them."
                    )

                if (
                    feedback.decision == VerificationDecision.ACCEPT
                    and settings.VERIFICATION_MODE == "general"
                    and not ev_items
                    and GENERAL_KNOWLEDGE_PREFIX not in feedback.reasoning
                ):
                    feedback.reasoning = f"{GENERAL_KNOWLEDGE_PREFIX} {feedback.reasoning}"

                # Map decision to ClaimStatus
                if feedback.decision == VerificationDecision.ACCEPT:
                    status = ClaimStatus.SUPPORTED
                elif feedback.decision == VerificationDecision.REJECT:
                    status = ClaimStatus.REJECTED
                elif feedback.decision == VerificationDecision.REQUEST_REVISION:
                    status = ClaimStatus.NEEDS_REVISION
                else:
                    status = ClaimStatus.UNVERIFIABLE

                # Update in-memory claim object
                claim.status = status
                claim.reviewer_feedback = f"[{feedback.decision}] {feedback.reasoning}"

                # Update ledger state
                await ledger.update_claim_status(
                    claim_id=claim.claim_id,
                    status=status,
                    reviewer_feedback=claim.reviewer_feedback,
                )

                feedback_results.append(feedback)

        accepted_count = sum(1 for f in feedback_results if f.decision == VerificationDecision.ACCEPT)
        rejected_count = sum(1 for f in feedback_results if f.decision == VerificationDecision.REJECT)
        revision_count = sum(1 for f in feedback_results if f.decision == VerificationDecision.REQUEST_REVISION)
        unverifiable_count = sum(1 for f in feedback_results if f.decision == VerificationDecision.UNVERIFIABLE)

        logger.info(
            f"Adversarial audit completed: "
            f"Accepted: {accepted_count}, Rejected: {rejected_count}, "
            f"Needs Revision: {revision_count}, Unverifiable: {unverifiable_count}"
        )

        return feedback_results

    async def review_single_claim(
        self,
        claim: Claim,
        ledger: EvidenceLedger,
        task_context: str = "",
        deliberation_context: str = "",
    ) -> ReviewFeedback:
        """Audit an individual claim against its registered evidence (backward compatibility)."""
        results = await self.review_claims(
            ledger=ledger,
            task_context=task_context,
            target_claims=[claim],
            deliberation_context=deliberation_context,
        )
        if results:
            return results[0]

        return ReviewFeedback(
            claim_id=claim.claim_id,
            decision=VerificationDecision.UNVERIFIABLE,
            reasoning="Audit produced no feedback.",
            is_supported_by_evidence=False,
        )
