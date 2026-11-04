import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from config.settings import MAX_REPAIR_CLAIMS_PER_ITERATION, MAX_REPAIR_ITERATIONS
from core.agent_factory import AgentFactory
from core.llm_client import BaseLLMClient, LLMClient
from models.schemas import ClaimStatus, EvidenceSourceType, VerificationDecision
from models.verification_models import Claim, Evidence, RepairAttempt
from utils.helpers import generate_id, parse_json_safely
from utils.logger import get_logger
from utils.retry import LLMBudgetExceededError, LLMQuotaExhaustedError
from verification.adversarial_reviewer import AdversarialReviewer
from verification.evidence_ledger import EvidenceLedger

logger = get_logger("repair_engine")


class ClaimRevisionOutput(BaseModel):
    """Pydantic schema for agent claim repair output."""

    revised_claim_text: str = Field(description="Corrected, bounded, and rigorously substantiated assertion")
    additional_reasoning: str = Field(description="Explanatory logic addressing the reviewer's critique")
    new_evidence_references: List[str] = Field(
        default_factory=list,
        description="New citations, benchmarks, or explicit boundary limits (never fabricate fake URLs/data)",
    )
    converted_to_assumption: bool = Field(
        default=False,
        description="Whether the assertion was tempered into an explicit working assumption or design target",
    )


class RepairEngine:
    """Targeted repair loop: routes reviewer critiques back to originating agents for precision correction.

    STRICT POLICY:
    - Only NEEDS_REVISION claims are eligible for automatic repair.
    - REJECTED claims remain permanently rejected.
    - SUPPORTED and UNVERIFIABLE claims are not repaired.
    - Re-reviews only revised claims in each iteration.
    - Halts immediately on quota or budget exhaustion.
    """

    def __init__(
        self,
        max_repair_iterations: int = MAX_REPAIR_ITERATIONS,
        max_repair_claims_per_iteration: int = MAX_REPAIR_CLAIMS_PER_ITERATION,
        llm_client: Optional[BaseLLMClient] = None,
    ):
        self.max_repair_iterations = max_repair_iterations
        self.max_repair_claims_per_iteration = max(1, max_repair_claims_per_iteration)
        self.client = llm_client or LLMClient.get_instance()

    async def execute_repair_loop(
        self,
        ledger: EvidenceLedger,
        reviewer: AdversarialReviewer,
        agent_factory: AgentFactory,
        task_context: str = "",
    ) -> List[RepairAttempt]:
        """Execute the iterative repair cycle exclusively on revision-requested claims."""
        repair_history: List[RepairAttempt] = []

        for iteration in range(1, self.max_repair_iterations + 1):
            all_claims = await ledger.get_all_claims()

            # STRICT ELIGIBILITY: Only NEEDS_REVISION claims can be repaired.
            # REJECTED claims are excluded. SUPPORTED claims need no repair.
            eligible_claims = [
                c for c in all_claims
                if c.status == ClaimStatus.NEEDS_REVISION
            ]

            if not eligible_claims:
                logger.info(
                    f"Repair iteration {iteration}: No NEEDS_REVISION claims remain. Exiting repair loop early."
                )
                break

            # Bound repair count by configured budget per iteration
            claims_to_repair = eligible_claims[: self.max_repair_claims_per_iteration]
            logger.info(
                f"Repair iteration {iteration}/{self.max_repair_iterations}: "
                f"Repairing {len(claims_to_repair)} eligible claim(s) "
                f"(budget limit: {self.max_repair_claims_per_iteration})..."
            )

            claims_to_rereview: List[Claim] = []

            for claim in claims_to_repair:
                attempt = await self._repair_single_claim(
                    claim=claim,
                    attempt_number=iteration,
                    ledger=ledger,
                    agent_factory=agent_factory,
                )
                repair_history.append(attempt)

                if attempt.revised_claim_text:
                    claims_to_rereview.append(claim)

            # Re-review ONLY the revised claims
            if claims_to_rereview:
                logger.info(f"Re-reviewing {len(claims_to_rereview)} revised claim(s)...")
                await reviewer.review_claims(
                    ledger=ledger,
                    task_context=task_context,
                    target_claims=claims_to_rereview,
                )

                # Update repair attempt statuses based on re-review outcomes
                for claim in claims_to_rereview:
                    latest_status = claim.status
                    for att in reversed(repair_history):
                        if att.claim_id == claim.claim_id:
                            att.status = latest_status
                            break

        # Final check: preserve uncertainty for any remaining unverified claims
        remaining_unverified = [
            c for c in await ledger.get_all_claims()
            if c.status == ClaimStatus.NEEDS_REVISION
        ]
        for c in remaining_unverified:
            logger.warning(
                f"Claim [{c.claim_id}] could not be verified within max repairs ({self.max_repair_iterations}). "
                f"Marking as UNVERIFIABLE to preserve uncertainty."
            )
            await ledger.update_claim_status(
                c.claim_id,
                ClaimStatus.UNVERIFIABLE,
                reviewer_feedback="Could not be substantiated within maximum repair iterations.",
            )

        return repair_history

    async def _repair_single_claim(
        self,
        claim: Claim,
        attempt_number: int,
        ledger: EvidenceLedger,
        agent_factory: AgentFactory,
    ) -> RepairAttempt:
        """Route critique back to agent and collect corrected claim."""
        agent = agent_factory.get_agent(claim.originating_agent)
        role = claim.originating_role or (agent.role if agent else "Domain Specialist")
        orig_text = claim.claim_text

        logger.info(
            f"Repairing claim [{claim.claim_id}] from [{role}]. "
            f"Critique: {claim.reviewer_feedback[:80] if claim.reviewer_feedback else 'Needs revision'}..."
        )

        system_instruction = (
            f"You are the {role} who authored a claim that was audited and flagged by an adversarial reviewer.\n"
            f"Your responsibility is to correct the claim:\n"
            f"1. Directly address the reviewer's specific critique and flaws.\n"
            f"2. Narrow and bound any over-generalized conclusions.\n"
            f"3. If an assertion cannot be empirically proven, convert it into an explicitly declared assumption or design target.\n"
            f"4. Do NOT double down on unsupported assertions or repeat the same flawed statement.\n"
            f"5. ANTI-FABRICATION RULE: Do NOT invent fake URLs, benchmarks, load-test results, or experimental data. "
            f"Mere rewording without empirical basis will be rejected."
        )

        prompt = (
            f"ORIGINAL CLAIM:\n{claim.claim_text}\n\n"
            f"ORIGINAL REASONING:\n{claim.supporting_reasoning}\n\n"
            f"REVIEWER AUDIT FEEDBACK:\n{claim.reviewer_feedback}\n\n"
            f"Provide a revised, defensible claim formatted as JSON:\n"
            f"- 'revised_claim_text': carefully bounded, verified version of the claim\n"
            f"- 'additional_reasoning': logical justification addressing the critique\n"
            f"- 'new_evidence_references': specific principles, standards, or boundaries grounding this fix (no fabricated URLs)\n"
            f"- 'converted_to_assumption': true if framed as an explicit assumption or design target rather than proven fact"
        )

        try:
            revision = await self.client.generate_structured(
                prompt=prompt,
                response_schema=ClaimRevisionOutput,
                system_instruction=system_instruction,
                temperature=0.2,
            )

            # Update the claim in the ledger
            claim.claim_text = revision.revised_claim_text
            claim.supporting_reasoning += f"\n[Repair {attempt_number}]: {revision.additional_reasoning}"
            if revision.converted_to_assumption:
                claim.assumptions.append(f"Derived from repair: {revision.revised_claim_text}")

            # Register any new evidence
            for ev_ref in revision.new_evidence_references:
                ev = Evidence(
                    evidence_id=generate_id("ev_rep"),
                    source_type=EvidenceSourceType.REASONING_INFERENCE,
                    source_reference=f"Repair {attempt_number} by {role}",
                    supporting_content=ev_ref,
                    originating_agent=claim.originating_agent,
                    associated_claim_ids=[claim.claim_id],
                )
                await ledger.register_evidence(ev, [claim.claim_id])

            # Reset status to PENDING for re-review in memory and in ledger
            claim.status = ClaimStatus.PENDING
            await ledger.update_claim_status(
                claim_id=claim.claim_id,
                status=ClaimStatus.PENDING,
                reviewer_feedback=f"Repaired claim awaiting re-audit (Iteration {attempt_number})",
            )

            return RepairAttempt(
                attempt_number=attempt_number,
                claim_id=claim.claim_id,
                agent_id=claim.originating_agent,
                role=role,
                original_claim_text=orig_text,
                rejection_reason=claim.reviewer_feedback or "Flagged by reviewer",
                revised_claim_text=revision.revised_claim_text,
                additional_reasoning=revision.additional_reasoning,
                new_evidence_references=revision.new_evidence_references,
                status=ClaimStatus.PENDING,
            )

        except (LLMQuotaExhaustedError, LLMBudgetExceededError):
            raise

        except Exception as e:
            logger.error(f"Repair attempt failed for claim [{claim.claim_id}]: {e}")
            return RepairAttempt(
                attempt_number=attempt_number,
                claim_id=claim.claim_id,
                agent_id=claim.originating_agent,
                role=role,
                original_claim_text=orig_text,
                rejection_reason=claim.reviewer_feedback or "Failed repair",
                revised_claim_text=None,
                additional_reasoning=f"Repair failed: {e}",
                status=claim.status,
            )
