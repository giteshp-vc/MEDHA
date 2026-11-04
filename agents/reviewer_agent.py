import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from models.agent_models import AgentConfig, AgentResponse
from models.schemas import AgentStatus, VerificationDecision
from models.verification_models import (
    BatchReviewSchema,
    Claim,
    ClaimEvaluationSchema,
    Evidence,
    ReviewFeedback,
)
from utils.helpers import parse_json_safely
from utils.logger import get_logger
from utils.retry import LLMBudgetExceededError, LLMQuotaExhaustedError

logger = get_logger("reviewer_agent")


REVIEWER_SYSTEM_INSTRUCTION = (
    "You are an adversarial claim auditor in the MEDHA multi-agent intelligence system. "
    "Your duty is to uphold rigorous truthfulness, sound deduction, and empirical fidelity.\n"
    "DO NOT accept a claim simply because it sounds plausible or because an agent asserted it with high confidence.\n"
    "Scrutinize whether assumptions are masquerading as established facts, whether conclusions outpace the premises, "
    "and whether claims are verifiable.\n\n"
    "STRICT EVIDENCE PROVENANCE & ANTI-FABRICATION RULES:\n"
    "1. Distinguish actual external evidence from assumptions, model reasoning, and proposed design decisions.\n"
    "2. Never invent or accept fabricated URLs, citations, benchmarks, load-test results, internal experiments, or empirical measurements.\n"
    "3. A statement about a completed test, measurement, or benchmark requires a real test artifact or provided external evidence; "
    "otherwise flag it as an unverified empirical assertion and decide REQUEST_REVISION or REJECT.\n"
    "4. A proposed target, design goal, or SLA must be classified as an ASSUMPTION/TARGET, not a measured fact.\n"
    "5. Model agreement or consensus alone NEVER qualifies as independent external evidence. Reviewer acceptance confirms deductive "
    "soundness and absence of contradiction, NOT empirical reality.\n"
    "6. Model-generated reasoning, assumptions, and design recommendations are internal inferences and must NOT be treated as external evidence.\n"
    "7. Unsupported claims must NOT become ACCEPT merely because they were reworded without new supporting evidence.\n\n"
    "Evaluate along these 8 criteria:\n"
    "1. Is the claim supported by evidence?\n"
    "2. Is the evidence relevant?\n"
    "3. Does the evidence actually support the claim?\n"
    "4. Are there contradictions?\n"
    "5. Are assumptions being presented as facts?\n"
    "6. Is the reasoning logically valid?\n"
    "7. Is information missing?\n"
    "8. Is the conclusion stronger than the evidence allows?\n\n"
    "Decisions:\n"
    "- ACCEPT: The claim is strictly supported by evidence or sound deductive necessity.\n"
    "- REJECT: The claim is demonstrably false, deeply contradictory, or dangerous.\n"
    "- REQUEST_REVISION: The claim is promising but unsupported, over-generalized, or confusing assumption with fact.\n"
    "- UNVERIFIABLE: The claim relies on unknowable external facts or unprovable speculation."
)


class ReviewerAgent(BaseAgent):
    """Adversarial Reviewer Agent dedicated to independent, evidence-grounded claim scrutiny."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        cfg = config or AgentConfig(
            role="Adversarial Claim Reviewer",
            objective="Independently interrogate and verify factual and architectural claims.",
            instructions=(
                "Reject claims that lack evidence, confuse assumptions with facts, "
                "or exceed the bounds of their premises. Never accept by consensus alone. "
                "Never allow fabricated evidence or ungrounded empirical assertions."
            ),
        )
        super().__init__(cfg, llm_client=llm_client)

    async def evaluate_claim(
        self,
        claim: Claim,
        evidence_items: List[Evidence],
        task_context: str = "",
    ) -> ReviewFeedback:
        """Evaluate a single claim against associated evidence and task context (backward compatibility)."""
        batch_results = await self.evaluate_claims_batch(
            claims=[claim],
            evidence_by_claim={claim.claim_id: evidence_items},
            task_context=task_context,
        )
        if batch_results:
            return batch_results[0]

        # Defensive fallback if batch returned empty
        return ReviewFeedback(
            claim_id=claim.claim_id,
            decision=VerificationDecision.UNVERIFIABLE,
            reasoning="Evaluation produced no feedback.",
            critique="Evaluation failed to produce output.",
            is_supported_by_evidence=False,
        )

    async def evaluate_claims_batch(
        self,
        claims: List[Claim],
        evidence_by_claim: Dict[str, List[Evidence]],
        task_context: str = "",
    ) -> List[ReviewFeedback]:
        """Evaluate a batch of claims in a single structured LLM call with strict provenance and validation."""
        if not claims:
            return []

        logger.info(f"Reviewer evaluating batch of {len(claims)} claim(s)...")

        # Format each claim with its associated evidence
        claim_blocks: List[str] = []
        expected_claim_ids = set()

        for idx, claim in enumerate(claims, 1):
            expected_claim_ids.add(claim.claim_id)
            ev_items = evidence_by_claim.get(claim.claim_id, [])
            if ev_items:
                ev_lines = []
                for e in ev_items:
                    meta_tags = [f"SourceType: {e.source_type}"]
                    if getattr(e, "artifact_type", None):
                        meta_tags.append(f"Artifact: {e.artifact_type}")
                    if getattr(e, "source_identifier", None):
                        meta_tags.append(f"Identifier: {e.source_identifier}")
                    if getattr(e, "evidence_location", None):
                        meta_tags.append(f"Location: {e.evidence_location}")
                    if getattr(e, "retrieval_timestamp", None):
                        meta_tags.append(f"Retrieved: {e.retrieval_timestamp}")
                    meta_str = ", ".join(meta_tags)
                    ev_lines.append(
                        f"    - [Evidence {e.evidence_id} ({meta_str})]: {e.supporting_content} (Reference: {e.source_reference})"
                    )
                ev_text = "\n".join(ev_lines)
            else:
                ev_text = "    None provided."

            block = (
                f"### Claim #{idx} [ID: {claim.claim_id}]\n"
                f"- Text: {claim.claim_text}\n"
                f"- Classification: {claim.claim_kind}\n"
                f"- Originating Specialist: {claim.originating_role} ({claim.originating_agent})\n"
                f"- Specialist Reasoning: {claim.supporting_reasoning or 'None provided.'}\n"
                f"- Stated Assumptions: {', '.join(claim.assumptions) if claim.assumptions else 'None stated.'}\n"
                f"- Registered Evidence:\n{ev_text}\n"
            )
            claim_blocks.append(block)

        prompt = (
            f"AUDIT THE FOLLOWING {len(claims)} CLAIM(S) INDEPENDENTLY:\n\n"
            + "\n".join(claim_blocks)
            + f"\nGlobal Task Context:\n{task_context[:500]}\n\n"
            f"MANDATORY INSTRUCTIONS:\n"
            f"1. You MUST independently evaluate EVERY claim listed above.\n"
            f"2. Your output JSON MUST contain a 'reviews' array containing EXACTLY ONE evaluation item for each of the following claim IDs: {sorted(list(expected_claim_ids))}.\n"
            f"3. Do NOT omit any claim ID. Do NOT include duplicate claim IDs.\n"
            f"4. For each claim, determine 'decision' ('ACCEPT', 'REJECT', 'REQUEST_REVISION', or 'UNVERIFIABLE') "
            f"strictly adhering to evidence provenance rules.\n"
            f"5. A claim with no external evidence artifact that asserts empirical performance (e.g. latency, throughput, benchmark) MUST NOT be ACCEPTed.\n"
            f"6. A claim classified as EMPIRICAL_RESULT MUST NOT be ACCEPTed without external evidence.\n"
            f"7. Reviewer agreement or LLM consensus does NOT transform internal reasoning into external empirical verification."
        )

        try:
            batch_out = await self.client.generate_structured(
                prompt=prompt,
                response_schema=BatchReviewSchema,
                system_instruction=REVIEWER_SYSTEM_INSTRUCTION,
                model=self.model_name,
                temperature=0.1,
            )

            # Validate batch output: check for missing, duplicate, or misassigned claim IDs
            reviews_by_id: Dict[str, ClaimEvaluationSchema] = {}
            duplicate_ids = set()
            for r in batch_out.reviews:
                if r.claim_id in reviews_by_id:
                    duplicate_ids.add(r.claim_id)
                reviews_by_id[r.claim_id] = r

            if duplicate_ids:
                logger.warning(f"Batch review returned duplicate evaluations for IDs: {duplicate_ids}")

            results: List[ReviewFeedback] = []
            missing_claims: List[Claim] = []

            for claim in claims:
                cid = claim.claim_id
                if cid in reviews_by_id and cid not in duplicate_ids:
                    ev_out = reviews_by_id[cid]
                    results.append(
                        ReviewFeedback(
                            claim_id=cid,
                            decision=ev_out.decision,
                            reasoning=ev_out.reasoning,
                            critique=ev_out.critique,
                            suggested_fix=ev_out.suggested_fix,
                            is_supported_by_evidence=ev_out.is_supported_by_evidence,
                            is_evidence_relevant=ev_out.is_evidence_relevant,
                            has_contradictions=ev_out.has_contradictions,
                            assumptions_presented_as_facts=ev_out.assumptions_presented_as_facts,
                            logically_valid=ev_out.logically_valid,
                        )
                    )
                else:
                    missing_claims.append(claim)

            # If all claims were successfully audited, return results
            if not missing_claims:
                return results

            # Bounded recovery for missing or duplicate claims: attempt individual audits
            logger.warning(
                f"Batch review missed {len(missing_claims)} claim(s). Executing bounded recovery..."
            )
            for claim in missing_claims:
                recovered = await self._recover_single_claim(
                    claim=claim,
                    evidence_items=evidence_by_claim.get(claim.claim_id, []),
                    task_context=task_context,
                )
                results.append(recovered)

            return results

        except (LLMQuotaExhaustedError, LLMBudgetExceededError):
            # Quota/budget errors must propagate immediately without bounded fallback
            raise

        except Exception as batch_err:
            logger.warning(
                f"Batch review request failed ({batch_err.__class__.__name__}: {batch_err}). "
                f"Initiating bounded individual recovery for {len(claims)} claim(s)..."
            )
            # Bounded recovery: fall back to auditing each claim individually
            results = []
            for claim in claims:
                try:
                    single_rev = await self._recover_single_claim(
                        claim=claim,
                        evidence_items=evidence_by_claim.get(claim.claim_id, []),
                        task_context=task_context,
                    )
                    results.append(single_rev)
                except (LLMQuotaExhaustedError, LLMBudgetExceededError):
                    raise
                except Exception as rec_err:
                    logger.error(
                        f"Individual recovery failed for claim [{claim.claim_id}]: {rec_err}. "
                        f"Assigning UNVERIFIABLE verdict."
                    )
                    # Never treat batch-level failure as claim acceptance!
                    results.append(
                        ReviewFeedback(
                            claim_id=claim.claim_id,
                            decision=VerificationDecision.UNVERIFIABLE,
                            reasoning=f"Audit failed due to error: {rec_err}",
                            critique="Batch and individual audit recovery failed.",
                            is_supported_by_evidence=False,
                        )
                    )
            return results

    async def _recover_single_claim(
        self,
        claim: Claim,
        evidence_items: List[Evidence],
        task_context: str = "",
    ) -> ReviewFeedback:
        """Single-claim audit helper used as bounded fallback during batch recovery."""
        if evidence_items:
            ev_text = "\n".join(
                f"- [Evidence {e.evidence_id} ({e.source_type})]: {e.supporting_content} (Source: {e.source_reference})"
                for e in evidence_items
            )
        else:
            ev_text = "No explicit evidence items provided."

        prompt = (
            f"AUDIT THE FOLLOWING CLAIM:\n"
            f"Claim ID: {claim.claim_id}\n"
            f"Claim Text: {claim.claim_text}\n"
            f"Originating Agent: {claim.originating_role} ({claim.originating_agent})\n"
            f"Agent's Supporting Reasoning: {claim.supporting_reasoning or 'None provided.'}\n"
            f"Assumptions Stated: {', '.join(claim.assumptions) if claim.assumptions else 'None stated.'}\n"
            f"Available Evidence:\n{ev_text}\n\n"
            f"Global Task Context: {task_context[:500]}\n\n"
            f"Return a structured evaluation JSON for Claim ID '{claim.claim_id}'."
        )

        try:
            eval_out = await self.client.generate_structured(
                prompt=prompt,
                response_schema=ClaimEvaluationSchema,
                system_instruction=REVIEWER_SYSTEM_INSTRUCTION,
                model=self.model_name,
                temperature=0.1,
            )
            return ReviewFeedback(
                claim_id=claim.claim_id,
                decision=eval_out.decision,
                reasoning=eval_out.reasoning,
                critique=eval_out.critique,
                suggested_fix=eval_out.suggested_fix,
                is_supported_by_evidence=eval_out.is_supported_by_evidence,
                is_evidence_relevant=eval_out.is_evidence_relevant,
                has_contradictions=eval_out.has_contradictions,
                assumptions_presented_as_facts=eval_out.assumptions_presented_as_facts,
                logically_valid=eval_out.logically_valid,
            )
        except (LLMQuotaExhaustedError, LLMBudgetExceededError):
            raise
        except Exception as e:
            logger.warning(f"Structured recovery failed ({e}), attempting text parse fallback...")
            try:
                raw_text = await self.client.generate_text(
                    prompt=prompt + "\nReturn ONLY valid JSON.",
                    system_instruction=REVIEWER_SYSTEM_INSTRUCTION,
                    model=self.model_name,
                    temperature=0.1,
                )
                parsed = parse_json_safely(raw_text)
                dec_str = parsed.get("decision", "UNVERIFIABLE").upper()
                # If parsed decision is not valid or was malformed, default to UNVERIFIABLE
                decision = VerificationDecision(dec_str) if dec_str in VerificationDecision.__members__ else VerificationDecision.UNVERIFIABLE
                # Crucial safety check: never silently accept on recovery fallback if evidence was not validated
                if decision == VerificationDecision.ACCEPT and not parsed.get("is_supported_by_evidence"):
                    decision = VerificationDecision.REQUEST_REVISION

                return ReviewFeedback(
                    claim_id=claim.claim_id,
                    decision=decision,
                    reasoning=parsed.get("reasoning", "Audit completed via text fallback."),
                    critique=parsed.get("critique", "Evaluated via bounded fallback."),
                    suggested_fix=parsed.get("suggested_fix", None),
                    is_supported_by_evidence=bool(parsed.get("is_supported_by_evidence", False)),
                    is_evidence_relevant=bool(parsed.get("is_evidence_relevant", False)),
                    has_contradictions=bool(parsed.get("has_contradictions", False)),
                    assumptions_presented_as_facts=bool(parsed.get("assumptions_presented_as_facts", False)),
                    logically_valid=bool(parsed.get("logically_valid", False)),
                )
            except (LLMQuotaExhaustedError, LLMBudgetExceededError):
                raise
            except Exception as final_e:
                # Never accept on failure
                return ReviewFeedback(
                    claim_id=claim.claim_id,
                    decision=VerificationDecision.UNVERIFIABLE,
                    reasoning=f"Audit failed during text recovery: {final_e}",
                    critique="Failed to verify claim due to parsing errors.",
                    is_supported_by_evidence=False,
                )

    async def execute(self, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Generic execution implementation to satisfy BaseAgent contract."""
        self.update_status(AgentStatus.COMPLETED)
        return AgentResponse(
            agent_id=self.agent_id,
            role=self.role,
            task_id=self.assigned_task_id,
            subtask_id=self.assigned_subtask_id,
            content="Adversarial Reviewer initialized and ready for claim audit.",
            status=self.status,
        )
