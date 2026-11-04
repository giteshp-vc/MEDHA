import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from config.settings import MAX_CLAIMS_PER_TASK
from core.llm_client import BaseLLMClient, LLMClient
from models.schemas import ClaimKind, ClaimStatus
from models.task_models import TaskResult
from models.verification_models import Claim
from utils.helpers import generate_id
from utils.logger import get_logger

logger = get_logger("claim_extractor")


class ExtractedClaimsSchema(BaseModel):
    """Pydantic schema for structured claim extraction."""

    claims: List[str] = Field(description="Discrete, atomic, verifiable factual or technical assertions")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions underlying the assertions")


@dataclass
class ClaimExtractionOutcome:
    claims: List[Claim]
    source: str
    error: Optional[Exception] = None


class ClaimExtractor:
    """Extracts atomic, testable claims from agent deliverables and task results with strict evidence provenance."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None, max_claims: int = MAX_CLAIMS_PER_TASK):
        self.client = llm_client or LLMClient.get_instance()
        self.max_claims = max_claims

    async def extract_claims_from_result(
        self,
        result: TaskResult,
        task_id: str = "",
        max_claims_to_extract: Optional[int] = None,
    ) -> List[Claim]:
        """Compatibility wrapper returning claims or re-raising extraction errors."""
        outcome = await self.extract_claims_with_diagnostics(
            result=result,
            task_id=task_id,
            max_claims_to_extract=max_claims_to_extract,
        )
        if outcome.error:
            raise outcome.error
        return outcome.claims

    async def extract_claims_with_diagnostics(
        self,
        result: TaskResult,
        task_id: str = "",
        max_claims_to_extract: Optional[int] = None,
    ) -> ClaimExtractionOutcome:
        """Extract claims and return failures explicitly instead of losing the specialist output."""
        claims: List[Claim] = []

        # Structured specialist claims are already atomic enough to register directly.
        if result.claims:
            for claim_text in result.claims:
                text_clean = claim_text.strip()
                if text_clean:
                    claims.append(
                        Claim(
                            claim_id=generate_id("claim"),
                            claim_text=text_clean,
                            originating_agent=result.agent_id,
                            originating_role=result.role,
                            task_id=task_id or result.task_id,
                            subtask_id=result.subtask_id,
                            supporting_reasoning=result.output[:300],
                            assumptions=result.assumptions,
                            claim_kind=classify_claim_kind(text_clean, result.assumptions),
                            source_provenance=[_source_provenance(result)],
                            status=ClaimStatus.PENDING,
                        )
                    )
            logger.info(f"Adopted {len(claims)} pre-identified claims from [{result.role}]")
            return ClaimExtractionOutcome(claims=claims, source="specialist_structured")

        # Otherwise, extract atomic claims from the deliverable content using LLM
        if not result.output:
            return ClaimExtractionOutcome(claims=[], source="empty_output")

        limit = max_claims_to_extract or self.max_claims

        prompt = (
            f"Extract discrete, verifiable factual, performance, or architectural claims from this text:\n\n"
            f"TEXT:\n{result.output}\n\n"
            f"STRICT EVIDENCE PROVENANCE RULES:\n"
            f"1. Extract up to {limit} atomic, testable assertions that could be empirically true or false.\n"
            f"2. Separate established factual statements from acknowledged assumptions or proposed targets.\n"
            f"3. Do NOT invent URLs, citations, benchmarks, or test results not in the text.\n"
            f"4. Label proposed latency/uptime targets as targets, never as measured or achieved results.\n"
            f"5. Do not convert recommendations or assumptions into established facts."
        )

        try:
            extracted = await self.client.generate_structured(
                prompt=prompt,
                response_schema=ExtractedClaimsSchema,
                temperature=0.1,
            )
            for c_text in extracted.claims[:limit]:
                claims.append(
                    Claim(
                        claim_id=generate_id("claim"),
                        claim_text=c_text,
                        originating_agent=result.agent_id,
                        originating_role=result.role,
                        task_id=task_id or result.task_id,
                        subtask_id=result.subtask_id,
                        supporting_reasoning=result.output[:300],
                        assumptions=extracted.assumptions or result.assumptions,
                        claim_kind=classify_claim_kind(c_text, extracted.assumptions or result.assumptions),
                        source_provenance=[_source_provenance(result)],
                        status=ClaimStatus.PENDING,
                    )
                )
        except Exception as e:
            logger.warning(
                f"Structured claim extraction failed for [{result.subtask_id}] "
                f"({type(e).__name__}: {e}); preserving {len(claims)} discovered claim(s)."
            )
            return ClaimExtractionOutcome(claims=claims, source="failed", error=e)

        logger.info(f"Extracted {len(claims)} claims from [{result.role}]")
        return ClaimExtractionOutcome(claims=claims, source="llm_structured")


def _source_provenance(result: TaskResult) -> Dict[str, str]:
    return {
        "agent_id": result.agent_id,
        "role": result.role,
        "subtask_id": result.subtask_id,
    }


_PERFORMANCE_TERMS = re.compile(
    r"\b(latency|uptime|availability|throughput|accuracy|precision|recall|response time|error rate)\b",
    re.IGNORECASE,
)
_TARGET_TERMS = re.compile(
    r"\b(target|goal|objective|aim|proposed|designed to|should(?:\s+(?:achieve|remain|maintain|be|stay))?)\b",
    re.IGNORECASE,
)
_EMPIRICAL_TERMS = re.compile(
    r"\b(measured|observed|achieved|recorded|benchmarked|load[- ]tested|pilot showed|production showed)\b",
    re.IGNORECASE,
)
_ASSUMPTION_TERMS = re.compile(r"\b(assume|assumption|assumed|estimated|estimate)\b", re.IGNORECASE)
_RECOMMENDATION_TERMS = re.compile(
    r"\b(recommend(?:ation|ed|s)?|should|propose(?:d)?|consider|design recommendation|"
    r"use|adopt|deploy|configure|implement)\b",
    re.IGNORECASE,
)


def classify_claim_kind(claim_text: str, assumptions: Optional[List[str]] = None) -> ClaimKind:
    """Classify assertion framing without treating targets or advice as measured outcomes."""
    text = claim_text.strip()
    if _PERFORMANCE_TERMS.search(text) and _TARGET_TERMS.search(text):
        return ClaimKind.PROPOSED_TARGET
    if _ASSUMPTION_TERMS.search(text) or any(
        text.casefold() == assumption.strip().casefold() for assumption in (assumptions or [])
    ):
        return ClaimKind.ASSUMPTION
    if _EMPIRICAL_TERMS.search(text):
        return ClaimKind.EMPIRICAL_RESULT
    if _RECOMMENDATION_TERMS.search(text):
        return ClaimKind.RECOMMENDATION
    return ClaimKind.FACTUAL_ASSERTION
