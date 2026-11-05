import asyncio
from typing import Any, Dict, List, Optional

from config.settings import MAX_CLAIMS_PER_TASK
from models.schemas import ClaimStatus, EvidenceSourceType
from models.verification_models import Claim, Evidence, RepairAttempt, ReviewFeedback, VerificationReport
from utils.helpers import utcnow_iso
from utils.logger import get_logger

logger = get_logger("evidence_ledger")


class EvidenceLedger:
    """Centralized verifiable repository maintaining all claims, supporting evidence, and verification statuses."""

    def __init__(self, task_id: str = "", max_claims: int = MAX_CLAIMS_PER_TASK):
        self.task_id = task_id
        self.max_claims = max_claims
        self._claims: Dict[str, Claim] = {}
        self._claim_text_index: Dict[str, str] = {}
        self._evidence: Dict[str, Evidence] = {}
        self._claim_evidence_map: Dict[str, List[str]] = {}
        self._lock = asyncio.Lock()

    async def register_claim(self, claim: Claim) -> str:
        """Register a claim in the ledger up to max_claims limit."""
        async with self._lock:
            key = self._normalize_claim_text(claim.claim_text)
            existing_id = self._claim_text_index.get(key)
            if existing_id:
                self._merge_claim(self._claims[existing_id], claim)
                return existing_id
            if len(self._claims) >= self.max_claims and claim.claim_id not in self._claims:
                logger.warning(
                    f"Ledger claim budget ({self.max_claims}) reached. Claim [{claim.claim_id}] was not registered."
                )
                return claim.claim_id

            self._claims[claim.claim_id] = claim
            self._claim_text_index[key] = claim.claim_id
            self._ensure_claim_provenance(claim)
            if claim.claim_id not in self._claim_evidence_map:
                self._claim_evidence_map[claim.claim_id] = []
            logger.debug(f"Registered claim [{claim.claim_id}] from [{claim.originating_role}]")
            return claim.claim_id

    async def register_claims_batch(self, claims: List[Claim]) -> List[str]:
        """Register multiple claims up to max_claims limit."""
        async with self._lock:
            ids = []
            for claim in claims:
                key = self._normalize_claim_text(claim.claim_text)
                existing_id = self._claim_text_index.get(key)
                if existing_id:
                    self._merge_claim(self._claims[existing_id], claim)
                    ids.append(existing_id)
                    continue
                if len(self._claims) >= self.max_claims and claim.claim_id not in self._claims:
                    logger.warning(
                        f"Ledger claim budget ({self.max_claims}) reached. Excess claim [{claim.claim_id}] was not registered."
                    )
                    continue
                self._claims[claim.claim_id] = claim
                self._claim_text_index[key] = claim.claim_id
                self._ensure_claim_provenance(claim)
                if claim.claim_id not in self._claim_evidence_map:
                    self._claim_evidence_map[claim.claim_id] = []
                ids.append(claim.claim_id)
            return ids

    @staticmethod
    def _normalize_claim_text(claim_text: str) -> str:
        return " ".join(claim_text.split()).casefold()

    @staticmethod
    def _ensure_claim_provenance(claim: Claim) -> None:
        if not claim.source_provenance:
            claim.source_provenance.append({
                "agent_id": claim.originating_agent,
                "role": claim.originating_role,
                "subtask_id": claim.subtask_id,
            })

    @classmethod
    def _merge_claim(cls, existing: Claim, incoming: Claim) -> None:
        cls._ensure_claim_provenance(existing)
        cls._ensure_claim_provenance(incoming)
        for source in incoming.source_provenance:
            if source not in existing.source_provenance:
                existing.source_provenance.append(source)
        for assumption in incoming.assumptions:
            if assumption not in existing.assumptions:
                existing.assumptions.append(assumption)
        for evidence_id in incoming.evidence_references:
            if evidence_id not in existing.evidence_references:
                existing.evidence_references.append(evidence_id)
        for reference in incoming.unverified_evidence_references:
            if reference not in existing.unverified_evidence_references:
                existing.unverified_evidence_references.append(reference)
        if not existing.supporting_reasoning and incoming.supporting_reasoning:
            existing.supporting_reasoning = incoming.supporting_reasoning

    async def register_evidence(
        self,
        evidence: Evidence,
        associated_claim_ids: Optional[List[str]] = None,
    ) -> str:
        """Register a piece of evidence and link to any associated claims."""
        async with self._lock:
            self._evidence[evidence.evidence_id] = evidence
            claim_ids = associated_claim_ids or evidence.associated_claim_ids or []
            for c_id in claim_ids:
                if c_id in self._claims:
                    if evidence.evidence_id not in self._claim_evidence_map.setdefault(c_id, []):
                        self._claim_evidence_map[c_id].append(evidence.evidence_id)
                    if evidence.evidence_id not in self._claims[c_id].evidence_references:
                        self._claims[c_id].evidence_references.append(evidence.evidence_id)
            return evidence.evidence_id

    async def link_claim_to_evidence(self, claim_id: str, evidence_id: str) -> None:
        """Associate existing claim and evidence."""
        async with self._lock:
            if claim_id in self._claims and evidence_id in self._evidence:
                if evidence_id not in self._claim_evidence_map.setdefault(claim_id, []):
                    self._claim_evidence_map[claim_id].append(evidence_id)
                if evidence_id not in self._claims[claim_id].evidence_references:
                    self._claims[claim_id].evidence_references.append(evidence_id)

    async def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Fetch a claim by ID."""
        async with self._lock:
            return self._claims.get(claim_id)

    async def get_evidence_for_claim(self, claim_id: str) -> List[Evidence]:
        """Retrieve all evidence items linked to a given claim."""
        async with self._lock:
            ev_ids = self._claim_evidence_map.get(claim_id, [])
            return [self._evidence[e_id] for e_id in ev_ids if e_id in self._evidence]

    async def update_claim_status(
        self,
        claim_id: str,
        status: ClaimStatus,
        reviewer_feedback: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Update verification status and audit feedback of a claim."""
        async with self._lock:
            if claim_id in self._claims:
                claim = self._claims[claim_id]
                old_status = claim.status
                claim.status = status
                claim.reviewer_feedback = reviewer_feedback or claim.reviewer_feedback
                claim.updated_at = utcnow_iso()
                claim.verification_history.append({
                    "from_status": old_status,
                    "to_status": status,
                    "feedback": reviewer_feedback,
                    "notes": notes,
                    "timestamp": utcnow_iso(),
                })
                logger.info(f"Claim [{claim_id}] status updated: {old_status} -> {status}")

    async def get_all_claims(self, status: Optional[ClaimStatus] = None) -> List[Claim]:
        """Retrieve all claims, optionally filtered by status."""
        async with self._lock:
            if status:
                return [c for c in self._claims.values() if c.status == status]
            return list(self._claims.values())

    async def get_all_evidence(self) -> List[Evidence]:
        """Retrieve all evidence stored in the ledger."""
        async with self._lock:
            return list(self._evidence.values())

    async def generate_report(
        self,
        review_decisions: Optional[List[ReviewFeedback]] = None,
        repair_history: Optional[List[RepairAttempt]] = None,
    ) -> VerificationReport:
        """Synthesize a complete verification audit report with rigorous support rate calculation.

        Support Rate Formula:
        Supported / (Supported + Rejected + Needs Revision + Unverifiable) * 100
        Excludes unreviewed/pending claims. Avoids division by zero on empty collections.
        """
        async with self._lock:
            claims = list(self._claims.values())
            total = len(claims)
            accepted = [c for c in claims if c.status == ClaimStatus.SUPPORTED]
            rejected = [c for c in claims if c.status == ClaimStatus.REJECTED]
            needs_rev = [c for c in claims if c.status == ClaimStatus.NEEDS_REVISION]
            unverifiable = [c for c in claims if c.status == ClaimStatus.UNVERIFIABLE]
            unreviewed = [c for c in claims if c.status in (ClaimStatus.PENDING, ClaimStatus.UNREVIEWED)]

            # Denominator: strictly reviewed claims (excluding PENDING and UNREVIEWED)
            reviewed_claims = [
                c for c in claims
                if c.status in (
                    ClaimStatus.SUPPORTED,
                    ClaimStatus.REJECTED,
                    ClaimStatus.NEEDS_REVISION,
                    ClaimStatus.UNVERIFIABLE,
                )
            ]
            reviewed_count = len(reviewed_claims)

            if reviewed_count > 0:
                support_rate = round((len(accepted) / reviewed_count) * 100.0, 1)
                unsupported_rate = round(((len(rejected) + len(unverifiable)) / reviewed_count) * 100.0, 1)
            else:
                support_rate = None
                unsupported_rate = None

            repairs = repair_history or []
            successful_repairs = sum(1 for r in repairs if r.status == ClaimStatus.SUPPORTED)
            repair_success_rate = (
                round((successful_repairs / len(repairs)) * 100.0, 1) if repairs else 0.0
            )

            if total == 0:
                summary_msg = "No claims were available for verification; support rate is unavailable."
            elif reviewed_count == 0:
                summary_msg = (
                    f"No claims were reviewed; support rate is unavailable. "
                    f"{len(unreviewed)} claim(s) remain unreviewed."
                )
            else:
                summary_msg = f"{len(accepted)} of {reviewed_count} reviewed claims supported ({support_rate:.1f}%)."
                if unreviewed:
                    summary_msg += f" {len(unreviewed)} claim(s) unreviewed due to audit interruption."

            return VerificationReport(
                total_claims=total,
                accepted_claims=accepted,
                rejected_claims=rejected,
                needs_revision_claims=needs_rev,
                unverifiable_claims=unverifiable,
                unreviewed_claims=unreviewed,
                review_decisions=review_decisions or [],
                repair_history=repairs,
                support_rate=support_rate,
                unsupported_rate=unsupported_rate,
                repair_success_rate=repair_success_rate,
                summary_message=summary_msg,
            )

    async def clear(self) -> None:
        """Reset ledger state."""
        async with self._lock:
            self._claims.clear()
            self._evidence.clear()
            self._claim_evidence_map.clear()
