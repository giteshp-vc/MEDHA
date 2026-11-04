import re
from typing import Any, List, Optional, Tuple
from models.schemas import ClaimKind
from models.verification_models import Claim
from utils.logger import get_logger

logger = get_logger("claim_validator")

# Red-flag superlative phrases that suggest unverified or exaggerated claims
UNSUBSTANTIATED_SUPERLATIVES = [
    r"100%\s+secure",
    r"zero\s+latency",
    r"completely\s+unhackable",
    r"flawless\s+scalability",
    r"infinite\s+throughput",
    r"guaranteed\s+100%",
    r"impossible\s+to\s+fail",
    r"no\s+single\s+point\s+of\s+failure\s+whatsoever",
]

# Patterns asserting empirical experiments or completed tests
EMPIRICAL_ASSERTION_PATTERNS = [
    r"\b(?:our\s+)?(?:benchmarks?|load\s*tests?|stress\s*tests?)\s+(?:proved|showed|demonstrated|measured|revealed)\b",
    r"\bwe\s+(?:measured|observed|tested|recorded|verified\s+empirically)\b",
    r"\b(?:pilot\s+study|internal\s+experiment|production\s+test)\s+(?:showed|proved|confirmed)\b",
    r"\bmeasured\s+(?:latency|throughput|write\s*speed)\s+of\b",
]


class ClaimValidator:
    """Pre-audit heuristics and consistency checks on claims before adversarial LLM scrutiny."""

    @staticmethod
    def validate_claim_structure(
        claim: Claim,
        evidence_items: Optional[List[Any]] = None,
    ) -> Tuple[bool, List[str]]:
        """Validate syntactic, structural, and evidence provenance validity of a claim."""
        issues: List[str] = []

        if not claim.claim_text or len(claim.claim_text.strip()) < 10:
            issues.append("Claim text is too short or empty to be verifiable.")

        text_lower = claim.claim_text.lower()

        if claim.claim_kind == ClaimKind.EMPIRICAL_RESULT:
            if evidence_items is not None:
                has_external_evidence = any(getattr(e, "is_external", False) for e in evidence_items)
            else:
                has_external_evidence = bool(claim.evidence_references)
            if not has_external_evidence and not any("empirical" in issue for issue in issues):
                issues.append(
                    "Claim is classified as an empirical result but has no external evidence artifact or test output."
                )

        # Check for ungrounded superlatives
        for pattern in UNSUBSTANTIATED_SUPERLATIVES:
            match = re.search(pattern, text_lower)
            if match:
                issues.append(f"Claim contains ungrounded absolute superlative: '{match.group(0)}'")

        # Check for empirical claims asserting completed tests, benchmarks, or measurements
        for pattern in EMPIRICAL_ASSERTION_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                has_external_evidence = False
                if evidence_items is not None:
                    # When evidence items are provided, strictly verify that at least one is external
                    has_external_evidence = any(
                        getattr(e, "is_external", False) for e in evidence_items
                    )
                elif claim.evidence_references:
                    # When evidence items aren't passed, check reference existence as basic fallback
                    has_external_evidence = True

                if not has_external_evidence:
                    user_statements_only = (
                        evidence_items is not None
                        and len(evidence_items) > 0
                        and all(getattr(e, "is_user_assertion", False) for e in evidence_items)
                    )
                    if user_statements_only:
                        issues.append(
                            f"Claim asserts empirical test result or measurement ('{match.group(0)}') "
                            f"supported only by unverified user statements or assertions, "
                            f"which cannot independently validate empirical claims without an uploaded artifact or benchmark data."
                        )
                    else:
                        issues.append(
                            f"Claim asserts empirical test result or measurement ('{match.group(0)}') "
                            f"without referencing any external evidence artifact or test output."
                        )
                    break

        # Check if the claim is merely subjective opinion
        if text_lower.startswith(("i think", "i feel", "it might be nice", "perhaps we could")):
            issues.append("Claim expresses subjective preference rather than a verifiable assertion.")

        is_valid = len(issues) == 0
        return is_valid, issues

    @staticmethod
    def has_sufficient_evidence_grounding(claim: Claim, evidence_count: int) -> bool:
        """Heuristic check: does the claim cite concrete evidence or verified reasoning?"""
        if evidence_count > 0:
            return True
        return len(claim.supporting_reasoning.strip()) > 40
