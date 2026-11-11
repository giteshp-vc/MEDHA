import json
import time
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from agents.reviewer_agent import GENERAL_KNOWLEDGE_PREFIX
from models.agent_models import AgentConfig, AgentResponse
from models.deliberation_models import DeliberationReport
from models.schemas import AgentStatus, ClaimKind
from models.verification_models import Claim, RepairAttempt, VerificationReport
from utils.logger import get_logger
from verification.claim_extractor import classify_claim_kind

logger = get_logger("synthesizer_agent")


class SynthesizerAgent(BaseAgent):
    """Synthesizer agent responsible for combining accepted specialist findings into a transparent final answer."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        cfg = config or AgentConfig(
            role="Final Synthesizer",
            objective="Synthesize verified specialist outputs into a rigorous, transparent solution.",
            instructions=(
                "Synthesize accepted claims, explicitly exclude rejected claims, "
                "disclose assumptions, and preserve unresolved questions and uncertainties. "
                "Maintain strict evidence provenance: never present ungrounded targets as measured facts."
            ),
        )
        super().__init__(cfg, llm_client=llm_client)

    async def synthesize(
        self,
        task_description: str,
        specialist_results: List[Dict[str, Any]],
        verification_report: VerificationReport,
        unresolved_questions: List[str],
        partial_execution_notice: Optional[str] = None,
        deliberation_report: Optional[DeliberationReport] = None,
    ) -> str:
        """Produce the final transparent synthesized answer."""
        self.update_status(AgentStatus.EXECUTING)
        start_time = time.perf_counter()

        logger.info(f"Synthesizer synthesizing {len(specialist_results)} specialist results...")

        support_rate_text = (
            f"{verification_report.support_rate:.1f}%"
            if verification_report.support_rate is not None
            else "N/A (no claims reviewed)"
        )

        # Format accepted claims
        accepted_text = "\n".join(
            self._format_claim(c)
            + (
                " | basis: GENERAL KNOWLEDGE (not externally evidenced)"
                if GENERAL_KNOWLEDGE_PREFIX in (c.reviewer_feedback or "")
                else ""
            )
            for c in verification_report.accepted_claims
        ) or "None explicitly accepted."

        # Format rejected / excluded claims
        rejected_text = "\n".join(
            f"- [REJECTED] {self._format_claim(c)} | Reason: {c.reviewer_feedback or 'Unsupported'}"
            for c in verification_report.rejected_claims
        ) or "None rejected."

        # Format needs revision / unverifiable claims
        needs_rev_text = "\n".join(
            f"- [NEEDS REVISION] {self._format_claim(c)}"
            for c in verification_report.needs_revision_claims
        ) or "None pending revision."

        unverifiable_text = "\n".join(
            f"- [UNVERIFIABLE] {self._format_claim(c)} | Note: {c.reviewer_feedback or 'Unverifiable'}"
            for c in verification_report.unverifiable_claims
        ) or "None unverifiable."

        # Format unreviewed claims (interrupted before review)
        unreviewed_text = "\n".join(
            f"- [UNREVIEWED] {self._format_claim(c)} | Note: {c.reviewer_feedback or 'Not audited due to audit interruption'}"
            for c in verification_report.unreviewed_claims
        ) or "None unreviewed."

        # Format repairs
        repairs_text = "\n".join(
            f"- [Repaired Claim {r.claim_id}]: Original '{r.original_claim_text}' -> Revised: '{r.revised_claim_text}'"
            for r in verification_report.repair_history if r.revised_claim_text
        ) or "No repairs performed."

        # Format specialist deliverables
        specialist_text = ""
        for s in specialist_results:
            specialist_text += (
                f"\n\n### Specialist: {s.get('role')} (Subtask: {s.get('subtask_id')})\n"
                f"Status: {s.get('status', 'UNKNOWN')}\n"
                f"Execution diagnostic: {s.get('error') or 'None'}\n"
                f"Findings: {json.dumps(s.get('findings', []))}\n"
                f"Assumptions: {json.dumps(s.get('assumptions', []))}\n"
                f"Deliverable Details:\n{s.get('output', '')}\n"
            )

        questions_text = "\n".join(f"- {q}" for q in unresolved_questions) or "None recorded."

        partial_notice_str = f"\n\n### NOTICE:\n{partial_execution_notice}\n" if partial_execution_notice else ""
        deliberation_text = self._format_deliberation_report(deliberation_report, verification_report)

        system_instruction = (
            "You are the Lead Synthesizer for MEDHA (Multi-Agent Evidence-Driven Architecture).\n"
            "Your objective is to craft the definitive, verification-aware final answer to the user's task.\n\n"
            "SYNTHESIS GUIDELINES:\n"
            "1. Coherent Integration: Harmonize all validated contributions into an actionable, comprehensive solution.\n"
            "2. Evidence Traceability: Base factual conclusions strictly on ACCEPTED claims. Reference claim IDs where helpful.\n"
            "3. Explicit Exclusions: Do NOT include rejected claims as facts. Disclose what was rejected or unverified and why.\n"
            "4. Transparent Uncertainty: Do not paper over disagreements or unknowns. Emphasize unresolved questions.\n"
            "5. Verified Fidelity: Never present fallback text or unverified assertions as verified results. "
            f"Under Evidence & Verification Audit, you MUST report the official Claim Support Rate ({support_rate_text}) "
            f"and exact counts (Accepted: {len(verification_report.accepted_claims)}, Rejected: {len(verification_report.rejected_claims)}, "
            f"Unverifiable: {len(verification_report.unverifiable_claims)}, Unreviewed: {len(verification_report.unreviewed_claims)}). "
            f"Explain verification unavailability using this audit status: {verification_report.summary_message or 'No audit status available.'}\n"
            "6. Strict Evidence Provenance: Distinguish actual external evidence from assumptions and proposed design targets. "
            "Model agreement or reviewer acceptance alone confirms logical soundness, not empirical reality. "
            "Never present assumptions or proposed metrics as verified experimental measurements.\n"
            "7. Deliberation context is not evidence. Use the independent VerificationReport status for each claim; "
            "present unresolved disagreements separately and never promote consensus.\n"
            "8. Structure: Start with a '## Direct Answer' section that answers the ORIGINAL USER TASK exactly in the "
            "format and length it requests (e.g. 'one sentence each', a list, a number); keep it short. Then use clean "
            "Markdown headers (Evidence & Verification Audit, Operating Assumptions, Unresolved Edge Cases) and add "
            "Architecture/Solution Design only if the task asks for a design.\n"
            "9. Accepted claims marked 'basis: GENERAL KNOWLEDGE' were accepted as well-established knowledge without "
            "external evidence. Describe them as 'accepted as general knowledge', never as evidence-backed, "
            "evidence-grounded, or independently verified."
        )

        prompt = (
            f"ORIGINAL USER TASK:\n{task_description}\n"
            f"{partial_notice_str}\n"
            f"VERIFICATION STATUS:\n"
            f"- Official Support Rate: {support_rate_text}\n"
            f"- Verification Availability: {verification_report.summary_message or 'No audit status available.'}\n"
            f"- Accepted Claims ({len(verification_report.accepted_claims)}):\n{accepted_text}\n\n"
            f"- Rejected Claims ({len(verification_report.rejected_claims)}):\n{rejected_text}\n\n"
            f"- Needs Revision Claims ({len(verification_report.needs_revision_claims)}):\n{needs_rev_text}\n\n"
            f"- Unverifiable Claims ({len(verification_report.unverifiable_claims)}):\n{unverifiable_text}\n\n"
            f"- Unreviewed Claims ({len(verification_report.unreviewed_claims)}):\n{unreviewed_text}\n\n"
            f"- Repaired Claims ({len(verification_report.repair_history)}):\n{repairs_text}\n\n"
            f"DELIBERATION TRANSCRIPT (UNVERIFIED CONTEXT, NOT EVIDENCE):\n{deliberation_text}\n\n"
            f"SPECIALIST CONTRIBUTIONS:\n{specialist_text}\n\n"
            f"UNRESOLVED QUESTIONS & RISKS:\n{questions_text}\n\n"
            f"Please synthesize the final, comprehensive, and evidence-traceable response for the user."
        )

        try:
            with self.client.get_token_tracker().call_context(
                agent_id=self.agent_id,
                agent_role=self.role,
                stage="synthesis",
            ):
                final_answer = await self.client.generate_text(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    model=self.model_name,
                    temperature=self.temperature,
                )
        except Exception as e:
            logger.error(f"Synthesizer failed to generate final text: {e}")
            final_answer = self._build_deterministic_fallback(
                specialist_results=specialist_results,
                verification_report=verification_report,
                unresolved_questions=unresolved_questions,
                partial_execution_notice=partial_execution_notice,
                synthesis_error=e,
                deliberation_report=deliberation_report,
            )

        duration = round(time.perf_counter() - start_time, 3)
        self.last_execution_duration = duration
        self.update_status(AgentStatus.COMPLETED)

        logger.info(f"Synthesizer completed final answer in {duration}s ({len(final_answer)} chars).")
        return final_answer

    @staticmethod
    def _build_deterministic_fallback(
        specialist_results: List[Dict[str, Any]],
        verification_report: VerificationReport,
        unresolved_questions: List[str],
        partial_execution_notice: Optional[str],
        synthesis_error: Exception,
        deliberation_report: Optional[DeliberationReport] = None,
    ) -> str:
        def claim_list(claims: List[Claim]) -> str:
            formatted = []
            for claim in claims:
                provenance = claim.source_provenance or [{
                    "agent_id": claim.originating_agent,
                    "role": claim.originating_role,
                    "subtask_id": claim.subtask_id,
                }]
                sources = "; ".join(
                    f"{source.get('role', '')} ({source.get('agent_id', '')}, "
                    f"{source.get('subtask_id', '')})"
                    for source in provenance
                )
                evidence = ", ".join(claim.evidence_references) or "none"
                formatted.append(
                    f"- [{claim.claim_id}] status={claim.status}; sources=[{sources}]; "
                    f"evidence_references=[{evidence}]: {claim.claim_text}"
                )
            return "\n".join(formatted) or "- None."

        all_claims = (
            verification_report.accepted_claims
            + verification_report.rejected_claims
            + verification_report.needs_revision_claims
            + verification_report.unverifiable_claims
            + verification_report.unreviewed_claims
        )
        targets = [claim for claim in all_claims if claim.claim_kind == ClaimKind.PROPOSED_TARGET]
        measurements = [claim for claim in all_claims if claim.claim_kind == ClaimKind.EMPIRICAL_RESULT]
        target_texts = [claim.claim_text for claim in targets]
        measurement_texts = [claim.claim_text for claim in measurements]
        recommendations: List[str] = []
        assumptions: List[str] = []
        specialist_sections: List[str] = []

        for specialist in specialist_results:
            findings = specialist.get("findings", [])
            specialist_sections.append(
                f"### {specialist.get('role', 'Unknown specialist')} "
                f"({specialist.get('subtask_id', 'unknown')})\n"
                f"Status: {specialist.get('status', 'UNKNOWN')}\n\n"
                "Unverified findings:\n"
                + ("\n".join(f"- {finding}" for finding in findings) or "- None.")
                + "\n\nUnverified pre-identified claims:\n"
                + ("\n".join(f"- {text}" for text in specialist.get("claims", [])) or "- None.")
                + "\n\nUnverified deliverable:\n"
                + (specialist.get("output") or "None.")
            )
            assumptions.extend(specialist.get("assumptions", []))
            for text in list(findings) + list(specialist.get("claims", [])):
                kind = classify_claim_kind(text, specialist.get("assumptions", []))
                if kind == ClaimKind.PROPOSED_TARGET:
                    target_texts.append(text)
                elif kind == ClaimKind.EMPIRICAL_RESULT:
                    measurement_texts.append(text)
                elif kind == ClaimKind.RECOMMENDATION:
                    recommendations.append(text)
                elif kind == ClaimKind.ASSUMPTION:
                    assumptions.append(text)

        recommendations.extend(
            claim.claim_text for claim in all_claims if claim.claim_kind == ClaimKind.RECOMMENDATION
        )
        assumptions.extend(
            claim.claim_text for claim in all_claims if claim.claim_kind == ClaimKind.ASSUMPTION
        )

        support_rate = (
            f"{verification_report.support_rate:.1f}%"
            if verification_report.support_rate is not None
            else "N/A (no claims reviewed)"
        )
        notice = partial_execution_notice or "No separate interruption notice was recorded."
        questions = "\n".join(f"- {question}" for question in unresolved_questions) or "- None recorded."
        deliberation_text = SynthesizerAgent._format_deliberation_report(
            deliberation_report,
            verification_report,
        )

        return (
            "# Deterministic Synthesis Report\n\n"
            "**UNVERIFIED COMPILATION:** All specialist-generated content below is unverified unless explicitly "
            "listed in the review status section. Reviewer acceptance does not establish empirical results or "
            "replace missing external evidence.\n\n"
            + SynthesizerAgent._fallback_direct_answer(specialist_results)
            + "### Interruption\n"
            f"- Synthesis generation failed: {type(synthesis_error).__name__}: {synthesis_error}\n"
            f"- Verification / execution status: {verification_report.summary_message or 'Unavailable.'}\n"
            f"- Execution notice: {notice}\n\n"
            "### Specialist Contributions (Unverified)\n"
            + ("\n\n".join(specialist_sections) or "No specialist output was available.")
            + "\n\n### Assumptions (Unverified)\n"
            + ("\n".join(f"- {item}" for item in dict.fromkeys(assumptions)) or "- None recorded.")
            + "\n\n### Recommendations (Unverified)\n"
            + ("\n".join(f"- {item}" for item in dict.fromkeys(recommendations)) or "- None recorded.")
            + "\n\n### Proposed Performance Targets (Not Measurements)\n"
            + ("\n".join(f"- {text}" for text in dict.fromkeys(target_texts)) or "- None.")
            + "\n\n### Specialist-Reported Empirical Results (Unverified)\n"
            + ("\n".join(f"- {text}" for text in dict.fromkeys(measurement_texts)) or "- None.")
            + "\n\n### Verification Audit\n"
            + f"- Support rate: {support_rate}\n"
            + f"- Reviewed / supported: {claim_list(verification_report.accepted_claims)}\n"
            + f"- Rejected: {claim_list(verification_report.rejected_claims)}\n"
            + f"- Unverifiable: {claim_list(verification_report.unverifiable_claims)}\n"
            + f"- Unreviewed: {claim_list(verification_report.unreviewed_claims)}\n"
            + f"- Needs revision: {claim_list(verification_report.needs_revision_claims)}\n\n"
            "### Deliberation Transcript (Unverified Context; Not Evidence)\n"
            + deliberation_text
            + "\n\n"
            "### Unresolved Questions\n"
            + questions
        )

    @staticmethod
    def _fallback_direct_answer(specialist_results: List[Dict[str, Any]]) -> str:
        """Use the last completed specialist deliverable (usually the integrating step) as an unverified answer."""
        for specialist in reversed(specialist_results):
            output = (specialist.get("output") or "").strip()
            if not output or not str(specialist.get("status", "")).endswith("COMPLETED") or specialist.get("error"):
                continue
            body_lines = []
            for line in output.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    line = f"**{stripped.lstrip('#').strip()}**"
                elif stripped == "---":
                    line = ""
                body_lines.append(line.rstrip())
            return (
                "## Direct Answer\n"
                f"_Unverified: final synthesis failed, so this is the deliverable of "
                f"{specialist.get('role', 'a specialist')} ({specialist.get('subtask_id', 'unknown')})._\n\n"
                + "\n".join(body_lines).strip()
                + "\n\n---\n\n"
            )
        return ""

    @staticmethod
    def _format_claim(claim: Claim) -> str:
        provenance = claim.source_provenance or [{
            "agent_id": claim.originating_agent,
            "role": claim.originating_role,
            "subtask_id": claim.subtask_id,
        }]
        sources = "; ".join(
            f"{item.get('role', '')} ({item.get('agent_id', '')}, {item.get('subtask_id', '')})"
            for item in provenance
        )
        unverified_refs = ", ".join(claim.unverified_evidence_references) or "none"
        return (
            f"[Claim {claim.claim_id}] ({claim.originating_role}; sources: {sources}) "
            f"{claim.claim_text} | unverified agent references: {unverified_refs}"
        )

    @staticmethod
    def _format_deliberation_report(
        deliberation_report: Optional[DeliberationReport],
        verification_report: VerificationReport,
    ) -> str:
        if deliberation_report is None:
            return "No deliberation report was produced."

        claim_statuses = {
            claim.claim_id: claim.status
            for claim in (
                verification_report.accepted_claims
                + verification_report.rejected_claims
                + verification_report.needs_revision_claims
                + verification_report.unverifiable_claims
                + verification_report.unreviewed_claims
            )
        }
        lines = [f"Termination reason: {deliberation_report.termination_reason}"]
        for exchange in deliberation_report.exchanges:
            lines.append(
                f"- Exchange {exchange.exchange_id}; response message {exchange.response_message_id}; "
                f"trigger={exchange.trigger_type}; claim_reference={exchange.claim_reference or 'none'}"
            )
            if exchange.response_content:
                lines.append(f"  Response (unverified): {exchange.response_content}")
            for index, text in enumerate(exchange.introduced_claims):
                claim_id = exchange.introduced_claim_ids[index] if index < len(exchange.introduced_claim_ids) else None
                status = claim_statuses.get(claim_id) if claim_id else None
                label = str(status) if status is not None else "UNREGISTERED / UNREVIEWED"
                lines.append(f"  Deliberation-introduced claim [{claim_id or 'no ledger ID'}; {label}]: {text}")
            lines.extend(f"  Clarification (unverified): {text}" for text in exchange.clarifications)
            if exchange.evidence_references:
                lines.append(
                    "  Agent-supplied evidence references (UNVERIFIED, not ledger evidence): "
                    + ", ".join(exchange.evidence_references)
                )
            lines.extend(f"  Unresolved question: {item}" for item in exchange.unresolved_questions)
            if exchange.disagreement:
                lines.append("  Unresolved disagreement: specialist disagreement is not a factual conclusion.")
            if exchange.diagnostic:
                lines.append(f"  Diagnostic: {exchange.diagnostic}")
        lines.extend(
            f"- Suppressed {item.trigger_type} trigger for {item.claim_reference or 'unreferenced claim'}: {item.reason}"
            for item in deliberation_report.suppressed_triggers
        )
        lines.extend(f"- Unresolved disagreement: {item}" for item in deliberation_report.unresolved_disagreements)
        lines.extend(f"- Unresolved deliberation question: {item}" for item in deliberation_report.unresolved_questions)
        lines.extend(f"- Deliberation diagnostic: {item}" for item in deliberation_report.diagnostics)
        return "\n".join(lines)

    async def execute(self, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Generic execution implementation to satisfy BaseAgent contract."""
        self.update_status(AgentStatus.COMPLETED)
        return AgentResponse(
            agent_id=self.agent_id,
            role=self.role,
            task_id=self.assigned_task_id,
            subtask_id=self.assigned_subtask_id,
            content="Synthesizer initialized.",
            status=self.status,
        )
