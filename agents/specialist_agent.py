import json
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from agents.base_agent import BaseAgent
from models.agent_models import AgentConfig, AgentResponse
from models.schemas import AgentStatus
from utils.helpers import parse_json_safely
from utils.logger import get_logger
from utils.retry import LLMError, LLMStructuredOutputError

logger = get_logger("specialist_agent")


class SpecialistOutputSchema(BaseModel):
    """Pydantic schema requested from LLM for specialist agent execution."""

    deliverable: str = Field(description="Comprehensive analysis and technical deliverable")
    findings: List[str] = Field(default_factory=list, description="Core proposals, decisions, or results")
    claims: List[str] = Field(default_factory=list, description="Factual assertions or quantitative statements requiring verification")
    evidence_references: List[str] = Field(default_factory=list, description="Sources, benchmarks, or logical derivations supporting claims")
    assumptions: List[str] = Field(default_factory=list, description="Underlying assumptions made")
    unresolved_questions: List[str] = Field(default_factory=list, description="Open questions, risks, or caveats")

    @field_validator(
        "findings",
        "claims",
        "evidence_references",
        "assumptions",
        "unresolved_questions",
        mode="before",
    )
    @classmethod
    def normalize_text_lists(cls, value: Any) -> List[str]:
        normalized: List[str] = []

        def append_value(item: Any) -> None:
            if item is None:
                return
            if isinstance(item, (list, tuple)):
                for nested_item in item:
                    append_value(nested_item)
            elif isinstance(item, dict):
                normalized.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            elif isinstance(item, str):
                if item.strip():
                    normalized.append(item)
            else:
                normalized.append(str(item))

        append_value(value)
        return normalized


class SpecialistOutputRecoveryError(RuntimeError):
    """All bounded recovery paths for a specialist response failed."""


class SpecialistAgent(BaseAgent):
    """Dynamic specialist agent created on the fly based on planner requirements."""

    def __init__(
        self,
        config: AgentConfig,
        llm_client: Optional[Any] = None,
    ):
        super().__init__(config, llm_client=llm_client)

    def _build_system_instruction(self) -> str:
        return (
            f"You are a dedicated, top-tier domain specialist operating as: {self.role}.\n"
            f"Your specific objective is: {self.objective}\n"
            f"Guidelines and Instructions: {self.instructions or 'Provide rigorous, evidence-grounded analysis.'}\n\n"
            f"CRITICAL VERIFICATION RULES:\n"
            f"1. Clearly separate established facts from working assumptions and inferences.\n"
            f"2. Every concrete assertion (performance metrics, architectural limits, trade-offs, security postures) "
            f"must be presented as a verifiable claim.\n"
            f"3. Do not invent fake research papers, URLs, or benchmarks. If a number is an estimate, state it as an assumption.\n"
            f"4. Highlight any edge cases, risks, or unresolved questions that require attention."
        )

    def _build_execution_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        context_str = ""
        if context:
            # If there are upstream dependencies or shared context
            upstream_results = context.get("upstream_results", [])
            if upstream_results:
                context_str += "\n\n### Upstream Findings from Collaborating Specialists:\n"
                for res in upstream_results:
                    context_str += (
                        f"- Specialist: {res.get('role')} (Subtask: {res.get('subtask_id')}):\n"
                        f"  Summary: {res.get('output', '')[:400]}\n"
                        f"  Key Findings: {json.dumps(res.get('findings', []))}\n"
                    )

            shared_notes = context.get("shared_notes", "")
            if shared_notes:
                context_str += f"\n\n### Shared Task Context:\n{shared_notes}\n"

        prompt = (
            f"Execute your assigned task:\n"
            f"Role: {self.role}\n"
            f"Objective: {self.objective}\n"
            f"{context_str}\n\n"
            f"Provide your response formatted as a structured JSON object with fields:\n"
            f"- 'deliverable': comprehensive technical explanation and recommendations\n"
            f"- 'findings': list of key analytical conclusions\n"
            f"- 'claims': list of specific factual, architectural, or performance claims requiring verification\n"
            f"- 'evidence_references': list of supporting reasons, benchmarks, or domain rationales for the claims\n"
            f"- 'assumptions': list of explicitly acknowledged assumptions\n"
            f"- 'unresolved_questions': list of open questions or unresolved risks"
        )
        return prompt

    async def _recover_structured_output(
        self,
        prompt: str,
        system_instruction: str,
        original_error: LLMStructuredOutputError,
    ) -> SpecialistOutputSchema:
        correction_error: Optional[Exception] = None
        correction_raw_text = ""

        correction_prompt = (
            f"{prompt}\n\nThe previous response failed schema validation: "
            f"{original_error.validation_error}\n"
            "Correct the JSON to match the required schema exactly. Preserve every meaningful finding and claim; "
            "do not invent missing information. Previous response:\n"
            f"{original_error.raw_text}"
        )

        try:
            return await self.client.generate_structured(
                prompt=correction_prompt,
                response_schema=SpecialistOutputSchema,
                system_instruction=system_instruction,
                model=self.model_name,
                temperature=self.temperature,
                max_retries=0,
            )
        except Exception as error:
            correction_error = error
            if isinstance(error, LLMStructuredOutputError):
                correction_raw_text = error.raw_text

        parse_errors: List[str] = []
        candidates = [text for text in (correction_raw_text, original_error.raw_text) if text]
        for candidate in candidates:
            try:
                return SpecialistOutputSchema.model_validate(parse_json_safely(candidate))
            except Exception as error:
                parse_errors.append(f"{type(error).__name__}: {error}")

        if isinstance(correction_error, LLMError) and not isinstance(
            correction_error, LLMStructuredOutputError
        ):
            details = "; ".join(parse_errors) or "No parseable original response was available."
            raise SpecialistOutputRecoveryError(
                "Structured output validation failed "
                f"[{type(original_error).__name__}: {original_error}]; schema-correction/provider failure "
                f"[{type(correction_error).__name__}: {correction_error}]; fallback parsing failed [{details}]"
            ) from correction_error

        try:
            fallback_text = await self.client.generate_text(
                prompt=correction_prompt + "\nReturn ONLY one valid JSON object.",
                system_instruction=system_instruction,
                model=self.model_name,
                temperature=self.temperature,
                max_retries=0,
            )
            return SpecialistOutputSchema.model_validate(parse_json_safely(fallback_text))
        except Exception as fallback_error:
            details = "; ".join(parse_errors) or "No parseable original response was available."
            raise SpecialistOutputRecoveryError(
                "Structured output validation failed "
                f"[{type(original_error).__name__}: {original_error}]; schema-correction failed "
                f"[{type(correction_error).__name__}: {correction_error}]; fallback parsing failed "
                f"[{type(fallback_error).__name__}: {fallback_error}]; "
                f"original-response parsing errors: {details}"
            ) from fallback_error

    async def execute(self, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Run the specialist agent against Gemini LLM and parse structured output."""
        self.update_status(AgentStatus.EXECUTING)
        start_time = time.perf_counter()

        system_instruction = self._build_system_instruction()
        prompt = self._build_execution_prompt(context)

        logger.info(f"Agent [{self.role} ({self.agent_id})] executing objective: {self.objective[:70]}...")

        try:
            try:
                structured_out = await self.client.generate_structured(
                    prompt=prompt,
                    response_schema=SpecialistOutputSchema,
                    system_instruction=system_instruction,
                    model=self.model_name,
                    temperature=self.temperature,
                )
            except LLMStructuredOutputError as validation_error:
                logger.warning(
                    f"Agent [{self.role}] structured output validation failed; attempting bounded recovery."
                )
                structured_out = await self._recover_structured_output(
                    prompt, system_instruction, validation_error
                )

            deliverable = structured_out.deliverable
            findings = structured_out.findings
            claims = structured_out.claims
            ev_refs = structured_out.evidence_references
            assumptions = structured_out.assumptions
            questions = structured_out.unresolved_questions
        except Exception as error:
            logger.error(
                f"Agent [{self.role}] execution failed [{type(error).__name__}]: {error}"
            )
            self.update_status(AgentStatus.FAILED)
            self.last_execution_duration = round(time.perf_counter() - start_time, 3)
            raise

        # Update internal state
        for f in findings:
            self.record_finding(f)
        for c in claims:
            self.record_claim(c)
        for a in assumptions:
            self.record_assumption(a)
        for r in ev_refs:
            self.record_evidence_ref(r)
        for q in questions:
            self.record_unresolved_question(q)

        duration = round(time.perf_counter() - start_time, 3)
        self.last_execution_duration = duration
        self.update_status(AgentStatus.COMPLETED)

        logger.info(
            f"Agent [{self.role}] completed in {duration}s. "
            f"Claims: {len(self.claims)}, Findings: {len(self.findings)}"
        )

        return AgentResponse(
            agent_id=self.agent_id,
            role=self.role,
            task_id=self.assigned_task_id,
            subtask_id=self.assigned_subtask_id,
            content=deliverable,
            findings=self.findings,
            claims=self.claims,
            evidence_references=self.evidence_references,
            assumptions=self.assumptions,
            unresolved_questions=self.unresolved_questions,
            status=self.status,
            duration_seconds=duration,
        )

