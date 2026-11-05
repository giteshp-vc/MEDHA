import asyncio
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from communication.message_bus import MessageBus
from communication.deliberation_coordinator import DeliberationCoordinator
from communication.deliberation_policy import temporary_claim_reference
from communication.shared_context import SharedContext
from config.settings import (
    EXECUTION_TIMEOUT_SECONDS,
    MAX_CONCURRENT_AGENTS,
    MAX_REPAIR_ITERATIONS,
    validate_settings,
)
from core.agent_factory import AgentFactory
from core.execution_engine import ExecutionEngine
from core.llm_client import BaseLLMClient, create_llm_client
from core.planner import TaskPlanner
from core.workflow import WorkflowState, WorkflowTracker
from models.execution_policy import OrchestrationPolicy
from models.deliberation_models import DeliberationReport, DeliberationTerminationReason
from models.frozen_plan import FrozenTaskPlan
from models.provider_call import AccountingContext, AccountingSnapshot, ProviderCallRecord
from models.schemas import BaseSchema, ClaimStatus, EvidenceSourceType, TaskStatus
from models.task_models import TaskPlan, TaskResult
from models.verification_models import Claim, Evidence, RepairAttempt, ReviewFeedback
from utils.helpers import generate_id, utcnow_iso
from utils.logger import ExecutionRecord, get_logger
from utils.retry import LLMBudgetExceededError, LLMQuotaExhaustedError
from utils.token_tracker import TokenTracker
from verification.adversarial_reviewer import AdversarialReviewer
from verification.claim_extractor import ClaimExtractor
from verification.evidence_ledger import EvidenceLedger
from verification.repair_engine import RepairEngine

logger = get_logger("orchestrator")


def _format_deliberation_context(report: Optional[DeliberationReport]) -> str:
    """Format the transcript as unverified reviewer context, never as evidence."""
    if report is None:
        return ""

    trigger_by_exchange = {trigger.exchange_id: trigger for trigger in report.triggers}
    lines = [
        "DELIBERATION TRANSCRIPT: UNVERIFIED CONTEXT ONLY; NOT REGISTERED EVIDENCE.",
        f"Termination reason: {report.termination_reason}",
    ]
    for exchange in report.exchanges:
        trigger = trigger_by_exchange.get(exchange.exchange_id)
        lines.append(
            f"Exchange {exchange.exchange_id}; round={exchange.round_number}; "
            f"type={exchange.trigger_type}; initiator={exchange.initiator_agent_id}; "
            f"target={exchange.target_agent_id}; claim_reference={exchange.claim_reference or 'none'}; "
            f"correlation_id={exchange.correlation_id}."
        )
        if trigger is not None:
            lines.append(f"- Challenge/request reason (unverified): {trigger.reason}")
            lines.append(f"- Target-selection rationale: {trigger.target_selection_rationale}")
        lines.append(
            f"- Message IDs: request={exchange.request_message_id}; response={exchange.response_message_id}"
        )
        if exchange.response_content:
            lines.append(f"- Response ({exchange.response_type}, unverified): {exchange.response_content}")
        for index, claim_text in enumerate(exchange.introduced_claims):
            claim_id = exchange.introduced_claim_ids[index] if index < len(exchange.introduced_claim_ids) else None
            claim_ref = (
                exchange.introduced_claim_references[index]
                if index < len(exchange.introduced_claim_references)
                else "unmapped"
            )
            lines.append(
                f"- Introduced claim [{claim_id or 'unregistered'}; temporary_ref={claim_ref}] "
                f"(unverified until reviewed): {claim_text}"
            )
        lines.extend(f"- Clarification (unverified): {text}" for text in exchange.clarifications)
        lines.extend(f"- Assumption (unverified): {text}" for text in exchange.assumptions)
        if exchange.evidence_references:
            lines.append(
                "- Agent-supplied evidence references (unverified, not ledger evidence): "
                + ", ".join(exchange.evidence_references)
            )
        lines.extend(f"- Unresolved question: {question}" for question in exchange.unresolved_questions)
        if exchange.disagreement:
            lines.append("- Disagreement remains unresolved; disagreement is not evidence.")
        if exchange.diagnostic:
            lines.append(f"- Diagnostic: {exchange.diagnostic}")

    lines.extend(
        f"Suppressed trigger: type={item.trigger_type}; claim={item.claim_reference or 'none'}; "
        f"source={item.source_agent_id}; target={item.target_agent_id}; reason={item.reason}"
        for item in report.suppressed_triggers
    )
    lines.extend(f"Unresolved disagreement: {item}" for item in report.unresolved_disagreements)
    lines.extend(f"Unresolved question: {item}" for item in report.unresolved_questions)
    lines.extend(f"Deliberation diagnostic: {item}" for item in report.diagnostics)
    return "\n".join(lines)


class MedhaResult(BaseSchema):
    """Complete, transparent outcome returned by the MEDHA multi-agent system."""

    execution_id: str
    task: str
    final_answer: str
    plan_artifact: Optional[FrozenTaskPlan] = None
    accounting: Optional[AccountingSnapshot] = None
    provider_call_records: List[ProviderCallRecord] = Field(default_factory=list)
    status: str = "SUCCESS"  # Truthful status: SUCCESS, PARTIAL_FAILURE, FAILED
    duration_seconds: float = 0.0
    token_usage: Dict[str, Any] = Field(default_factory=dict)
    generated_agents: List[Dict[str, Any]] = Field(default_factory=list)
    task_results: List[Dict[str, Any]] = Field(default_factory=list)
    deliberation_report: Optional[DeliberationReport] = None
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    review_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    repair_history: List[Dict[str, Any]] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    claim_extraction_errors: List[str] = Field(default_factory=list)
    verification_report: Dict[str, Any] = Field(default_factory=dict)
    workflow_stages: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


class MedhaOrchestrator:
    """Master controller managing the dynamic lifecycle of MEDHA."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        max_concurrency: int = MAX_CONCURRENT_AGENTS,
        max_repair_iterations: int = MAX_REPAIR_ITERATIONS,
        timeout_seconds: float = EXECUTION_TIMEOUT_SECONDS,
    ):
        if llm_client is not None:
            self.client = llm_client
            self.tracker = llm_client.get_token_tracker()
        else:
            self.tracker = TokenTracker(name="medha_orchestrator")
            self.client = create_llm_client(token_tracker=self.tracker)
        self.max_concurrency = max_concurrency
        self.max_repair_iterations = max_repair_iterations
        self.timeout_seconds = timeout_seconds

    async def run_task(self, user_input: str) -> MedhaResult:
        execution_id = generate_id("exec")
        with self.tracker.call_context(execution_id=execution_id):
            async with self.tracker.task_budget():
                return await self._run_task(
                    user_input,
                    policy=OrchestrationPolicy(),
                    execution_id=execution_id,
                )

    async def prepare_frozen_plan(
        self,
        user_input: str,
        accounting_context: Optional[AccountingContext] = None,
    ) -> FrozenTaskPlan:
        """Generate and validate one reusable, deterministic plan artifact for a task."""
        planning_execution_id = generate_id("exec_plan")
        base_context = accounting_context or AccountingContext()
        context_values = base_context.model_dump(exclude_none=True)
        context_values.pop("trial_id", None)
        context_values.pop("treatment_id", None)
        context_values.update(
            execution_id=planning_execution_id,
            scope_kind="plan_preparation",
        )
        plan_context = AccountingContext.model_validate(context_values)
        execution_record = ExecutionRecord(
            execution_id=planning_execution_id,
            task_description=user_input,
        )
        preparation_accounting: Optional[AccountingSnapshot] = None

        def persist_plan_record(status: str, error: Optional[BaseException] = None) -> None:
            provider_calls = self.tracker.get_provider_call_records(planning_execution_id)
            execution_record.metadata["provider_calls"] = [
                call.model_dump(mode="json") for call in provider_calls
            ]
            execution_record.metadata["provider_call_records_dropped"] = (
                preparation_accounting.dropped_provider_call_records
                if preparation_accounting is not None
                else self.tracker.get_provider_call_records_dropped()
            )
            execution_record.metadata["accounting"] = (
                preparation_accounting.model_dump(mode="json")
                if preparation_accounting is not None
                else None
            )
            if error is not None:
                execution_record.record_error(
                    component="plan_preparation",
                    message=f"{type(error).__name__} during plan preparation.",
                )
            execution_record.finish(
                status=status,
                token_usage=(preparation_accounting.token_usage if preparation_accounting else None),
            )
            execution_record.save()

        try:
            async with self.tracker.accounting_session(plan_context) as accounting_scope:
                with self.tracker.call_context(
                    execution_id=planning_execution_id,
                    stage="planning",
                ):
                    async with self.tracker.task_budget():
                        try:
                            planner = TaskPlanner(llm_client=self.client)
                            with self.tracker.operation("planning"):
                                plan = await planner.create_plan(
                                    task_description=user_input,
                                    max_subtasks=self.max_concurrency,
                                )
                            preparation_accounting = accounting_scope.snapshot()
                        except asyncio.CancelledError:
                            preparation_accounting = accounting_scope.snapshot()
                            raise
                        except Exception:
                            preparation_accounting = accounting_scope.snapshot()
                            raise
            frozen_plan = FrozenTaskPlan.from_plan(
                user_input,
                plan,
                preparation_accounting=preparation_accounting,
            )
            execution_record.metadata["frozen_plan"] = frozen_plan.model_dump(mode="json")
            persist_plan_record("SUCCESS")
            return frozen_plan
        except asyncio.CancelledError as error:
            try:
                error.preparation_accounting = preparation_accounting
            except Exception:
                pass
            persist_plan_record("CANCELLED", error)
            raise
        except Exception as error:
            try:
                error.preparation_accounting = preparation_accounting
            except Exception:
                pass
            persist_plan_record("FAILED", error)
            raise

    async def run_planned_task(
        self,
        user_input: str,
        frozen_plan: FrozenTaskPlan,
        policy: OrchestrationPolicy,
        accounting_context: Optional[AccountingContext] = None,
    ) -> MedhaResult:
        """Run the multi-agent lifecycle against a previously prepared frozen plan."""
        if not policy.verification_enabled or not policy.synthesis_enabled:
            raise ValueError(
                "Planned multi-agent runs require verification and synthesis; "
                "the baseline runs through SingleAgentBaseline."
            )
        frozen_plan.load_plan(user_input)
        execution_id = generate_id("exec")
        base_context = accounting_context or AccountingContext()
        trial_context = AccountingContext.model_validate({
            **base_context.model_dump(exclude_none=True),
            "execution_id": execution_id,
            "scope_kind": "trial_execution",
        })
        with self.tracker.call_context(execution_id=execution_id):
            async with self.tracker.accounting_session(trial_context) as accounting_scope:
                try:
                    async with self.tracker.task_budget():
                        result = await self._run_task(
                            user_input,
                            frozen_plan=frozen_plan,
                            policy=policy,
                            execution_id=execution_id,
                        )
                        trial_accounting = accounting_scope.snapshot()
                    result.accounting = trial_accounting
                    result.provider_call_records = result.accounting.provider_calls
                    result.token_usage = result.accounting.token_usage
                    return result
                except asyncio.CancelledError as error:
                    try:
                        error.accounting_snapshot = accounting_scope.snapshot()
                    except Exception:
                        pass
                    raise

    async def _run_task(
        self,
        user_input: str,
        frozen_plan: Optional[FrozenTaskPlan] = None,
        policy: Optional[OrchestrationPolicy] = None,
        execution_id: Optional[str] = None,
    ) -> MedhaResult:
        """Execute the end-to-end evidence-driven multi-agent lifecycle for a given user task."""
        policy = policy or OrchestrationPolicy()
        execution_id = execution_id or generate_id("exec")
        exec_record = ExecutionRecord(execution_id=execution_id, task_description=user_input)
        workflow = WorkflowTracker(execution_id=execution_id)
        start_perf = time.perf_counter()

        logger.info(f"=== Starting MEDHA Execution [{execution_id}] ===")
        logger.info(f"Task: '{user_input[:80]}...'")

        ledger = EvidenceLedger(task_id=execution_id)
        review_feedbacks: List[ReviewFeedback] = []
        repair_attempts: List[RepairAttempt] = []
        claim_extraction_errors: List[str] = []
        quota_interrupted_note: Optional[str] = None
        task_results: List[TaskResult] = []
        deliberation_report: Optional[DeliberationReport] = None
        plan_artifact = frozen_plan
        agent_factory = AgentFactory(llm_client=self.client)

        try:
            # -------------------------------------------------------------
            # Stage 1: DYNAMIC TASK PLANNING & SPECIALIST IDENTIFICATION
            # -------------------------------------------------------------
            workflow.transition_to(WorkflowState.PLANNING)
            if frozen_plan is None:
                planner = TaskPlanner(llm_client=self.client)
                with self.tracker.operation("planning"):
                    plan: TaskPlan = await planner.create_plan(
                        task_description=user_input,
                        max_subtasks=self.max_concurrency,
                    )
            else:
                plan = frozen_plan.load_plan(user_input)
            if plan_artifact is not None:
                exec_record.metadata["frozen_plan"] = plan_artifact.model_dump(mode="json")

            # -------------------------------------------------------------
            # Stage 2: AGENT FACTORY & DEPENDENCY-AWARE EXECUTION
            # -------------------------------------------------------------
            workflow.transition_to(
                WorkflowState.EXECUTING_SUBTASKS,
                details={"planned_subtasks": [s.role for s in plan.subtasks]},
            )
            message_bus = MessageBus()
            shared_context = SharedContext(task_id=execution_id, task_description=user_input)

            execution_engine = ExecutionEngine(
                max_concurrency=self.max_concurrency,
                timeout_seconds=self.timeout_seconds,
            )
            with self.tracker.operation("specialist"):
                task_results = await execution_engine.execute_plan(
                    plan=plan,
                    agent_factory=agent_factory,
                    message_bus=message_bus,
                    shared_context=shared_context,
                    task_id=execution_id,
                    execution_policy=policy.execution,
                )

            # Log agent activity to execution record
            for r in task_results:
                exec_record.log_agent_activity(
                    agent_id=r.agent_id,
                    role=r.role,
                    task_id=r.subtask_id,
                    status=r.status,
                    duration_seconds=r.duration_seconds,
                    error=r.error,
                )
            for message in message_bus.get_history(task_id=execution_id):
                exec_record.log_message(message.model_dump(mode="json"))

            # -------------------------------------------------------------
            # Evaluate Specialist Subtask Outcomes
            # -------------------------------------------------------------
            completed_tasks = [r for r in task_results if str(r.status) == "COMPLETED" and not r.error]
            failed_tasks = [r for r in task_results if str(r.status) != "COMPLETED" or r.error]

            if not completed_tasks:
                execution_status = "FAILED"
            elif failed_tasks:
                execution_status = "PARTIAL_FAILURE"
            else:
                execution_status = "SUCCESS"

            execution_notices = [quota_interrupted_note] if quota_interrupted_note else []
            if failed_tasks:
                execution_notices.append(
                    "Some specialist subtasks failed or were dependency-blocked: "
                    + "; ".join(
                        f"{r.subtask_id} [{r.status}]: {r.error or 'No diagnostic provided'}"
                        for r in failed_tasks
                    )
                )
            partial_execution_notice = "\n".join(execution_notices) or None

            workflow.transition_to(
                WorkflowState.DELIBERATING,
                details={
                    "completed_specialists": len(completed_tasks),
                    "failed_or_blocked_specialists": len(failed_tasks),
                    "deliberation_enabled": policy.deliberation_enabled,
                },
            )
            usable_completed_tasks = [
                result for result in completed_tasks
                if result.output.strip()
                or result.findings
                or result.claims
                or result.assumptions
                or result.unresolved_questions
                or result.evidence_items
            ]
            if not policy.deliberation_enabled:
                deliberation_report = None
            elif not completed_tasks:
                deliberation_report = DeliberationReport(
                    execution_id=execution_id,
                    diagnostics=["Deliberation skipped: no specialist completed successfully."],
                    termination_reason=DeliberationTerminationReason.AGENT_FAILURE,
                )
            elif not usable_completed_tasks:
                deliberation_report = DeliberationReport(
                    execution_id=execution_id,
                    diagnostics=["Deliberation skipped: completed specialists produced no usable output."],
                    termination_reason=DeliberationTerminationReason.NO_RELEVANT_TRIGGER,
                )
            else:
                try:
                    coordinator = DeliberationCoordinator(
                        execution_id=execution_id,
                        task_results=completed_tasks,
                        plan=plan,
                        agent_factory=agent_factory,
                        message_bus=message_bus,
                        shared_context=shared_context,
                        token_tracker=self.tracker,
                    )
                    deliberation_report = await coordinator.run()
                    if deliberation_report.termination_reason == DeliberationTerminationReason.TASK_CANCELLED:
                        raise asyncio.CancelledError("Deliberation was cancelled.")
                except asyncio.CancelledError:
                    raise
                except Exception as deliberation_error:
                    diagnostic = (
                        "Deliberation coordinator failed: "
                        f"{type(deliberation_error).__name__}: {deliberation_error}"
                    )
                    exec_record.record_error("deliberation", diagnostic)
                    deliberation_report = DeliberationReport(
                        execution_id=execution_id,
                        triggered=True,
                        diagnostics=[diagnostic],
                        termination_reason=DeliberationTerminationReason.AGENT_FAILURE,
                    )

            if deliberation_report is not None:
                exec_record.metadata["deliberation_report"] = deliberation_report.model_dump(mode="json")
                if deliberation_report.termination_reason in {
                    DeliberationTerminationReason.AGENT_FAILURE,
                    DeliberationTerminationReason.TOKEN_BUDGET_EXHAUSTED,
                    DeliberationTerminationReason.TIMEOUT,
                    DeliberationTerminationReason.TASK_CANCELLED,
                    DeliberationTerminationReason.MESSAGE_BUDGET_EXHAUSTED,
                } and deliberation_report.triggered:
                    diagnostics = "; ".join(deliberation_report.diagnostics) or str(
                        deliberation_report.termination_reason
                    )
                    execution_notices.append(f"Deliberation incomplete: {diagnostics}")
                    partial_execution_notice = "\n".join(execution_notices)
                    if completed_tasks and execution_status == "SUCCESS":
                        execution_status = "PARTIAL_FAILURE"

            workflow.transition_to(WorkflowState.EXTRACTING_CLAIMS)
            claim_extractor = ClaimExtractor(llm_client=self.client)
            extraction_interruption: Optional[Exception] = None
            temporary_claim_ids: Dict[str, str] = {}

            for result in task_results:
                if str(result.status) != "COMPLETED" or result.error:
                    continue

                if not result.claims and extraction_interruption is not None:
                    diagnostic = (
                        f"Claim extraction skipped [subtask {result.subtask_id}, {result.role}]: "
                        f"previous extraction interruption: {type(extraction_interruption).__name__}: "
                        f"{extraction_interruption}"
                    )
                    claim_extraction_errors.append(diagnostic)
                    exec_record.record_error("claim_extractor", diagnostic)
                    continue

                with self.tracker.operation("claim_extraction", release_reserve_on_exit=False):
                    outcome = await claim_extractor.extract_claims_with_diagnostics(
                        result=result,
                        task_id=execution_id,
                    )
                extracted_claims = outcome.claims
                if outcome.error is not None:
                    diagnostic = (
                        f"Claim extraction failure [subtask {result.subtask_id}, {result.role}]: "
                        f"{type(outcome.error).__name__}: {outcome.error}"
                    )
                    claim_extraction_errors.append(diagnostic)
                    exec_record.record_error(
                        "claim_extractor",
                        diagnostic,
                        details=f"source={outcome.source}",
                    )
                    if isinstance(outcome.error, (LLMQuotaExhaustedError, LLMBudgetExceededError)):
                        extraction_interruption = outcome.error

                registered_claim_ids = await ledger.register_claims_batch(extracted_claims)
                if len(registered_claim_ids) < len(extracted_claims):
                    registered_unique_count = len(set(registered_claim_ids))
                    merged_count = len(registered_claim_ids) - registered_unique_count
                    unregistered_count = len(extracted_claims) - len(registered_claim_ids)
                    diagnostic = (
                        f"Claim ledger capacity overflow [subtask {result.subtask_id}]: "
                        f"discovered={len(extracted_claims)}, registered={registered_unique_count}, "
                        f"merged_duplicates={merged_count}, unregistered={unregistered_count}, "
                        f"capacity={ledger.max_claims}."
                    )
                    claim_extraction_errors.append(diagnostic)
                    exec_record.record_error("claim_ledger", diagnostic)

                ledger_claims = await ledger.get_all_claims()
                claim_ids_by_text = {
                    " ".join(claim.claim_text.split()).casefold(): claim.claim_id
                    for claim in ledger_claims
                }
                for claim_index, claim_text in enumerate(result.claims):
                    canonical_id = claim_ids_by_text.get(" ".join(claim_text.split()).casefold())
                    if canonical_id:
                        temporary_claim_ids[
                            temporary_claim_reference(execution_id, result.subtask_id, claim_index)
                        ] = canonical_id

                # Register specialist evidence items/references as model reasoning/inference provenance.
                claim_ids = list(dict.fromkeys(registered_claim_ids))
                for ev_item in getattr(result, "evidence_items", []):
                    ref_text = ev_item.get("reference") if isinstance(ev_item, dict) else str(ev_item)
                    if ref_text:
                        ev = Evidence(
                            source_type=EvidenceSourceType.REASONING_INFERENCE,
                            source_reference=f"Specialist [{result.role}] analytical derivation",
                            supporting_content=ref_text,
                            originating_agent=result.agent_id,
                            associated_claim_ids=claim_ids,
                        )
                        await ledger.register_evidence(ev)
                        for claim in extracted_claims:
                            if ev.evidence_id not in claim.evidence_references:
                                claim.evidence_references.append(ev.evidence_id)

            if deliberation_report is not None:
                for exchange in deliberation_report.exchanges:
                    if not exchange.introduced_claims:
                        continue
                    response_agent = agent_factory.get_agent(exchange.target_agent_id)
                    response_result_source = next(
                        (item for item in completed_tasks if item.agent_id == exchange.target_agent_id),
                        None,
                    )
                    if response_agent is None or response_result_source is None:
                        diagnostic = (
                            f"Deliberation claim extraction skipped [exchange {exchange.exchange_id}]: "
                            "responding specialist identity was not available."
                        )
                        claim_extraction_errors.append(diagnostic)
                        exec_record.record_error("deliberation_claim_extractor", diagnostic)
                        continue

                    response_subtask_id = (
                        response_agent.assigned_subtask_id or response_result_source.subtask_id
                    )
                    response_task_result = TaskResult(
                        task_id=execution_id,
                        subtask_id=response_subtask_id,
                        agent_id=response_agent.agent_id,
                        role=response_agent.role,
                        status=TaskStatus.COMPLETED,
                        output=exchange.response_content or "",
                        claims=list(exchange.introduced_claims),
                        assumptions=list(exchange.assumptions),
                    )
                    with self.tracker.operation("claim_extraction", release_reserve_on_exit=False):
                        response_outcome = await claim_extractor.extract_claims_with_diagnostics(
                            result=response_task_result,
                            task_id=execution_id,
                        )
                    response_claims = response_outcome.claims
                    if response_outcome.error is not None:
                        diagnostic = (
                            f"Deliberation claim extraction failure [exchange {exchange.exchange_id}, "
                            f"response {exchange.response_message_id}]: "
                            f"{type(response_outcome.error).__name__}: {response_outcome.error}"
                        )
                        claim_extraction_errors.append(diagnostic)
                        exec_record.record_error(
                            "deliberation_claim_extractor",
                            diagnostic,
                            details=f"source={response_outcome.source}",
                        )

                    if len(exchange.introduced_claim_references) == len(response_claims):
                        response_claim_references = exchange.introduced_claim_references
                    else:
                        response_claim_references = [
                            temporary_claim_reference(
                                execution_id,
                                f"{response_subtask_id}:response:{exchange.response_message_id or exchange.exchange_id}",
                                index,
                            )
                            for index in range(len(response_claims))
                        ]
                    parent_claim_id = temporary_claim_ids.get(exchange.claim_reference or "", "")
                    for index, claim in enumerate(response_claims):
                        claim.source_provenance.append({
                            "agent_id": response_agent.agent_id,
                            "role": response_agent.role,
                            "subtask_id": response_subtask_id,
                            "source_kind": "DELIBERATION_RESPONSE",
                            "execution_id": execution_id,
                            "exchange_id": exchange.exchange_id,
                            "response_message_id": exchange.response_message_id or "",
                            "introduced_claim_reference": response_claim_references[index],
                            "parent_claim_reference": exchange.claim_reference or "",
                            "parent_claim_id": parent_claim_id,
                        })
                        claim.unverified_evidence_references = list(exchange.evidence_references)

                    response_registered_ids = await ledger.register_claims_batch(response_claims)
                    ledger_claims = await ledger.get_all_claims()
                    canonical_response_ids = {
                        source.get("introduced_claim_reference", ""): claim.claim_id
                        for claim in ledger_claims
                        for source in claim.source_provenance
                        if source.get("source_kind") == "DELIBERATION_RESPONSE"
                        and source.get("exchange_id") == exchange.exchange_id
                    }
                    exchange.introduced_claim_ids = [
                        canonical_response_ids.get(reference)
                        for reference in response_claim_references
                    ]
                    if len(response_registered_ids) < len(response_claims):
                        registered_unique_count = len(set(response_registered_ids))
                        merged_count = len(response_registered_ids) - registered_unique_count
                        unregistered_count = len(response_claims) - len(response_registered_ids)
                        diagnostic = (
                            f"Deliberation claim ledger capacity overflow [exchange {exchange.exchange_id}]: "
                            f"discovered={len(response_claims)}, registered={registered_unique_count}, "
                            f"merged_duplicates={merged_count}, unregistered={unregistered_count}, "
                            f"capacity={ledger.max_claims}."
                        )
                        claim_extraction_errors.append(diagnostic)
                        exec_record.record_error("claim_ledger", diagnostic)

            if deliberation_report is not None:
                exec_record.metadata["deliberation_report"] = deliberation_report.model_dump(mode="json")

            if claim_extraction_errors:
                execution_notices.append("Claim extraction diagnostics: " + " | ".join(claim_extraction_errors))
                partial_execution_notice = "\n".join(execution_notices)
                if completed_tasks and execution_status == "SUCCESS":
                    execution_status = "PARTIAL_FAILURE"

            self.tracker.release_operation_reserve("claim_extraction")

            if extraction_interruption is not None:
                if isinstance(extraction_interruption, LLMQuotaExhaustedError):
                    quota_interrupted_note = (
                        f"Claim extraction interrupted after provider quota error: {extraction_interruption}"
                    )
                else:
                    quota_interrupted_note = (
                        f"Claim extraction interrupted by local application budget: {extraction_interruption}"
                    )
                execution_notices.insert(0, quota_interrupted_note)
                partial_execution_notice = "\n".join(execution_notices)
                for claim in await ledger.get_all_claims(status=ClaimStatus.PENDING):
                    await ledger.update_claim_status(
                        claim_id=claim.claim_id,
                        status=ClaimStatus.UNREVIEWED,
                        reviewer_feedback=quota_interrupted_note,
                    )
                if completed_tasks:
                    execution_status = "PARTIAL_FAILURE"
            else:
                # Stages 4 and 5: audit and repair, with quota protection.
                try:
                    workflow.transition_to(WorkflowState.AUDITING_CLAIMS)
                    with self.tracker.operation("verification"):
                        reviewer_agent = agent_factory.create_reviewer()
                        reviewer = AdversarialReviewer(reviewer_agent=reviewer_agent, llm_client=self.client)
                        review_feedbacks = await reviewer.review_claims(
                            ledger=ledger,
                            task_context=user_input,
                            deliberation_context=_format_deliberation_context(deliberation_report),
                        )

                        workflow.transition_to(WorkflowState.REPAIRING_CLAIMS)
                        repair_engine = RepairEngine(
                            max_repair_iterations=self.max_repair_iterations,
                            llm_client=self.client,
                        )
                        repair_attempts = await repair_engine.execute_repair_loop(
                            ledger=ledger,
                            reviewer=reviewer,
                            agent_factory=agent_factory,
                            task_context=user_input,
                        )
                except (LLMQuotaExhaustedError, LLMBudgetExceededError) as quota_err:
                    logger.warning(
                        f"Verification pipeline stopped due to quota/budget exhaustion: {quota_err}"
                    )
                    quota_interrupted_note = f"Verification pipeline interrupted: {quota_err}"
                    execution_notices.insert(0, quota_interrupted_note)
                    partial_execution_notice = "\n".join(execution_notices)

                    for claim in await ledger.get_all_claims(status=ClaimStatus.PENDING):
                        await ledger.update_claim_status(
                            claim_id=claim.claim_id,
                            status=ClaimStatus.UNREVIEWED,
                            reviewer_feedback="Audit interrupted before verification due to LLM quota or budget limit.",
                        )

                    if completed_tasks:
                        execution_status = "PARTIAL_FAILURE"

            if claim_extraction_errors and completed_tasks and execution_status == "SUCCESS":
                execution_status = "PARTIAL_FAILURE"

            # -------------------------------------------------------------
            # Stage 6: FINAL SYNTHESIS & REPORT GENERATION
            # -------------------------------------------------------------
            workflow.transition_to(WorkflowState.SYNTHESIZING)
            verification_report = await ledger.generate_report(
                review_decisions=review_feedbacks,
                repair_history=repair_attempts,
            )

            # Collect unresolved questions from results and agents
            all_questions = []
            for r in task_results:
                all_questions.extend(r.unresolved_questions)
            for a in agent_factory.list_agents():
                all_questions.extend(a.unresolved_questions)
            if deliberation_report is not None:
                all_questions.extend(deliberation_report.unresolved_questions)
                all_questions.extend(
                    f"Unresolved deliberation disagreement: {item}"
                    for item in deliberation_report.unresolved_disagreements
                )
            all_questions = list(dict.fromkeys(all_questions))

            # Synthesize final answer truthfully based on execution status
            if execution_status == "FAILED":
                final_answer = (
                    "# Execution Failure Report\n\n"
                    "### Status: FAILED\n\n"
                    "No specialist subtask completed successfully. "
                    "No verified claims or findings could be established.\n\n"
                    "### Subtask Outcomes:\n"
                    + "\n".join(
                        f"- **{r.role}** ({r.subtask_id}) [{r.status}]: "
                        f"{r.error or 'No deliverable was produced.'}"
                        for r in task_results
                    )
                    + "\n\n### Verification Status:\n"
                    f"{verification_report.summary_message or 'Verification was unavailable.'}\n\n"
                    "### Unresolved Questions / Risks:\n"
                    + ("\n".join(f"- {q}" for q in all_questions) if all_questions else "- Workflow aborted before domain verification.")
                )
            else:
                synthesizer = agent_factory.create_synthesizer()
                with self.tracker.operation("synthesis"):
                    final_answer = await synthesizer.synthesize(
                        task_description=user_input,
                        specialist_results=[r.model_dump() for r in task_results],
                        verification_report=verification_report,
                        unresolved_questions=all_questions,
                        partial_execution_notice=partial_execution_notice,
                        deliberation_report=deliberation_report,
                    )

            if execution_status == "FAILED":
                workflow.fail("Execution produced no successful specialist deliverables.")
            elif execution_status == "PARTIAL_FAILURE":
                workflow.transition_to(
                    WorkflowState.PARTIAL_FAILURE,
                    details={
                        "completed": len(completed_tasks),
                        "failed": len(failed_tasks),
                        "quota_interrupted": bool(quota_interrupted_note),
                    },
                )
            else:
                workflow.transition_to(
                    WorkflowState.SUCCESS,
                    details={"completed": len(completed_tasks)},
                )

            total_duration = round(time.perf_counter() - start_perf, 3)

            # Retrieve token usage
            token_summary = await self.tracker.get_summary()
            provider_call_records = self.tracker.get_provider_call_records(execution_id)
            exec_record.metadata["provider_calls"] = [
                call.model_dump(mode="json") for call in provider_call_records
            ]
            exec_record.metadata["provider_call_records_dropped"] = (
                self.tracker.get_provider_call_records_dropped()
            )
            accounting_snapshot = self.tracker.get_active_accounting_snapshot()
            if accounting_snapshot is not None:
                exec_record.metadata["accounting"] = accounting_snapshot.model_dump(mode="json")

            # Finalize execution record and save structured log
            exec_record.finish(status=execution_status, token_usage=token_summary)
            exec_record.save()

            logger.info(
                f"=== MEDHA Execution [{execution_id}] {execution_status} in {total_duration}s ==="
            )

            all_claims_list = [c.model_dump() for c in await ledger.get_all_claims()]
            all_evidence_list = [e.model_dump() for e in await ledger.get_all_evidence()]

            return MedhaResult(
                execution_id=execution_id,
                task=user_input,
                final_answer=final_answer,
                plan_artifact=plan_artifact,
                accounting=accounting_snapshot,
                provider_call_records=provider_call_records,
                status=execution_status,
                duration_seconds=total_duration,
                token_usage=token_summary,
                generated_agents=[a.get_summary() for a in agent_factory.list_agents()],
                task_results=[r.model_dump() for r in task_results],
                deliberation_report=deliberation_report,
                claims=all_claims_list,
                evidence=all_evidence_list,
                review_decisions=[rf.model_dump() for rf in review_feedbacks],
                repair_history=[ra.model_dump() for ra in repair_attempts],
                unresolved_questions=all_questions,
                claim_extraction_errors=claim_extraction_errors,
                verification_report=verification_report.model_dump(),
                workflow_stages=workflow.get_summary(),
                error=quota_interrupted_note,
            )

        except asyncio.CancelledError:
            cancellation_message = "MEDHA execution was cancelled."
            workflow.cancel(cancellation_message)
            exec_record.record_error(component="orchestrator", message=cancellation_message)
            provider_call_records = self.tracker.get_provider_call_records(execution_id)
            exec_record.metadata["provider_calls"] = [
                call.model_dump(mode="json") for call in provider_call_records
            ]
            accounting_snapshot = self.tracker.get_active_accounting_snapshot()
            if accounting_snapshot is not None:
                exec_record.metadata["accounting"] = accounting_snapshot.model_dump(mode="json")
            exec_record.metadata["provider_call_records_dropped"] = (
                self.tracker.get_provider_call_records_dropped()
            )
            if deliberation_report is not None:
                exec_record.metadata["deliberation_report"] = deliberation_report.model_dump(mode="json")
            exec_record.finish(status="CANCELLED", token_usage=await self.tracker.get_summary())
            exec_record.save()
            raise

        except Exception as e:
            total_duration = round(time.perf_counter() - start_perf, 3)
            logger.error(f"MEDHA Execution [{execution_id}] encountered fatal error: {e}", exc_info=True)
            workflow.fail(str(e))
            exec_record.record_error(component="orchestrator", message=str(e))
            provider_call_records = self.tracker.get_provider_call_records(execution_id)
            exec_record.metadata["provider_calls"] = [
                call.model_dump(mode="json") for call in provider_call_records
            ]
            accounting_snapshot = self.tracker.get_active_accounting_snapshot()
            if accounting_snapshot is not None:
                exec_record.metadata["accounting"] = accounting_snapshot.model_dump(mode="json")
            exec_record.metadata["provider_call_records_dropped"] = (
                self.tracker.get_provider_call_records_dropped()
            )
            exec_record.finish(status="FAILED")
            exec_record.save()

            token_summary = await self.tracker.get_summary()
            all_claims_list = [c.model_dump() for c in await ledger.get_all_claims()]
            all_evidence_list = [e.model_dump() for e in await ledger.get_all_evidence()]
            v_report = (await ledger.generate_report()).model_dump()

            return MedhaResult(
                execution_id=execution_id,
                task=user_input,
                final_answer=f"# Execution Error\n\nMEDHA encountered an error during task processing: {e}",
                plan_artifact=plan_artifact,
                accounting=accounting_snapshot,
                provider_call_records=provider_call_records,
                status="FAILED",
                duration_seconds=total_duration,
                token_usage=token_summary,
                task_results=[r.model_dump() for r in task_results],
                deliberation_report=deliberation_report,
                claim_extraction_errors=claim_extraction_errors,
                claims=all_claims_list,
                evidence=all_evidence_list,
                verification_report=v_report,
                workflow_stages=workflow.get_summary(),
                error=str(e),
            )
