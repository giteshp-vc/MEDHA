import asyncio
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from communication.message_bus import MessageBus
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


class MedhaResult(BaseSchema):
    """Complete, transparent outcome returned by the MEDHA multi-agent system."""

    execution_id: str
    task: str
    final_answer: str
    status: str = "SUCCESS"  # Truthful status: SUCCESS, PARTIAL_FAILURE, FAILED
    duration_seconds: float = 0.0
    token_usage: Dict[str, Any] = Field(default_factory=dict)
    generated_agents: List[Dict[str, Any]] = Field(default_factory=list)
    task_results: List[Dict[str, Any]] = Field(default_factory=list)
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
        async with self.tracker.task_budget():
            return await self._run_task(user_input)

    async def _run_task(self, user_input: str) -> MedhaResult:
        """Execute the end-to-end evidence-driven multi-agent lifecycle for a given user task."""
        execution_id = generate_id("exec")
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
        agent_factory = AgentFactory(llm_client=self.client)

        try:
            # -------------------------------------------------------------
            # Stage 1: DYNAMIC TASK PLANNING & SPECIALIST IDENTIFICATION
            # -------------------------------------------------------------
            workflow.transition_to(WorkflowState.PLANNING)
            planner = TaskPlanner(llm_client=self.client)
            with self.tracker.operation("planning"):
                plan: TaskPlan = await planner.create_plan(
                    task_description=user_input,
                    max_subtasks=self.max_concurrency,
                )

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

            workflow.transition_to(WorkflowState.EXTRACTING_CLAIMS)
            claim_extractor = ClaimExtractor(llm_client=self.client)
            extraction_interruption: Optional[Exception] = None

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
                status=execution_status,
                duration_seconds=total_duration,
                token_usage=token_summary,
                generated_agents=[a.get_summary() for a in agent_factory.list_agents()],
                task_results=[r.model_dump() for r in task_results],
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

        except Exception as e:
            total_duration = round(time.perf_counter() - start_perf, 3)
            logger.error(f"MEDHA Execution [{execution_id}] encountered fatal error: {e}", exc_info=True)
            workflow.fail(str(e))
            exec_record.record_error(component="orchestrator", message=str(e))
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
                status="FAILED",
                duration_seconds=total_duration,
                token_usage=token_summary,
                task_results=[r.model_dump() for r in task_results],
                claim_extraction_errors=claim_extraction_errors,
                claims=all_claims_list,
                evidence=all_evidence_list,
                verification_report=v_report,
                workflow_stages=workflow.get_summary(),
                error=str(e),
            )
