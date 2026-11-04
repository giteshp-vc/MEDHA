import asyncio
import time
from typing import Any, Dict, List, Optional

from communication.message import Message
from communication.message_bus import MessageBus
from communication.shared_context import SharedContext
from config.settings import EXECUTION_TIMEOUT_SECONDS, MAX_CONCURRENT_AGENTS
from core.agent_factory import AgentFactory
from core.task_manager import TaskManager
from models.schemas import MessageType, TaskStatus
from models.task_models import SubTask, TaskPlan, TaskResult
from utils.logger import get_logger
from utils.retry import (
    LLMBudgetExceededError,
    LLMError,
    LLMQuotaExhaustedError,
    LLMStructuredOutputError,
)

logger = get_logger("execution_engine")


def _describe_exception(error: Exception) -> str:
    description = f"{type(error).__name__}: {error}"
    original = error.__cause__
    if original is not None:
        description += f" (original {type(original).__name__}: {original})"
    return description


def _format_subtask_error(error: Exception, subtask_id: str) -> str:
    if isinstance(error, (LLMBudgetExceededError, LLMQuotaExhaustedError)):
        category = "Budget/quota interruption"
    elif isinstance(error, LLMStructuredOutputError):
        category = "Structured output validation failure"
    elif error.__class__.__name__ == "SpecialistOutputRecoveryError":
        category = "Structured output recovery failure"
    elif isinstance(error, LLMError):
        category = "Provider/API failure"
    else:
        category = "Specialist execution failure"
    return f"{category} [subtask {subtask_id}]: {_describe_exception(error)}"


class ExecutionEngine:
    """Dependency-aware parallel execution engine with concurrency limits and fault tolerance."""

    def __init__(
        self,
        max_concurrency: int = MAX_CONCURRENT_AGENTS,
        timeout_seconds: float = EXECUTION_TIMEOUT_SECONDS,
    ):
        self.max_concurrency = max_concurrency
        self.timeout_seconds = timeout_seconds
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def execute_plan(
        self,
        plan: TaskPlan,
        agent_factory: AgentFactory,
        message_bus: MessageBus,
        shared_context: SharedContext,
        task_id: str = "",
    ) -> List[TaskResult]:
        """Execute all planned subtasks respecting dependencies, running parallel stages concurrently."""
        task_manager = TaskManager(plan=plan, task_id=task_id)
        start_time = time.perf_counter()

        logger.info(
            f"ExecutionEngine starting plan with {len(plan.subtasks)} subtasks. "
            f"Max concurrency: {self.max_concurrency}, Timeout: {self.timeout_seconds}s"
        )

        while not task_manager.is_all_finished():
            # Check for global timeout
            elapsed = time.perf_counter() - start_time
            if elapsed > self.timeout_seconds:
                logger.error(f"ExecutionEngine exceeded global timeout of {self.timeout_seconds}s. Halting.")
                task_manager.fail_pending_subtasks(
                    f"global execution timeout exceeded ({self.timeout_seconds}s)"
                )
                break

            ready_subtasks = task_manager.get_ready_subtasks()
            if not ready_subtasks:
                if not task_manager.is_all_finished():
                    blocked = task_manager.block_unrunnable_subtasks()
                    logger.warning(
                        "No runnable subtasks remain; recorded %d dependency-blocked task(s).",
                        len(blocked),
                    )
                    continue
                break

            logger.info(
                f"Executing parallel batch of {len(ready_subtasks)} ready subtask(s): "
                f"{[s.id for s in ready_subtasks]}"
            )

            # Mark all ready tasks as in progress
            for subtask in ready_subtasks:
                task_manager.mark_in_progress(subtask.id)

            # Execute batch concurrently under semaphore control
            coros = [
                self._execute_single_subtask(
                    subtask=subtask,
                    task_manager=task_manager,
                    agent_factory=agent_factory,
                    message_bus=message_bus,
                    shared_context=shared_context,
                    task_id=task_id,
                )
                for subtask in ready_subtasks
            ]

            results = await asyncio.gather(*coros, return_exceptions=True)

            # Record results into TaskManager
            for subtask, outcome in zip(ready_subtasks, results):
                if isinstance(outcome, Exception):
                    logger.error(f"Worker for subtask [{subtask.id}] threw an unhandled exception: {outcome}")
                    failed_res = TaskResult(
                        subtask_id=subtask.id,
                        agent_id="unknown",
                        role=subtask.role,
                        status=TaskStatus.FAILED,
                        output="",
                        error=_format_subtask_error(outcome, subtask.id),
                    )
                    task_manager.record_result(failed_res)
                elif isinstance(outcome, TaskResult):
                    task_manager.record_result(outcome)

        total_duration = round(time.perf_counter() - start_time, 3)
        summary = task_manager.get_summary()
        logger.info(
            f"ExecutionEngine finished plan in {total_duration}s. "
            f"Completed: {summary['completed']}, Failed: {summary['failed']}, Blocked: {summary['blocked']}"
        )

        return list(task_manager.results.values())

    async def _execute_single_subtask(
        self,
        subtask: SubTask,
        task_manager: TaskManager,
        agent_factory: AgentFactory,
        message_bus: MessageBus,
        shared_context: SharedContext,
        task_id: str,
    ) -> TaskResult:
        """Run a single specialist agent within concurrency limits and error isolation."""
        async with self.semaphore:
            start_time = time.perf_counter()
            agent = agent_factory.create_from_subtask(subtask, task_id=task_id)
            message_bus.register_agent(agent.agent_id)

            try:
                # Gather upstream findings from dependency subtasks
                upstream_results = task_manager.get_upstream_results(subtask.id)

                # Fetch any messages specifically addressed to this agent
                pending_messages = await message_bus.drain_inbox(agent.agent_id)
                incoming_notes = "\n".join(f"[{m.sender_id}]: {m.content}" for m in pending_messages)

                context: Dict[str, Any] = {
                    "upstream_results": upstream_results,
                    "shared_notes": incoming_notes,
                    "task_description": shared_context.task_description,
                }

                # Execute specialist agent
                response = await agent.execute(context=context)

                # Broadcast findings to collaborators via message bus
                for finding in response.findings:
                    await message_bus.publish(
                        Message(
                            sender_id=agent.agent_id,
                            recipient_id="ALL",
                            task_id=task_id,
                            subtask_id=subtask.id,
                            message_type=MessageType.FINDING,
                            content=finding,
                        )
                    )
                    await shared_context.add_finding(agent.agent_id, agent.role, finding)

                # Share evidence references if any
                if response.evidence_references:
                    await message_bus.share_evidence(
                        sender_id=agent.agent_id,
                        content=f"Evidence provided by {agent.role} for subtask {subtask.id}",
                        evidence_references=response.evidence_references,
                        task_id=task_id,
                    )

                duration = round(time.perf_counter() - start_time, 3)

                return TaskResult(
                    subtask_id=subtask.id,
                    agent_id=agent.agent_id,
                    role=agent.role,
                    status=TaskStatus.COMPLETED,
                    output=response.content,
                    findings=response.findings,
                    claims=response.claims,
                    evidence_items=[{"reference": ref, "source": "agent_reasoning"} for ref in response.evidence_references],
                    assumptions=response.assumptions,
                    unresolved_questions=response.unresolved_questions,
                    duration_seconds=duration,
                )

            except Exception as e:
                duration = round(time.perf_counter() - start_time, 3)
                logger.error(f"Subtask [{subtask.id}] agent [{agent.role}] failed: {e}")
                return TaskResult(
                    subtask_id=subtask.id,
                    agent_id=agent.agent_id,
                    role=agent.role,
                    status=TaskStatus.FAILED,
                    output="",
                    error=_format_subtask_error(e, subtask.id),
                    duration_seconds=duration,
                )

