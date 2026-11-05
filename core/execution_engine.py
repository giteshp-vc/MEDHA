import asyncio
import math
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

from communication.message import Message
from communication.message_bus import MessageBus
from communication.shared_context import SharedContext
from config.settings import EXECUTION_TIMEOUT_SECONDS, MAX_CONCURRENT_AGENTS
from core.agent_factory import AgentFactory
from core.task_manager import TaskManager
from models.schemas import AgentStatus, MessageType, TaskStatus
from models.execution_policy import ExecutionPolicy
from models.task_models import SubTask, TaskPlan, TaskResult
from utils.logger import get_logger
from utils.retry import (
    LLMBudgetExceededError,
    LLMError,
    LLMQuotaExhaustedError,
    LLMStructuredOutputError,
)
from utils.token_tracker import TokenTracker

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
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number.")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1.")
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
        execution_policy: Optional[ExecutionPolicy] = None,
    ) -> List[TaskResult]:
        """Execute all planned subtasks respecting dependencies, running parallel stages concurrently."""
        policy = execution_policy or ExecutionPolicy()
        task_manager = TaskManager(
            plan=plan,
            task_id=task_id,
            honor_dependencies=policy.honor_dependencies,
        )
        start_time = time.perf_counter()
        deadline = start_time + self.timeout_seconds

        logger.info(
            f"ExecutionEngine starting plan with {len(plan.subtasks)} subtasks. "
            f"Max concurrency: {self.max_concurrency}, Timeout: {self.timeout_seconds}s"
        )

        while not task_manager.is_all_finished():
            remaining_seconds = deadline - time.perf_counter()
            if remaining_seconds <= 0:
                self._mark_deadline_expired_pending(task_manager, agent_factory)
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
            tasks = {
                asyncio.create_task(
                    self._execute_single_subtask_with_timestamp(
                        subtask=subtask,
                        task_manager=task_manager,
                        agent_factory=agent_factory,
                        message_bus=message_bus,
                        shared_context=shared_context,
                        task_id=task_id,
                        execution_policy=policy,
                    ),
                    name=f"medha-subtask-{subtask.id}",
                ): subtask
                for subtask in ready_subtasks
            }

            try:
                done, pending = await asyncio.wait(
                    tasks,
                    timeout=max(0.0, remaining_seconds),
                    return_when=asyncio.ALL_COMPLETED,
                )
            except asyncio.CancelledError:
                self._record_finished_tasks(
                    done_tasks={task for task in tasks if task.done()},
                    task_map=tasks,
                    task_manager=task_manager,
                    deadline=deadline,
                )
                unfinished = {task for task in tasks if not task.done()}
                for task in unfinished:
                    task.cancel()
                await self._cancel_and_wait(unfinished)
                self._record_terminal_tasks(
                    unfinished,
                    tasks,
                    task_manager,
                    agent_factory,
                    TaskStatus.CANCELLED,
                    "Parent MEDHA execution was cancelled.",
                )
                raise

            timed_out = set()
            if pending:
                timed_out.update(pending)
                for task in pending:
                    task.cancel()
                cancellation_received = await self._cancel_and_wait(pending)
                if cancellation_received:
                    self._record_terminal_tasks(
                        pending,
                        tasks,
                        task_manager,
                        agent_factory,
                        TaskStatus.CANCELLED,
                        "Parent MEDHA execution was cancelled during timeout cleanup.",
                    )
                    raise asyncio.CancelledError

            timed_out.update(
                task
                for task in done
                if not task.cancelled() and self._completion_time(task) > deadline
            )
            self._record_finished_tasks(
                done_tasks=done - timed_out,
                task_map=tasks,
                task_manager=task_manager,
                deadline=deadline,
            )

            for task in timed_out:
                self._record_terminal_tasks(
                    {task},
                    tasks,
                    task_manager,
                    agent_factory,
                    TaskStatus.TIMED_OUT,
                    f"Execution deadline of {self.timeout_seconds}s expired; worker cancellation was awaited.",
                )

            if timed_out or deadline - time.perf_counter() <= 0:
                self._mark_deadline_expired_pending(task_manager, agent_factory)
                break

        total_duration = round(time.perf_counter() - start_time, 3)
        summary = task_manager.get_summary()
        logger.info(
            f"ExecutionEngine finished plan in {total_duration}s. "
            f"Completed: {summary['completed']}, Failed: {summary['failed']}, Blocked: {summary['blocked']}"
        )

        return list(task_manager.results.values())

    async def _execute_single_subtask_with_timestamp(self, **kwargs):
        result = await self._execute_single_subtask(**kwargs)
        return result, time.perf_counter()

    @staticmethod
    async def _cancel_and_wait(tasks) -> bool:
        """Cancel child tasks and await cleanup, deferring any additional parent cancellation."""
        unfinished = [task for task in tasks if not task.done()]
        for task in unfinished:
            task.cancel()
        if not unfinished:
            return False

        cleanup = asyncio.gather(*unfinished, return_exceptions=True)
        cancellation_received = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancellation_received = True
        return cancellation_received

    @staticmethod
    def _completion_time(task: asyncio.Task) -> float:
        try:
            _, completed_at = task.result()
            return completed_at
        except BaseException:
            return float("inf")

    def _record_finished_tasks(self, done_tasks, task_map, task_manager, deadline) -> None:
        for task in done_tasks:
            subtask = task_map[task]
            try:
                result, completed_at = task.result()
            except asyncio.CancelledError:
                self._record_result_status(
                    task_manager,
                    subtask,
                    TaskStatus.CANCELLED,
                    "Worker task was cancelled before producing a result.",
                )
            except Exception as error:
                logger.error("Worker for subtask [%s] raised an unhandled exception: %s", subtask.id, error)
                task_manager.record_result(
                    TaskResult(
                        subtask_id=subtask.id,
                        agent_id="unknown",
                        role=subtask.role,
                        status=TaskStatus.FAILED,
                        output="",
                        error=_format_subtask_error(error, subtask.id),
                    )
                )
            else:
                if completed_at > deadline:
                    self._record_result_status(
                        task_manager,
                        subtask,
                        TaskStatus.TIMED_OUT,
                        f"Execution deadline of {self.timeout_seconds}s expired before the worker completed.",
                    )
                else:
                    task_manager.record_result(result)

    def _record_terminal_tasks(
        self,
        selected_tasks,
        task_map,
        task_manager,
        agent_factory,
        status: TaskStatus,
        reason: str,
    ) -> None:
        for task in selected_tasks:
            subtask = task_map[task]
            agents = agent_factory.list_agents()
            agent = next(
                (
                    item for item in agents
                    if getattr(item, "assigned_subtask_id", None) == subtask.id
                ),
                None,
            ) if isinstance(agents, list) else None
            if agent is not None:
                agent.update_status(
                    AgentStatus.TIMED_OUT if status == TaskStatus.TIMED_OUT else AgentStatus.CANCELLED
                )
            self._record_result_status(task_manager, subtask, status, reason, agent)

    @staticmethod
    def _record_result_status(task_manager, subtask, status, reason, agent=None) -> None:
        diagnostic = (
            f"Specialist execution timed out [subtask {subtask.id}]: {reason}"
            if status == TaskStatus.TIMED_OUT
            else f"Specialist execution cancelled [subtask {subtask.id}]: {reason}"
        )
        task_manager.record_result(
            TaskResult(
                subtask_id=subtask.id,
                agent_id=getattr(agent, "agent_id", "scheduler"),
                role=subtask.role,
                status=status,
                output="",
                error=diagnostic,
            )
        )

    def _mark_deadline_expired_pending(self, task_manager, agent_factory) -> None:
        task_manager.get_ready_subtasks()
        for subtask in task_manager.subtasks.values():
            if subtask.status == TaskStatus.PENDING:
                self._record_result_status(
                    task_manager,
                    subtask,
                    TaskStatus.TIMED_OUT,
                    f"Execution deadline of {self.timeout_seconds}s expired before the specialist could start.",
                )

    async def _execute_single_subtask(
        self,
        subtask: SubTask,
        task_manager: TaskManager,
        agent_factory: AgentFactory,
        message_bus: MessageBus,
        shared_context: SharedContext,
        task_id: str,
        execution_policy: ExecutionPolicy,
    ) -> TaskResult:
        """Run a single specialist agent within concurrency limits and error isolation."""
        async with self.semaphore:
            start_time = time.perf_counter()
            agent = agent_factory.create_from_subtask(subtask, task_id=task_id)
            if execution_policy.route_messages:
                message_bus.register_agent(
                    agent.agent_id,
                    execution_id=task_id,
                    subtask_id=subtask.id,
                )

            try:
                # Gather upstream findings from dependency subtasks
                upstream_results = (
                    task_manager.get_upstream_results(subtask.id)
                    if execution_policy.inject_upstream_results
                    else []
                )

                # Fetch any messages specifically addressed to this agent
                pending_messages = (
                    await message_bus.drain_inbox(agent.agent_id)
                    if execution_policy.route_messages
                    else []
                )
                incoming_notes = "\n".join(f"[{m.sender_id}]: {m.content}" for m in pending_messages)

                context: Dict[str, Any] = {
                    "upstream_results": upstream_results,
                    "shared_notes": incoming_notes,
                    "task_description": shared_context.task_description,
                }

                # Execute specialist agent
                client = getattr(agent, "client", None)
                tracker = getattr(client, "token_tracker", None)
                call_context = (
                    tracker.call_context(
                        execution_id=task_id,
                        agent_id=agent.agent_id,
                        agent_role=agent.role,
                    )
                    if isinstance(tracker, TokenTracker)
                    else nullcontext()
                )
                with call_context:
                    response = await agent.execute(context=context)

                if execution_policy.route_messages:
                    # Broadcast findings to collaborators via message bus
                    dependent_subtask_ids = [
                        candidate.id
                        for candidate in task_manager.plan.subtasks
                        if subtask.id in candidate.dependencies
                    ]
                    for finding in response.findings:
                        finding_message = Message(
                            sender_id=agent.agent_id,
                            recipient_id="ALL",
                            task_id=task_id,
                            execution_id=task_id,
                            subtask_id=subtask.id,
                            message_type=MessageType.FINDING,
                            content=finding,
                            metadata={"target_subtask_ids": dependent_subtask_ids},
                            )
                        await message_bus.publish(finding_message)
                        await shared_context.add_communication(finding_message.model_dump(mode="json"))
                        await shared_context.add_finding(agent.agent_id, agent.role, finding)

                    # Share evidence references if any
                    if response.evidence_references:
                        evidence_message = Message(
                            sender_id=agent.agent_id,
                            recipient_id="ALL",
                            task_id=task_id,
                            execution_id=task_id,
                            subtask_id=subtask.id,
                            message_type=MessageType.EVIDENCE,
                            content=f"Evidence provided by {agent.role} for subtask {subtask.id}",
                            evidence_references=response.evidence_references,
                            metadata={"target_subtask_ids": dependent_subtask_ids},
                            )
                        await message_bus.publish(evidence_message)
                        await shared_context.add_communication(evidence_message.model_dump(mode="json"))

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

