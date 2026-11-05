from typing import Any, Dict, List, Optional, Set

from models.schemas import TaskStatus
from models.task_models import SubTask, TaskPlan, TaskResult
from utils.logger import get_logger

logger = get_logger("task_manager")


class TaskManager:
    """Manages subtask lifecycle, dependency resolution, execution status, and collected results."""

    def __init__(
        self,
        plan: TaskPlan,
        task_id: str = "",
        honor_dependencies: bool = True,
    ):
        self.task_id = task_id
        self.plan = plan
        self.honor_dependencies = honor_dependencies
        self.subtasks: Dict[str, SubTask] = {s.id: s.model_copy() for s in plan.subtasks}
        self.results: Dict[str, TaskResult] = {}
        self.completed_subtask_ids: Set[str] = set()
        self.failed_subtask_ids: Set[str] = set()

    def get_ready_subtasks(self) -> List[SubTask]:
        """Return all subtasks whose dependencies have been successfully completed and are pending execution."""
        if not self.honor_dependencies:
            return [
                subtask for subtask in self.subtasks.values()
                if subtask.status == TaskStatus.PENDING
            ]

        changed = True
        while changed:
            changed = False
            for subtask_id, subtask in self.subtasks.items():
                if subtask.status != TaskStatus.PENDING:
                    continue

                missing = [dep for dep in subtask.dependencies if dep not in self.subtasks]
                unavailable = [
                    dep for dep in subtask.dependencies
                    if dep in self.subtasks
                    and self.subtasks[dep].status in (
                        TaskStatus.FAILED,
                        TaskStatus.BLOCKED,
                        TaskStatus.TIMED_OUT,
                        TaskStatus.CANCELLED,
                    )
                ]
                if missing or unavailable:
                    reasons = []
                    if missing:
                        reasons.append(f"unknown dependency ID(s): {', '.join(missing)}")
                    if unavailable:
                        reasons.append(
                            f"failed, blocked, timed-out, or cancelled dependency ID(s): "
                            f"{', '.join(unavailable)}"
                        )
                    self._mark_blocked(subtask, "; ".join(reasons))
                    changed = True

        return [
            subtask for subtask in self.subtasks.values()
            if subtask.status == TaskStatus.PENDING
            and set(subtask.dependencies).issubset(self.completed_subtask_ids)
        ]

    def _mark_blocked(self, subtask: SubTask, reason: str) -> TaskResult:
        subtask.status = TaskStatus.BLOCKED
        result = TaskResult(
            subtask_id=subtask.id,
            agent_id="scheduler",
            role=subtask.role,
            status=TaskStatus.BLOCKED,
            error=f"Dependency-blocked task [subtask {subtask.id}]: {reason}",
        )
        self.results[subtask.id] = result
        logger.warning(result.error)
        return result

    def block_unrunnable_subtasks(self) -> List[TaskResult]:
        """Mark remaining pending tasks blocked when the dependency graph has no runnable node."""
        blocked = []
        for subtask in self.subtasks.values():
            if subtask.status != TaskStatus.PENDING:
                continue
            unresolved = [
                f"{dep} ({self.subtasks[dep].status if dep in self.subtasks else 'UNKNOWN'})"
                for dep in subtask.dependencies
                if dep not in self.completed_subtask_ids
            ]
            reason = "no runnable task remains; unresolved dependencies: " + ", ".join(unresolved)
            reason += " (dependency cycle or missing dependency)"
            blocked.append(self._mark_blocked(subtask, reason))
        return blocked

    def fail_pending_subtasks(self, reason: str) -> List[TaskResult]:
        """Record pending tasks that could not be scheduled because execution was interrupted."""
        failed = []
        for subtask in self.subtasks.values():
            if subtask.status != TaskStatus.PENDING:
                continue
            result = TaskResult(
                subtask_id=subtask.id,
                agent_id="scheduler",
                role=subtask.role,
                status=TaskStatus.FAILED,
                error=f"Execution interruption [subtask {subtask.id}]: {reason}",
            )
            subtask.status = TaskStatus.FAILED
            self.results[subtask.id] = result
            self.failed_subtask_ids.add(subtask.id)
            failed.append(result)
        return failed

    def mark_in_progress(self, subtask_id: str) -> None:
        """Mark a subtask as currently executing."""
        if subtask_id in self.subtasks:
            self.subtasks[subtask_id].status = TaskStatus.IN_PROGRESS

    def record_result(self, result: TaskResult) -> None:
        """Store the execution result of a subtask and update lifecycle state."""
        subtask_id = result.subtask_id
        self.results[subtask_id] = result

        if subtask_id in self.subtasks:
            self.subtasks[subtask_id].status = result.status

        if result.status == TaskStatus.COMPLETED:
            self.completed_subtask_ids.add(subtask_id)
            logger.info(f"Subtask [{subtask_id}] marked COMPLETED.")
        elif result.status in (TaskStatus.FAILED, TaskStatus.TIMED_OUT, TaskStatus.CANCELLED):
            self.failed_subtask_ids.add(subtask_id)
            logger.error(f"Subtask [{subtask_id}] marked {result.status} ({result.error}).")
        elif result.status == TaskStatus.BLOCKED:
            logger.warning(f"Subtask [{subtask_id}] marked BLOCKED ({result.error}).")

    def get_upstream_results(self, subtask_id: str) -> List[Dict[str, Any]]:
        """Get the outputs of all dependencies that completed before this subtask."""
        if subtask_id not in self.subtasks:
            return []
        subtask = self.subtasks[subtask_id]
        upstream = []
        for dep_id in subtask.dependencies:
            if dep_id in self.results:
                r = self.results[dep_id]
                upstream.append({
                    "subtask_id": dep_id,
                    "role": r.role,
                    "output": r.output,
                    "findings": r.findings,
                    "claims": r.claims,
                })
        return upstream

    def is_all_finished(self) -> bool:
        """Check if every subtask reached a completed, failed, blocked, timed-out, or cancelled state."""
        terminal_states = {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.TIMED_OUT,
            TaskStatus.CANCELLED,
        }
        return all(s.status in terminal_states for s in self.subtasks.values())

    def get_summary(self) -> Dict[str, Any]:
        """Return execution summary statistics."""
        return {
            "total_subtasks": len(self.subtasks),
            "completed": len(self.completed_subtask_ids),
            "failed": len(self.failed_subtask_ids),
            "blocked": sum(1 for s in self.subtasks.values() if s.status == TaskStatus.BLOCKED),
            "all_finished": self.is_all_finished(),
        }

