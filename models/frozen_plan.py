import hashlib
import json
from collections import deque

from pydantic import ConfigDict

from models.provider_call import AccountingSnapshot
from models.schemas import BaseSchema
from models.task_models import TaskPlan


class FrozenTaskPlan(BaseSchema):
    """Immutable task-bound serialized TaskPlan reusable across treatment runs."""

    model_config = ConfigDict(frozen=True)

    source_task: str
    serialized_plan: str
    plan_hash: str
    preparation_accounting: AccountingSnapshot | None = None

    @classmethod
    def from_plan(
        cls,
        source_task: str,
        plan: TaskPlan,
        preparation_accounting: AccountingSnapshot | None = None,
    ) -> "FrozenTaskPlan":
        validate_task_plan(plan, source_task)
        serialized_plan = _serialize_plan(plan)
        return cls(
            source_task=source_task,
            serialized_plan=serialized_plan,
            plan_hash=hashlib.sha256(serialized_plan.encode("utf-8")).hexdigest(),
            preparation_accounting=preparation_accounting,
        )

    def load_plan(self, expected_task: str) -> TaskPlan:
        """Verify task binding and artifact integrity, then return a fresh mutable plan copy."""
        if expected_task != self.source_task:
            raise ValueError("Frozen task plan belongs to a different source task.")
        actual_hash = hashlib.sha256(self.serialized_plan.encode("utf-8")).hexdigest()
        if actual_hash != self.plan_hash:
            raise ValueError("Frozen task plan hash does not match its serialized contents.")

        plan = TaskPlan.model_validate_json(self.serialized_plan)
        validate_task_plan(plan, self.source_task)
        if _serialize_plan(plan) != self.serialized_plan:
            raise ValueError("Frozen task plan is not in canonical serialized form.")
        return plan


def validate_task_plan(plan: TaskPlan, source_task: str) -> None:
    """Validate task identity, required subtask fields, dependency references, and DAG structure."""
    if not source_task or not source_task.strip():
        raise ValueError("Source task description cannot be empty.")
    if not plan.task or not plan.task.strip():
        raise ValueError("Task plan identity cannot be empty.")
    if not plan.subtasks:
        raise ValueError("Task plan must contain at least one subtask.")

    subtask_ids = []
    for subtask in plan.subtasks:
        if not subtask.id or not subtask.id.strip():
            raise ValueError("Every subtask must have a non-empty ID.")
        if not subtask.role or not subtask.role.strip():
            raise ValueError(f"Subtask [{subtask.id}] must have a non-empty role.")
        if not subtask.objective or not subtask.objective.strip():
            raise ValueError(f"Subtask [{subtask.id}] must have a non-empty objective.")
        if len(subtask.dependencies) != len(set(subtask.dependencies)):
            raise ValueError(f"Subtask [{subtask.id}] contains duplicate dependency IDs.")
        subtask_ids.append(subtask.id)

    if len(subtask_ids) != len(set(subtask_ids)):
        raise ValueError("Task plan contains duplicate subtask IDs.")

    subtask_id_set = set(subtask_ids)
    dependents = {subtask_id: [] for subtask_id in subtask_ids}
    in_degrees = {}
    for subtask in plan.subtasks:
        unknown = set(subtask.dependencies) - subtask_id_set
        if unknown:
            raise ValueError(
                f"Subtask [{subtask.id}] references unknown dependency IDs: {', '.join(sorted(unknown))}."
            )
        in_degrees[subtask.id] = len(subtask.dependencies)
        for dependency in subtask.dependencies:
            dependents[dependency].append(subtask.id)

    ready = deque(subtask_id for subtask_id, count in in_degrees.items() if count == 0)
    visited = 0
    while ready:
        subtask_id = ready.popleft()
        visited += 1
        for dependent in dependents[subtask_id]:
            in_degrees[dependent] -= 1
            if in_degrees[dependent] == 0:
                ready.append(dependent)
    if visited != len(subtask_ids):
        raise ValueError("Task plan dependency graph contains a cycle.")


def _serialize_plan(plan: TaskPlan) -> str:
    return json.dumps(
        plan.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )