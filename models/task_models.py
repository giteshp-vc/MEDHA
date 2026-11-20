from typing import Any, Dict, List, Optional
from pydantic import Field

from models.schemas import BaseSchema, TaskStatus
from utils.helpers import generate_id, utcnow_iso


class SubTask(BaseSchema):
    """Specification of a subtask generated dynamically by the planner."""

    id: str = Field(description="Unique subtask identifier, e.g. T1, T2")
    role: str = Field(description="Dynamically determined specialist role required for this subtask")
    objective: str = Field(description="Clear goal and expected deliverable of the subtask")
    instructions: str = Field(default="", description="Specific guidance or focus areas for the agent")
    dependencies: List[str] = Field(default_factory=list, description="IDs of subtasks that must complete first")
    expected_output: str = Field(default="", description="Description of expected output format and depth")
    status: TaskStatus = Field(default=TaskStatus.PENDING)


class TaskPlan(BaseSchema):
    """The structured execution plan produced by the Task Planner."""

    task: str = Field(description="The synthesized understanding of the primary task objective")
    subtasks: List[SubTask] = Field(description="List of dynamically planned subtasks")
    rationale: Optional[str] = Field(default="", description="Planner's reasoning for role selection and decomposition")
    completion_criteria: List[str] = Field(default_factory=list, description="Criteria defining overall task completion")


class TaskResult(BaseSchema):
    """The execution outcome of a single subtask executed by an agent."""

    task_id: str = Field(default_factory=lambda: generate_id("res"))
    subtask_id: str
    agent_id: str
    role: str
    status: TaskStatus = TaskStatus.COMPLETED
    output: str = ""
    findings: List[str] = Field(default_factory=list)
    claims: List[str] = Field(default_factory=list)
    evidence_items: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_diagnostics: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    error: Optional[str] = None


class Task(BaseSchema):
    """Top-level task lifecycle entity."""

    task_id: str = Field(default_factory=lambda: generate_id("task"))
    description: str
    plan: Optional[TaskPlan] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = Field(default_factory=utcnow_iso)
    completed_at: Optional[str] = None

