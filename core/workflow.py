from enum import Enum
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from utils.helpers import utcnow_iso
from utils.logger import get_logger

logger = get_logger("workflow")


class WorkflowState(str, Enum):
    """Lifecycle stages of the MEDHA end-to-end multi-agent execution."""

    INITIALIZED = "INITIALIZED"
    PLANNING = "PLANNING"
    EXECUTING_SUBTASKS = "EXECUTING_SUBTASKS"
    EXTRACTING_CLAIMS = "EXTRACTING_CLAIMS"
    AUDITING_CLAIMS = "AUDITING_CLAIMS"
    REPAIRING_CLAIMS = "REPAIRING_CLAIMS"
    SYNTHESIZING = "SYNTHESIZING"
    COMPLETED = "COMPLETED"
    SUCCESS = "SUCCESS"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    FAILED = "FAILED"


class StageRecord(BaseModel):
    """Record of an individual workflow stage transition."""

    stage: WorkflowState
    start_time: str = Field(default_factory=utcnow_iso)
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    status: str = "IN_PROGRESS"
    details: Dict[str, Any] = Field(default_factory=dict)


class WorkflowTracker:
    """Tracks and validates transitions through the MEDHA lifecycle stages."""

    def __init__(self, execution_id: str):
        self.execution_id = execution_id
        self.current_state = WorkflowState.INITIALIZED
        self.stage_history: List[StageRecord] = []
        self._current_stage_record: Optional[StageRecord] = None
        self._stage_start_perf: float = time.perf_counter()

    def transition_to(self, new_state: WorkflowState, details: Optional[Dict[str, Any]] = None) -> None:
        """Mark completion of previous stage and transition to new stage."""
        now_perf = time.perf_counter()

        if self._current_stage_record:
            self._current_stage_record.end_time = utcnow_iso()
            self._current_stage_record.duration_seconds = round(now_perf - self._stage_start_perf, 3)
            self._current_stage_record.status = "COMPLETED"

        self.current_state = new_state
        self._stage_start_perf = now_perf

        self._current_stage_record = StageRecord(
            stage=new_state,
            details=details or {},
        )
        self.stage_history.append(self._current_stage_record)
        logger.info(f"Workflow [{self.execution_id}] -> {new_state}")

    def fail(self, error_message: str) -> None:
        """Mark workflow as failed."""
        now_perf = time.perf_counter()
        if self._current_stage_record:
            self._current_stage_record.end_time = utcnow_iso()
            self._current_stage_record.duration_seconds = round(now_perf - self._stage_start_perf, 3)
            self._current_stage_record.status = "FAILED"
            self._current_stage_record.details["error"] = error_message

        self.current_state = WorkflowState.FAILED
        logger.error(f"Workflow [{self.execution_id}] FAILED: {error_message}")

    def get_summary(self) -> List[Dict[str, Any]]:
        """Return history of all completed stages."""
        return [record.model_dump() for record in self.stage_history]

