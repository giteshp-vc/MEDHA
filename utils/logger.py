import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.logging_config import setup_logging
from config.settings import LOGS_DIR

# Root MEDHA logger initialized once
_root_logger = setup_logging()


def get_logger(name: str) -> logging.Logger:
    """Return a logger child within the medha namespace."""
    if name == "medha" or name.startswith("medha."):
        return logging.getLogger(name)
    return logging.getLogger(f"medha.{name}")


class ExecutionRecord:
    """Structured record for a single execution cycle or task run."""

    def __init__(self, execution_id: str, task_description: str):
        self.execution_id = execution_id
        self.task_description = task_description
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.end_time: Optional[str] = None
        self.duration_seconds: Optional[float] = None
        self.status: str = "IN_PROGRESS"
        self.agent_records: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.message_records: List[Dict[str, Any]] = []
        self.token_usage: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def log_agent_activity(
        self,
        agent_id: str,
        role: str,
        task_id: str,
        status: str,
        duration_seconds: float,
        token_usage: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Record an individual agent execution event."""
        self.agent_records.append({
            "agent_id": agent_id,
            "role": role,
            "task_id": task_id,
            "status": status,
            "duration_seconds": round(duration_seconds, 3),
            "token_usage": token_usage or {},
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def record_error(self, component: str, message: str, details: Optional[str] = None) -> None:
        """Record an error occurrence during execution."""
        self.errors.append({
            "component": component,
            "message": message,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def log_message(self, message: Dict[str, Any]) -> None:
        """Record an inter-agent message in the execution audit trail."""
        self.message_records.append(dict(message))

    def finish(self, status: str = "COMPLETED", token_usage: Optional[Dict[str, Any]] = None) -> None:
        """Mark execution as finished and compute duration."""
        self.status = status
        self.end_time = datetime.now(timezone.utc).isoformat()
        if token_usage:
            self.token_usage = token_usage

        start_dt = datetime.fromisoformat(self.start_time)
        end_dt = datetime.fromisoformat(self.end_time)
        self.duration_seconds = round((end_dt - start_dt).total_seconds(), 3)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary representation."""
        return {
            "execution_id": self.execution_id,
            "task_description": self.task_description,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "agent_records": self.agent_records,
            "errors": self.errors,
            "message_records": self.message_records,
            "token_usage": self.token_usage,
            "metadata": self.metadata,
        }

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Persist structured execution record to a JSON log file."""
        target = filepath or (LOGS_DIR / f"execution_{self.execution_id}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return target

