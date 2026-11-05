import asyncio
from typing import Any, Dict, List, Optional
from utils.helpers import utcnow_iso
from utils.logger import get_logger

logger = get_logger("shared_context")


class SharedContext:
    """Thread-safe and async-safe shared state repository for coordinating agents."""

    def __init__(self, task_id: str = "", task_description: str = ""):
        self.task_id = task_id
        self.task_description = task_description
        self._findings: List[Dict[str, Any]] = []
        self._notes: Dict[str, Any] = {}
        self._agent_statuses: Dict[str, str] = {}
        self._communications: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def add_finding(self, agent_id: str, role: str, finding: str) -> None:
        """Record a specialist finding to the shared knowledge base."""
        async with self._lock:
            self._findings.append({
                "agent_id": agent_id,
                "role": role,
                "finding": finding,
                "timestamp": utcnow_iso(),
            })

    async def get_findings(self) -> List[Dict[str, Any]]:
        """Retrieve all shared findings."""
        async with self._lock:
            return list(self._findings)

    async def set_note(self, key: str, value: Any) -> None:
        """Store a key-value note in shared context."""
        async with self._lock:
            self._notes[key] = value

    async def get_note(self, key: str, default: Any = None) -> Any:
        """Retrieve a specific note."""
        async with self._lock:
            return self._notes.get(key, default)

    async def get_all_notes(self) -> Dict[str, Any]:
        """Retrieve all shared notes."""
        async with self._lock:
            return dict(self._notes)

    async def update_agent_status(self, agent_id: str, status: str) -> None:
        """Track agent status."""
        async with self._lock:
            self._agent_statuses[agent_id] = status

    async def add_communication(self, message: Dict[str, Any]) -> None:
        """Record an attributable communication without treating it as evidence."""
        async with self._lock:
            if message.get("task_id") and message["task_id"] != self.task_id:
                raise ValueError("Communication belongs to a different execution.")
            self._communications.append(dict(message))

    async def format_summary(self) -> str:
        """Generate a concise formatted summary of shared state for prompt inclusion."""
        async with self._lock:
            lines = [f"Task: {self.task_description}"]
            if self._findings:
                lines.append("\nAccumulated Findings:")
                for f in self._findings:
                    lines.append(f"- [{f['role']}]: {f['finding']}")
            if self._notes:
                lines.append("\nShared Notes:")
                for k, v in self._notes.items():
                    lines.append(f"- {k}: {v}")
            return "\n".join(lines)

    async def snapshot(self) -> Dict[str, Any]:
        """Return a complete dictionary snapshot of the shared context."""
        async with self._lock:
            return {
                "task_id": self.task_id,
                "task_description": self.task_description,
                "findings": list(self._findings),
                "notes": dict(self._notes),
                "agent_statuses": dict(self._agent_statuses),
                "communications": list(self._communications),
            }

