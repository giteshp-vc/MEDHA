from abc import ABC, abstractmethod
import time
from typing import Any, Dict, List, Optional

from core.llm_client import BaseLLMClient, LLMClient
from models.agent_models import AgentConfig, AgentResponse
from models.schemas import AgentStatus
from utils.helpers import generate_id
from utils.logger import get_logger

logger = get_logger("base_agent")


class BaseAgent(ABC):
    """Abstract base class for all MEDHA agents."""

    def __init__(
        self,
        config: AgentConfig,
        llm_client: Optional[BaseLLMClient] = None,
    ):
        self.agent_id: str = config.agent_id or generate_id("agent")
        self.role: str = config.role
        self.objective: str = config.objective
        self.instructions: str = config.instructions
        self.assigned_task_id: str = config.assigned_task_id
        self.assigned_subtask_id: str = config.assigned_subtask_id
        self.model_name: str = config.model_name
        self.temperature: float = config.temperature
        self.status: AgentStatus = AgentStatus.INITIALIZED

        # Agent state tracking
        self.findings: List[str] = []
        self.claims: List[str] = []
        self.assumptions: List[str] = []
        self.evidence_references: List[str] = []
        self.unresolved_questions: List[str] = []

        self.client = llm_client or LLMClient.get_instance()
        self.last_execution_duration: float = 0.0

    def update_status(self, new_status: AgentStatus) -> None:
        """Update the agent's current lifecycle status."""
        self.status = new_status
        logger.debug(f"Agent [{self.role} ({self.agent_id})] status -> {new_status}")

    def record_finding(self, finding: str) -> None:
        """Add a finding to the agent's state."""
        if finding and finding not in self.findings:
            self.findings.append(finding)

    def record_claim(self, claim: str) -> None:
        """Add a factual or analytical claim."""
        if claim and claim not in self.claims:
            self.claims.append(claim)

    def record_assumption(self, assumption: str) -> None:
        """Add an assumption."""
        if assumption and assumption not in self.assumptions:
            self.assumptions.append(assumption)

    def record_evidence_ref(self, ref: str) -> None:
        """Record an evidence reference."""
        if ref and ref not in self.evidence_references:
            self.evidence_references.append(ref)

    def record_unresolved_question(self, question: str) -> None:
        """Record an open or unresolved question."""
        if question and question not in self.unresolved_questions:
            self.unresolved_questions.append(question)

    def get_summary(self) -> Dict[str, Any]:
        """Return a structured dictionary snapshot of the agent."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "objective": self.objective,
            "status": self.status,
            "assigned_task_id": self.assigned_task_id,
            "assigned_subtask_id": self.assigned_subtask_id,
            "findings_count": len(self.findings),
            "claims_count": len(self.claims),
            "assumptions_count": len(self.assumptions),
            "unresolved_questions_count": len(self.unresolved_questions),
        }

    @abstractmethod
    async def execute(self, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Execute the assigned task with incoming context and return a structured response."""
        pass

