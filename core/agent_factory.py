from typing import Any, Dict, List, Optional

from agents.analyst_agent import AnalystAgent
from agents.base_agent import BaseAgent
from agents.critic_agent import CriticAgent
from agents.research_agent import ResearchAgent
from agents.reviewer_agent import ReviewerAgent
from agents.specialist_agent import SpecialistAgent
from agents.synthesizer_agent import SynthesizerAgent
from agents.technical_agent import TechnicalAgent
from core.llm_client import BaseLLMClient, LLMClient
from models.agent_models import AgentConfig
from models.task_models import SubTask
from utils.helpers import generate_id
from utils.logger import get_logger

logger = get_logger("agent_factory")


class AgentFactory:
    """Factory responsible for dynamically instantiating, configuring, and tracking specialist agents."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.client = llm_client or LLMClient.get_instance()
        self._registry: Dict[str, BaseAgent] = {}
        self._role_map: Dict[str, str] = {}  # (role, subtask_id) -> agent_id to prevent duplicates

    def create_from_subtask(
        self,
        subtask: SubTask,
        task_id: str = "",
        model_name: Optional[str] = None,
        temperature: float = 0.2,
    ) -> BaseAgent:
        """Dynamically instantiate an appropriate agent for a planned subtask."""
        key = f"{subtask.role}::{subtask.id}"
        if key in self._role_map:
            existing_id = self._role_map[key]
            if existing_id in self._registry:
                logger.info(f"Reusing existing agent [{existing_id}] for subtask {subtask.id}")
                return self._registry[existing_id]

        agent_id = generate_id("agent")
        config = AgentConfig(
            agent_id=agent_id,
            role=subtask.role,
            objective=subtask.objective,
            instructions=subtask.instructions,
            assigned_task_id=task_id,
            assigned_subtask_id=subtask.id,
            model_name=model_name or self.client.default_model,
            temperature=temperature,
        )

        # Route to specialized subclass if title fits known archetypes, otherwise instantiate dynamic SpecialistAgent
        role_lower = subtask.role.lower()
        if "research" in role_lower or "literature" in role_lower:
            agent = ResearchAgent(config, llm_client=self.client)
        elif "analyst" in role_lower or "quantitative" in role_lower or "performance" in role_lower:
            agent = AnalystAgent(config, llm_client=self.client)
        elif "technical" in role_lower or "engineer" in role_lower or "architect" in role_lower or "developer" in role_lower:
            agent = TechnicalAgent(config, llm_client=self.client)
        elif "critic" in role_lower or "audit" in role_lower or "security" in role_lower:
            agent = CriticAgent(config, llm_client=self.client)
        else:
            # Dynamic specialist agent accommodates any arbitrary profession
            agent = SpecialistAgent(config, llm_client=self.client)

        self._registry[agent_id] = agent
        self._role_map[key] = agent_id

        logger.info(f"Created dynamic agent: [{agent.role}] (ID: {agent_id}, Subtask: {subtask.id})")
        return agent

    def create_reviewer(
        self,
        model_name: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> ReviewerAgent:
        """Instantiate an Adversarial Reviewer agent."""
        agent_id = generate_id("reviewer")
        config = AgentConfig(
            agent_id=agent_id,
            role="Adversarial Claim Reviewer",
            objective="Independently interrogate claims and reject unsupported assertions.",
            instructions=instructions or "Apply strict evidentiary criteria. Do not accept by consensus.",
            model_name=model_name or self.client.default_model,
            temperature=0.1,
        )
        agent = ReviewerAgent(config=config, llm_client=self.client)
        self._registry[agent_id] = agent
        return agent

    def create_synthesizer(
        self,
        model_name: Optional[str] = None,
    ) -> SynthesizerAgent:
        """Instantiate a Synthesizer agent."""
        agent_id = generate_id("synthesizer")
        config = AgentConfig(
            agent_id=agent_id,
            role="Final Synthesizer",
            objective="Synthesize verified findings into a transparent final answer.",
            instructions="Exclude rejected claims. Disclose uncertainties and assumptions.",
            model_name=model_name or self.client.default_model,
            temperature=0.2,
        )
        agent = SynthesizerAgent(config=config, llm_client=self.client)
        self._registry[agent_id] = agent
        return agent

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Retrieve an agent by its unique identifier."""
        return self._registry.get(agent_id)

    def list_agents(self) -> List[BaseAgent]:
        """Return all instantiated agents."""
        return list(self._registry.values())

    def clear(self) -> None:
        """Clear factory registry and role cache."""
        self._registry.clear()
        self._role_map.clear()

