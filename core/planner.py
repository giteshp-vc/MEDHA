from typing import Any, Dict, List, Optional

from core.llm_client import BaseLLMClient, LLMClient
from models.task_models import SubTask, TaskPlan
from utils.helpers import parse_json_safely
from utils.logger import get_logger
from utils.retry import LLMBudgetExceededError, LLMQuotaExhaustedError

logger = get_logger("planner")

PLANNER_SYSTEM_PROMPT = (
    "You are the Dynamic Task Decomposition and Strategy Planner for the MEDHA Multi-Agent System.\n"
    "Your responsibility is to analyze the user's problem, dynamically determine the precise specialist "
    "domain expertise required, break the problem into focused subtasks, and establish dependency relationships.\n\n"
    "CRITICAL PLANNING RULES:\n"
    "1. DYNAMIC ROLES: Never hardcode or reuse the same generic roles. The specialist roles MUST be customized "
    "strictly to the domain of the input prompt (e.g. cybersecurity, distributed databases, pharmacology, "
    "compiler design, macroeconomic policy, frontend accessibility, etc.).\n"
    "2. PARALLELISM & DEPENDENCIES: Independent subtasks MUST have empty dependencies ('[]') so they execute in parallel.\n"
    "Add a dependency only when the subtask cannot produce a useful, independent result without a specific earlier output; shared topic or broad architecture scope is not a dependency.\n"
    "Default architecture, privacy, security, reliability, and deployment reviews of an attendance system to independent execution; add a dependency only when the objective explicitly requires a named upstream artifact.\n"
    "Subtasks that strictly rely on earlier findings must list the specific subtask IDs (e.g. ['T1', 'T2']) in 'dependencies'.\n"
    "3. SCOPE: Generate between 2 to 5 highly relevant subtasks.\n"
    "4. COMPLETION CRITERIA: Define explicit, measurable conditions for full task resolution."
)


class TaskPlanner:
    """Decomposes user tasks dynamically into specialist subtasks with dependency graphs."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.client = llm_client or LLMClient.get_instance()

    async def create_plan(
        self,
        task_description: str,
        max_subtasks: int = 4,
    ) -> TaskPlan:
        """Analyze a user task and generate a structured TaskPlan."""
        logger.info(f"TaskPlanner generating plan for: '{task_description[:70]}...'")

        prompt = (
            f"Analyze the following user task and create an optimal execution plan:\n\n"
            f"TASK:\n{task_description}\n\n"
            f"Produce a structured JSON plan with:\n"
            f"- 'task': synthesized statement of the primary goal\n"
            f"- 'subtasks': list of 2 to {max_subtasks} specialized subtasks, each with:\n"
            f"  - 'id': sequential ID (e.g. 'T1', 'T2', 'T3')\n"
            f"  - 'role': specific specialist title tailored to this task\n"
            f"  - 'objective': concrete objective of this subtask\n"
            f"  - 'instructions': specific focus areas or methodologies for this specialist\n"
            f"  - 'dependencies': list of subtask IDs that must complete BEFORE this one can begin (use [] for parallel subtasks)\n"
            f"  - 'expected_output': expected deliverable description\n"
            f"- 'rationale': why these specific specialist roles were selected\n"
            f"- 'completion_criteria': 2-4 validation checkpoints to verify successful completion"
        )

        try:
            plan = await self.client.generate_structured(
                prompt=prompt,
                response_schema=TaskPlan,
                system_instruction=PLANNER_SYSTEM_PROMPT,
                temperature=0.2,
            )
            # Validate plan sanity
            if not plan.subtasks:
                raise ValueError("Planner returned an empty list of subtasks.")

            logger.info(
                f"Generated plan with {len(plan.subtasks)} subtasks: "
                f"{[s.role for s in plan.subtasks]}"
            )
            return plan

        except (LLMBudgetExceededError, LLMQuotaExhaustedError):
            raise
        except Exception as e:
            logger.warning(f"Structured plan generation failed ({e}); attempting text fallback...")
            try:
                raw_text = await self.client.generate_text(
                    prompt=prompt + "\nReturn ONLY valid JSON.",
                    system_instruction=PLANNER_SYSTEM_PROMPT,
                    temperature=0.2,
                )
                clean_dict = parse_json_safely(raw_text)
                plan = TaskPlan.model_validate(clean_dict)
                logger.info(f"Fallback plan generated with {len(plan.subtasks)} subtasks.")
                return plan
            except Exception as fallback_err:
                logger.error(f"Fallback plan generation also failed: {fallback_err}. Creating emergency heuristic plan.")
                return self._create_emergency_plan(task_description)

    def _create_emergency_plan(self, task_description: str) -> TaskPlan:
        """Create a resilient, heuristic plan when LLM is unavailable or unparsable."""
        return TaskPlan(
            task=task_description,
            subtasks=[
                SubTask(
                    id="T1",
                    role="Primary Domain Specialist",
                    objective=f"Analyze core requirements for: {task_description[:80]}",
                    instructions="Analyze domain-specific architectures and requirements.",
                    dependencies=[],
                    expected_output="Detailed domain analysis.",
                ),
                SubTask(
                    id="T2",
                    role="Risk and Reliability Specialist",
                    objective="Assess failure modes, constraints, and scalability limits.",
                    instructions="Identify critical operational and security bottlenecks.",
                    dependencies=[],
                    expected_output="Risk and reliability evaluation.",
                ),
            ],
            rationale="Emergency fallback plan dynamically created due to planner generation failure.",
            completion_criteria=["All critical requirements addressed", "Bottlenecks identified"],
        )

