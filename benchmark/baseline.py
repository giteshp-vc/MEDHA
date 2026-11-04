import time
from typing import Any, Dict, Optional
from pydantic import Field

from core.llm_client import BaseLLMClient, create_llm_client
from models.schemas import BaseSchema
from utils.helpers import utcnow_iso
from utils.logger import get_logger
from utils.token_tracker import TokenTracker

logger = get_logger("baseline")

BASELINE_SYSTEM_PROMPT = (
    "You are an AI assistant answering complex analytical, technical, or planning questions directly. "
    "Provide a comprehensive, accurate, and structured answer addressing all aspects of the user's prompt."
)


class BaselineResult(BaseSchema):
    """Result produced by the conventional single-agent baseline."""

    task: str
    answer: str
    duration_seconds: float
    token_usage: Dict[str, Any] = Field(default_factory=dict)
    model: str
    timestamp: str = Field(default_factory=utcnow_iso)
    success: bool = True
    error: Optional[str] = None


class SingleAgentBaseline:
    """Executes tasks using a single Gemini model with no dynamic multi-agent orchestration or verification."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ):
        self.tracker = TokenTracker(name="baseline_tracker")
        self.client = llm_client or create_llm_client(token_tracker=self.tracker)
        self.model = model or self.client.default_model
        self.temperature = temperature

    async def run(self, task: str) -> BaselineResult:
        """Execute a user task directly with the single-agent baseline."""
        logger.info(f"Running single-agent baseline for task: '{task[:60]}...'")
        start_time = time.perf_counter()

        try:
            answer = await self.client.generate_text(
                prompt=task,
                system_instruction=BASELINE_SYSTEM_PROMPT,
                model=self.model,
                temperature=self.temperature,
            )
            duration = round(time.perf_counter() - start_time, 3)
            summary = await self.tracker.get_summary()

            return BaselineResult(
                task=task,
                answer=answer,
                duration_seconds=duration,
                token_usage=summary,
                model=self.model,
                success=True,
            )

        except Exception as e:
            duration = round(time.perf_counter() - start_time, 3)
            logger.error(f"Single-agent baseline failed: {e}")
            summary = await self.tracker.get_summary()

            return BaselineResult(
                task=task,
                answer="",
                duration_seconds=duration,
                token_usage=summary,
                model=self.model,
                success=False,
                error=str(e),
            )

