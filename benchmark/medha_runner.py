from typing import Any, Dict, Optional

from core.llm_client import BaseLLMClient
from core.orchestrator import MedhaOrchestrator, MedhaResult
from utils.logger import get_logger

logger = get_logger("medha_runner")


class MedhaBenchmarkRunner:
    """Benchmark runner for Mode B (MEDHA Dynamic Multi-Agent System)."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        max_concurrency: int = 4,
        max_repair_iterations: int = 2,
    ):
        self.orchestrator = MedhaOrchestrator(
            llm_client=llm_client,
            max_concurrency=max_concurrency,
            max_repair_iterations=max_repair_iterations,
        )

    async def run(self, task: str) -> MedhaResult:
        """Run task through complete MEDHA pipeline."""
        logger.info(f"Running MEDHA benchmark for: '{task[:60]}...'")
        return await self.orchestrator.run_task(task)

