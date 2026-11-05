from typing import Any, Dict, Optional

from config import settings
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

    def benchmark_configuration(self) -> Dict[str, Any]:
        tracker = self.orchestrator.tracker
        client = self.orchestrator.client
        return {
            "provider": client.__class__.__name__.replace("Client", "").lower(),
            "model": client.default_model or "unknown",
            "model_version": getattr(settings, "MODEL_VERSION", "unknown"),
            "provider_request_timeout_seconds": settings.PROVIDER_REQUEST_TIMEOUT_SECONDS,
            "temperature": 0.2,
            "input_token_limit": tracker.max_input_tokens_per_call,
            "output_token_limit": tracker.max_output_tokens_per_call,
            "task_token_budget": tracker.max_tokens_per_task,
            "call_budget": settings.MAX_LLM_CALLS_PER_TASK,
        }

    async def run(self, task: str) -> MedhaResult:
        """Run task through complete MEDHA pipeline."""
        logger.info(f"Running MEDHA benchmark for: '{task[:60]}...'")
        return await self.orchestrator.run_task(task)

