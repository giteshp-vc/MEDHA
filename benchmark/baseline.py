import asyncio
import time
from typing import Any, Dict, List, Optional
from pydantic import Field

from core.llm_client import BaseLLMClient, create_llm_client
from config import settings
from models.provider_call import AccountingContext, AccountingSnapshot, ProviderCallRecord
from models.schemas import BaseSchema
from utils.helpers import generate_id, utcnow_iso
from utils.logger import ExecutionRecord, get_logger
from utils.token_tracker import TokenTracker

logger = get_logger("baseline")

BASELINE_SYSTEM_PROMPT = (
    "You are an AI assistant answering complex analytical, technical, or planning questions directly. "
    "Provide a comprehensive, accurate, and structured answer addressing all aspects of the user's prompt."
)


def _safe_error_diagnostic(error: Exception) -> str:
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        status_code = getattr(error, "code", None)
    if isinstance(status_code, int):
        return f"{type(error).__name__} (status {status_code})"
    return type(error).__name__


class BaselineResult(BaseSchema):
    """Result produced by the conventional single-agent baseline."""

    task: str
    answer: str
    execution_id: Optional[str] = None
    duration_seconds: float
    token_usage: Dict[str, Any] = Field(default_factory=dict)
    model: str
    timestamp: str = Field(default_factory=utcnow_iso)
    success: bool = True
    error: Optional[str] = None
    accounting: Optional[AccountingSnapshot] = None
    provider_call_records: List[ProviderCallRecord] = Field(default_factory=list)


class SingleAgentBaseline:
    """Executes tasks using a single Gemini model with no dynamic multi-agent orchestration or verification."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ):
        if llm_client is not None:
            self.client = llm_client
            self.tracker = llm_client.get_token_tracker()
        else:
            self.tracker = TokenTracker(name="baseline_tracker")
            self.client = create_llm_client(token_tracker=self.tracker)
        self.model = model or self.client.default_model
        self.temperature = temperature

    def benchmark_configuration(self) -> Dict[str, Any]:
        tracker = self.client.get_token_tracker()
        return {
            "provider": self.client.__class__.__name__.replace("Client", "").lower(),
            "model": self.model or "unknown",
            "model_version": getattr(settings, "MODEL_VERSION", "unknown"),
            "temperature": self.temperature,
            "input_token_limit": tracker.max_input_tokens_per_call,
            "output_token_limit": tracker.max_output_tokens_per_call,
            "task_token_budget": tracker.max_tokens_per_task,
            "call_budget": getattr(settings, "MAX_LLM_CALLS_PER_TASK", None),
        }

    async def run(
        self,
        task: str,
        accounting_context: Optional[AccountingContext] = None,
    ) -> BaselineResult:
        """Execute a user task directly with the single-agent baseline."""
        logger.info(f"Running single-agent baseline for task: '{task[:60]}...'")
        start_time = time.perf_counter()
        execution_id = generate_id("baseline")
        execution_record = ExecutionRecord(
            execution_id=execution_id,
            task_description=f"Single-agent baseline execution {execution_id}",
        )
        context = accounting_context or AccountingContext()
        context = AccountingContext.model_validate({
            **context.model_dump(exclude_none=True),
            "execution_id": execution_id,
            "scope_kind": "baseline_execution",
        })
        accounting: Optional[AccountingSnapshot] = None
        try:
            async with self.tracker.accounting_session(context, isolate_usage=True) as accounting_scope:
                with self.tracker.call_context(execution_id=execution_id, stage="baseline"):
                    try:
                        async with self.tracker.task_budget():
                            try:
                                with self.tracker.operation("baseline"):
                                    answer = await self.client.generate_text(
                                        prompt=task,
                                        system_instruction=BASELINE_SYSTEM_PROMPT,
                                        model=self.model,
                                        temperature=self.temperature,
                                    )
                                success = bool(answer.strip())
                                error = None if success else "Baseline provider returned empty output."
                            except asyncio.CancelledError:
                                accounting = accounting_scope.snapshot()
                                raise
                            except Exception as exc:
                                logger.error("Single-agent baseline failed: %s", type(exc).__name__)
                                answer = ""
                                success = False
                                error = _safe_error_diagnostic(exc)
                            summary = await self.tracker.get_summary()
                            accounting = accounting_scope.snapshot()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.error("Single-agent baseline failed: %s", type(exc).__name__)
                        answer = ""
                        success = False
                        error = _safe_error_diagnostic(exc)
                        summary = await self.tracker.get_summary()
                        accounting = accounting_scope.snapshot()

                duration = round(time.perf_counter() - start_time, 3)
                result = BaselineResult(
                    task=task,
                    answer=answer,
                    execution_id=execution_id,
                    duration_seconds=duration,
                    token_usage=summary,
                    model=self.model,
                    success=success,
                    error=error,
                    accounting=accounting,
                    provider_call_records=accounting.provider_calls if accounting else [],
                )
                execution_record.metadata["provider_calls"] = [
                    call.model_dump(mode="json") for call in result.provider_call_records
                ]
                execution_record.metadata["provider_call_records_dropped"] = (
                    accounting.dropped_provider_call_records if accounting else 0
                )
                execution_record.metadata["accounting"] = (
                    accounting.model_dump(mode="json") if accounting else None
                )
                if error is not None:
                    execution_record.record_error(
                        component="baseline",
                        message=f"Baseline request failed ({error}).",
                    )
                execution_record.finish(
                    status="SUCCESS" if success else "FAILED",
                    token_usage=summary,
                )
                execution_record.save()
                return result
        except asyncio.CancelledError as error:
            try:
                error.accounting_snapshot = accounting
            except Exception:
                pass
            records = self.tracker.get_provider_call_records(execution_id)
            execution_record.metadata["provider_calls"] = [
                call.model_dump(mode="json") for call in records
            ]
            execution_record.record_error("baseline", "Baseline execution was cancelled.")
            execution_record.finish(
                status="CANCELLED",
                token_usage=(accounting.token_usage if accounting else None),
            )
            execution_record.save()
            raise

