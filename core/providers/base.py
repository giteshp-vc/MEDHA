import asyncio
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Optional, Type, TypeVar
from pydantic import BaseModel

from config.settings import MAX_LLM_CALLS_PER_TASK, MAX_RETRIES, MODEL_VERSION
from utils.token_tracker import TokenTracker
from models.provider_call import ProviderCallStatus

T = TypeVar("T", bound=BaseModel)


class BaseLLMClient(ABC):
    """Abstract base class for all MEDHA LLM provider implementations."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        token_tracker: Optional[TokenTracker] = None,
    ):
        self.api_key = api_key
        self.default_model = default_model
        self.token_tracker = token_tracker or TokenTracker(name="llm_client")

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: int = MAX_RETRIES,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate unstructured text from the LLM provider."""
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: int = MAX_RETRIES,
        max_tokens: Optional[int] = None,
    ) -> T:
        """Generate structured data adhering to a Pydantic schema."""
        pass

    def get_token_tracker(self) -> TokenTracker:
        """Return the token consumption tracker associated with this client."""
        return self.token_tracker

    def provider_call(
        self,
        *,
        attempt_number: int,
        retry_number: int = 0,
        is_fallback: bool = False,
        model: Optional[str],
        temperature: Optional[float],
        output_token_limit: Optional[int],
        timeout_seconds: Optional[float],
        estimated_prompt_tokens: Optional[int],
    ):
        """Create common telemetry around one actual provider transport attempt."""
        operator_model_version = MODEL_VERSION.strip() if MODEL_VERSION and MODEL_VERSION.strip() else None
        if operator_model_version and operator_model_version.casefold() == "unknown":
            operator_model_version = None
        return self.token_tracker.provider_call(
            attempt_number=attempt_number,
            retry_number=retry_number,
            is_fallback=is_fallback,
            operator_model_version=operator_model_version,
            model_identifier_policy="provider_may_resolve_mutable_alias",
            provider=self.__class__.__name__.removesuffix("Client").lower(),
            model=model,
            model_version=None,
            temperature=temperature,
            input_token_limit=self.token_tracker.max_input_tokens_per_call,
            output_token_limit=output_token_limit,
            timeout_seconds=timeout_seconds,
            estimated_prompt_tokens=estimated_prompt_tokens,
            estimated_candidate_tokens=None,
        )

    async def execute_provider_request(
        self,
        request: Callable[[], Awaitable[Any]],
        *,
        attempt_number: int,
        retry_number: int = 0,
        is_fallback: bool = False,
        model: Optional[str],
        temperature: Optional[float],
        output_token_limit: int,
        timeout_seconds: Optional[float],
        estimated_prompt_tokens: int,
        usage_callback: Callable[[Any, Any], Awaitable[None]],
        attempt_recorders: Optional[list[Any]] = None,
    ) -> tuple[Any, Any]:
        """Run and account for exactly one wire request using the shared provider instrumentation."""
        request_started = False
        response_received = False
        call_recorder = None
        try:
            async with self.token_tracker.track_call(
                MAX_LLM_CALLS_PER_TASK,
                estimated_input_tokens=estimated_prompt_tokens,
                max_output_tokens=output_token_limit,
            ):
                async with self.provider_call(
                    attempt_number=attempt_number,
                    retry_number=retry_number,
                    is_fallback=is_fallback,
                    model=model,
                    temperature=temperature,
                    output_token_limit=output_token_limit,
                    timeout_seconds=timeout_seconds,
                    estimated_prompt_tokens=estimated_prompt_tokens,
                ) as call_recorder:
                    request_started = True
                    if attempt_recorders is not None and call_recorder is not None:
                        try:
                            attempt_recorders.append(call_recorder)
                        except Exception:
                            pass
                    response = await request()
                    response_received = True
                    if call_recorder is not None:
                        call_recorder.set_response_metadata(response)
                # The provider call succeeded even if local accounting later reports an over-budget response.
                await usage_callback(response, call_recorder)
            return response, call_recorder
        except asyncio.CancelledError:
            if request_started and not response_received:
                await self.token_tracker.record_failure()
            raise
        except Exception:
            if request_started and not response_received:
                await self.token_tracker.record_failure()
            raise

    @staticmethod
    def record_provider_usage(
        call_recorder: Any,
        prompt_tokens: Optional[int],
        candidate_tokens: Optional[int],
        total_tokens: Optional[int],
    ) -> None:
        """Attach provider-reported usage best-effort without affecting a successful response."""
        if call_recorder is None:
            return
        try:
            call_recorder.set_provider_usage(prompt_tokens, candidate_tokens, total_tokens)
        except Exception:
            return

    @staticmethod
    def mark_timed_out_provider_calls(call_recorders: list[Any], error: Exception) -> None:
        """Reclassify transport cancellations when the retry wrapper confirms a timeout."""
        if error.__class__.__name__ not in {"LLMTimeoutError", "TimeoutError"}:
            return
        for recorder in call_recorders:
            if recorder is None or recorder.record.status != ProviderCallStatus.CANCELLED:
                continue
            try:
                recorder.finish(ProviderCallStatus.TIMED_OUT, error)
            except Exception:
                continue

