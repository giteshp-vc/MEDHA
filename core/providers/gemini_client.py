import json
from typing import Any, Optional, Type, TypeVar
from google import genai
from google.genai import types
from pydantic import BaseModel

from config.settings import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    INITIAL_RETRY_DELAY,
    MAX_LLM_CALLS_PER_TASK,
    MAX_OUTPUT_TOKENS_PER_CALL,
    MAX_RETRIES,
    MAX_RETRY_DELAY,
    TEMPERATURE,
    validate_settings,
)
from core.providers.base import BaseLLMClient
from utils.helpers import parse_json_safely
from utils.logger import get_logger
from utils.retry import (
    LLMClientError,
    LLMStructuredOutputError,
    execute_with_retry,
)
from utils.token_tracker import TokenTracker

logger = get_logger("gemini_client")
T = TypeVar("T", bound=BaseModel)


class GeminiClient(BaseLLMClient):
    """Reusable asynchronous Gemini LLM Client with retries, structured output, and token tracking."""

    _instance: Optional["GeminiClient"] = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        token_tracker: Optional[TokenTracker] = None,
        client: Optional[genai.Client] = None,
    ):
        resolved_key = api_key if api_key is not None else GEMINI_API_KEY
        super().__init__(
            api_key=resolved_key,
            default_model=default_model or GEMINI_MODEL,
            token_tracker=token_tracker or TokenTracker(name="gemini_client"),
        )
        if client is not None:
            self._client = client
        elif self.api_key:
            self._client = genai.Client(api_key=self.api_key)
        else:
            self._client = None

    @classmethod
    def get_instance(cls) -> "GeminiClient":
        """Get or initialize singleton GeminiClient instance."""
        if cls._instance is None:
            validate_settings()
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful in tests)."""
        cls._instance = None

    def _ensure_client(self) -> genai.Client:
        if self._client is None:
            if not self.api_key:
                raise LLMClientError("GEMINI_API_KEY is not configured.", status_code=401)
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def _track_usage_from_response(self, response: Any, call_recorder: Any = None) -> None:
        """Extract and record token usage metadata from response."""
        usage = getattr(response, "usage_metadata", None)
        prompt_value = getattr(usage, "prompt_token_count", None) if usage else None
        candidate_value = getattr(usage, "candidates_token_count", None) if usage else None
        total_value = getattr(usage, "total_token_count", None) if usage else None
        cached_value = getattr(usage, "cached_content_token_count", None) if usage else None
        prompt_tokens = prompt_value if isinstance(prompt_value, int) and not isinstance(prompt_value, bool) else None
        candidate_tokens = candidate_value if isinstance(candidate_value, int) and not isinstance(candidate_value, bool) else None
        total_tokens = total_value if isinstance(total_value, int) and not isinstance(total_value, bool) else None
        cached_tokens = cached_value if isinstance(cached_value, int) and not isinstance(cached_value, bool) else None

        self.record_provider_usage(call_recorder, prompt_tokens, candidate_tokens, total_tokens)
        if call_recorder is not None and candidate_tokens is None:
            try:
                content = getattr(response, "text", None)
                if isinstance(content, str):
                    call_recorder.set_estimated_candidate_tokens(
                        self.token_tracker.estimate_tokens(content)
                    )
            except Exception:
                pass

        await self.token_tracker.add_usage(
            prompt_tokens=prompt_tokens or 0,
            candidate_tokens=candidate_tokens or 0,
            total_tokens=total_tokens or 0,
            cached_tokens=cached_tokens or 0,
            usage_available=any(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in (prompt_tokens, candidate_tokens, total_tokens)
            ),
        )

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
        """Generate unstructured text from Gemini with retry and observability."""
        self.token_tracker.check_call_budget(MAX_LLM_CALLS_PER_TASK)
        client = self._ensure_client()
        target_model = model or self.default_model
        temp = temperature if temperature is not None else TEMPERATURE
        output_limit = max_tokens or MAX_OUTPUT_TOKENS_PER_CALL

        config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=output_limit,
            system_instruction=system_instruction if system_instruction else None,
        )
        estimated_input_tokens = self.token_tracker.estimate_tokens(
            prompt + (system_instruction or "")
        )
        retry_attempt = 0
        attempt_recorders: dict[int, list[Any]] = {}

        def on_attempt_failure(attempt_number: int, error: Exception) -> None:
            self.mark_timed_out_provider_calls(
                attempt_recorders.get(attempt_number, []),
                error,
            )

        async def _call():
            nonlocal retry_attempt
            retry_attempt += 1
            response, _ = await self.execute_provider_request(
                lambda: client.aio.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=config,
                ),
                attempt_number=retry_attempt,
                retry_number=retry_attempt - 1,
                model=target_model,
                temperature=temp,
                output_token_limit=output_limit,
                timeout_seconds=timeout_seconds,
                estimated_prompt_tokens=estimated_input_tokens,
                usage_callback=self._track_usage_from_response,
                attempt_recorders=attempt_recorders.setdefault(retry_attempt, []),
            )
            if not response.text:
                raise LLMClientError("Empty response returned by Gemini model", status_code=500)
            return response.text

        return await execute_with_retry(
            _call,
            max_retries=max_retries,
            initial_delay=INITIAL_RETRY_DELAY,
            max_delay=MAX_RETRY_DELAY,
            timeout_seconds=timeout_seconds,
            on_attempt_failure=on_attempt_failure,
        )

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
        """Generate structured JSON adhering to a Pydantic schema with retry and validation."""
        self.token_tracker.check_call_budget(MAX_LLM_CALLS_PER_TASK)
        client = self._ensure_client()
        target_model = model or self.default_model
        temp = temperature if temperature is not None else TEMPERATURE
        output_limit = max_tokens or MAX_OUTPUT_TOKENS_PER_CALL

        config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=output_limit,
            response_mime_type="application/json",
            response_schema=response_schema,
            system_instruction=system_instruction if system_instruction else None,
        )
        estimated_schema = json.dumps(response_schema.model_json_schema(), sort_keys=True)
        estimated_input_tokens = self.token_tracker.estimate_tokens(
            prompt + (system_instruction or "") + estimated_schema
        )
        retry_attempt = 0
        attempt_recorders: dict[int, list[Any]] = {}

        def on_attempt_failure(attempt_number: int, error: Exception) -> None:
            self.mark_timed_out_provider_calls(
                attempt_recorders.get(attempt_number, []),
                error,
            )

        async def _call() -> T:
            nonlocal retry_attempt
            retry_attempt += 1
            response, _ = await self.execute_provider_request(
                lambda: client.aio.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=config,
                ),
                attempt_number=retry_attempt,
                retry_number=retry_attempt - 1,
                model=target_model,
                temperature=temp,
                output_token_limit=output_limit,
                timeout_seconds=timeout_seconds,
                estimated_prompt_tokens=estimated_input_tokens,
                usage_callback=self._track_usage_from_response,
                attempt_recorders=attempt_recorders.setdefault(retry_attempt, []),
            )

            parsed = getattr(response, "parsed", None)
            if parsed is not None and isinstance(parsed, response_schema):
                return parsed

            raw_text = response.text or ""
            if not raw_text:
                raise LLMStructuredOutputError(
                    response_schema.__name__, raw_text, ValueError("Empty structured response.")
                )

            # Fallback to Pydantic model validation
            try:
                clean_dict = parse_json_safely(raw_text)
                return response_schema.model_validate(clean_dict)
            except Exception as e:
                logger.warning(f"Initial schema validation failed: {e}. Raw text: {raw_text[:200]}")
                try:
                    return response_schema.model_validate_json(raw_text)
                except Exception as final_err:
                        raise LLMStructuredOutputError(
                            response_schema.__name__, raw_text, final_err
                        ) from final_err

        return await execute_with_retry(
            _call,
            max_retries=max_retries,
            initial_delay=INITIAL_RETRY_DELAY,
            max_delay=MAX_RETRY_DELAY,
            timeout_seconds=timeout_seconds,
            on_attempt_failure=on_attempt_failure,
        )
