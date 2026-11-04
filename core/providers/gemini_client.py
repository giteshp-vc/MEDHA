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
    LLMBudgetExceededError,
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

    async def _track_usage_from_response(self, response: Any) -> None:
        """Extract and record token usage metadata from response."""
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0 if usage else 0
        candidate_tokens = getattr(usage, "candidates_token_count", 0) or 0 if usage else 0
        total_tokens = getattr(usage, "total_token_count", 0) or 0 if usage else 0
        cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0 if usage else 0

        await self.token_tracker.add_usage(
            prompt_tokens=prompt_tokens,
            candidate_tokens=candidate_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            usage_available=bool(usage and (prompt_tokens or candidate_tokens or total_tokens)),
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

        async def _call():
            async with self.token_tracker.track_call(
                MAX_LLM_CALLS_PER_TASK,
                estimated_input_tokens=estimated_input_tokens,
                max_output_tokens=output_limit,
            ):
                response = await client.aio.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=config,
                )
                await self._track_usage_from_response(response)
            if not response.text:
                raise LLMClientError("Empty response returned by Gemini model", status_code=500)
            return response.text

        try:
            return await execute_with_retry(
                _call,
                max_retries=max_retries,
                initial_delay=INITIAL_RETRY_DELAY,
                max_delay=MAX_RETRY_DELAY,
                timeout_seconds=timeout_seconds,
            )
        except Exception as error:
            if not isinstance(error, LLMBudgetExceededError):
                await self.token_tracker.record_failure()
            raise

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

        async def _call() -> T:
            async with self.token_tracker.track_call(
                MAX_LLM_CALLS_PER_TASK,
                estimated_input_tokens=estimated_input_tokens,
                max_output_tokens=output_limit,
            ):
                response = await client.aio.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=config,
                )
                await self._track_usage_from_response(response)

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

        try:
            return await execute_with_retry(
                _call,
                max_retries=max_retries,
                initial_delay=INITIAL_RETRY_DELAY,
                max_delay=MAX_RETRY_DELAY,
                timeout_seconds=timeout_seconds,
            )
        except Exception as error:
            if not isinstance(error, LLMBudgetExceededError):
                await self.token_tracker.record_failure()
            raise
