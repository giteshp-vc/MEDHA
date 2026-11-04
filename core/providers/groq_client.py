import json
from typing import Any, Dict, List, Optional, Type, TypeVar
from groq import AsyncGroq
from pydantic import BaseModel

from config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL,
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
    LLMAuthError,
    LLMBudgetExceededError,
    LLMClientError,
    LLMModelNotFoundError,
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    LLMServerError,
    LLMStructuredOutputError,
    LLMTimeoutError,
    classify_error,
    execute_with_retry,
    extract_status_code,
)
from utils.token_tracker import TokenTracker

logger = get_logger("groq_client")
T = TypeVar("T", bound=BaseModel)


class GroqClient(BaseLLMClient):
    """Groq LLM Client supporting async completions, JSON mode, retry, and concurrency-safe token tracking."""

    _instance: Optional["GroqClient"] = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        token_tracker: Optional[TokenTracker] = None,
        client: Optional[AsyncGroq] = None,
    ):
        resolved_key = api_key if api_key is not None else GROQ_API_KEY
        super().__init__(
            api_key=resolved_key,
            default_model=default_model or GROQ_MODEL,
            token_tracker=token_tracker or TokenTracker(name="groq_client"),
        )
        if client is not None:
            self._client = client
        elif self.api_key:
            self._client = AsyncGroq(api_key=self.api_key)
        else:
            self._client = None

    @classmethod
    def get_instance(cls) -> "GroqClient":
        """Get or initialize singleton GroqClient instance."""
        if cls._instance is None:
            validate_settings()
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance."""
        cls._instance = None

    def _ensure_client(self) -> AsyncGroq:
        if self._client is None:
            if not self.api_key:
                raise LLMClientError("GROQ_API_KEY is not configured.", status_code=401)
            self._client = AsyncGroq(api_key=self.api_key)
        return self._client

    @staticmethod
    def _is_unsupported_format_error(err: Exception) -> bool:
        """Determine if an error explicitly indicates that response_format='json_object' is unsupported.

        Crucially: Quota exhaustion, rate limits, authentication failures, timeouts, and server errors
        must NEVER be classified as format errors.
        """
        # 1. Any quota exhaustion, auth failure, rate limit, timeout, or server error is NOT a format error
        _, classified = classify_error(err)
        if isinstance(
            classified,
            (
                LLMQuotaExhaustedError,
                LLMBudgetExceededError,
                LLMAuthError,
                LLMRateLimitError,
                LLMServerError,
                LLMTimeoutError,
                LLMModelNotFoundError,
            ),
        ):
            return False

        # 2. Check HTTP status code
        code = extract_status_code(err)
        if code in (401, 403, 404, 429, 500, 502, 503, 504):
            return False

        # 3. Check for explicit format incompatibility keywords
        msg = str(err).lower()
        format_keywords = [
            "response_format",
            "json_object",
            "json mode",
            "response format",
            "unsupported response format",
            "not supported with this model",
        ]
        return any(kw in msg for kw in format_keywords)

    async def _track_usage_from_response(self, response: Any) -> None:
        """Extract and record token usage metadata from Groq response."""
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
        candidate_tokens = getattr(usage, "completion_tokens", 0) or 0 if usage else 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0 if usage else 0

        await self.token_tracker.add_usage(
            prompt_tokens=prompt_tokens,
            candidate_tokens=candidate_tokens,
            total_tokens=total_tokens,
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
        """Generate unstructured text from Groq with retry, token tracking, and central budget check."""
        self.token_tracker.check_call_budget(MAX_LLM_CALLS_PER_TASK)

        client = self._ensure_client()
        target_model = model or self.default_model
        temp = temperature if temperature is not None else TEMPERATURE
        output_limit = max_tokens or MAX_OUTPUT_TOKENS_PER_CALL

        messages: List[Dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        estimated_input_tokens = self.token_tracker.estimate_tokens(
            "\n".join(message["content"] for message in messages)
        )

        async def _call() -> str:
            kwargs: Dict[str, Any] = {
                "model": target_model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": output_limit,
            }
            if timeout_seconds is not None:
                kwargs["timeout"] = timeout_seconds

            async with self.token_tracker.track_call(
                MAX_LLM_CALLS_PER_TASK,
                estimated_input_tokens=estimated_input_tokens,
                max_output_tokens=output_limit,
            ):
                response = await client.chat.completions.create(**kwargs)
                await self._track_usage_from_response(response)

            content = ""
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    content = choice.message.content or ""
            if not content:
                raise LLMClientError("Empty response returned by Groq model", status_code=500)
            return content

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
        """Generate structured JSON adhering to a Pydantic schema using Groq JSON mode and validation."""
        self.token_tracker.check_call_budget(MAX_LLM_CALLS_PER_TASK)

        client = self._ensure_client()
        target_model = model or self.default_model
        temp = temperature if temperature is not None else TEMPERATURE
        output_limit = max_tokens or MAX_OUTPUT_TOKENS_PER_CALL

        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        schema_instruction = (
            f"You are a structured data generator. You MUST respond ONLY with valid JSON "
            f"strictly adhering to the following JSON schema:\n{schema_json}\n"
            f"Do not include any markdown fences (e.g. ```json), explanations, or text outside the JSON object."
        )
        full_system = f"{system_instruction.strip()}\n\n{schema_instruction}" if system_instruction else schema_instruction

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": f"{prompt}\n\nPlease output valid JSON matching the schema."},
        ]
        estimated_input_tokens = self.token_tracker.estimate_tokens(
            "\n".join(message["content"] for message in messages)
        )

        async def _call() -> T:
            kwargs: Dict[str, Any] = {
                "model": target_model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": output_limit,
            }
            if timeout_seconds is not None:
                kwargs["timeout"] = timeout_seconds

            async with self.token_tracker.track_call(
                MAX_LLM_CALLS_PER_TASK,
                estimated_input_tokens=estimated_input_tokens,
                max_output_tokens=output_limit,
            ):
                # Try JSON mode first where supported
                try:
                    response = await client.chat.completions.create(
                        **kwargs,
                        response_format={"type": "json_object"},
                    )
                except Exception as format_err:
                    # STRICT REQUIREMENT: Only fall back if the model explicitly reports unsupported response_format.
                    # Quota errors (429), auth errors (401), server errors (500), timeouts, etc. must NOT fall back.
                    if self._is_unsupported_format_error(format_err):
                        logger.warning(
                            f"Groq json_object format explicitly unsupported ({format_err}); "
                            f"falling back to standard completion with schema prompt..."
                        )
                        response = await client.chat.completions.create(**kwargs)
                    else:
                        logger.error(
                            f"Groq request failed with non-format error ({format_err.__class__.__name__}: {format_err}). "
                            f"Not falling back to plain text."
                        )
                        raise

                await self._track_usage_from_response(response)

            raw_text = ""
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    raw_text = choice.message.content or ""
            if not raw_text:
                raise LLMStructuredOutputError(
                    response_schema.__name__, raw_text, ValueError("Empty structured response.")
                )

            # Validate against Pydantic schema
            try:
                clean_dict = parse_json_safely(raw_text)
                return response_schema.model_validate(clean_dict)
            except Exception as err:
                logger.warning(f"Initial schema validation failed: {err}. Raw text: {raw_text[:200]}")
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
