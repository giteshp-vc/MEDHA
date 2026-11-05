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

    async def _track_usage_from_response(self, response: Any, call_recorder: Any = None) -> None:
        """Extract and record token usage metadata from Groq response."""
        usage = getattr(response, "usage", None)
        prompt_value = getattr(usage, "prompt_tokens", None) if usage else None
        candidate_value = getattr(usage, "completion_tokens", None) if usage else None
        total_value = getattr(usage, "total_tokens", None) if usage else None
        prompt_tokens = prompt_value if isinstance(prompt_value, int) and not isinstance(prompt_value, bool) else None
        candidate_tokens = candidate_value if isinstance(candidate_value, int) and not isinstance(candidate_value, bool) else None
        total_tokens = total_value if isinstance(total_value, int) and not isinstance(total_value, bool) else None

        self.record_provider_usage(call_recorder, prompt_tokens, candidate_tokens, total_tokens)
        if call_recorder is not None and candidate_tokens is None:
            try:
                content = response.choices[0].message.content if response.choices else None
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
        retry_attempt = 0
        wire_attempt = 0
        attempt_recorders: Dict[int, List[Any]] = {}

        def on_attempt_failure(attempt_number: int, error: Exception) -> None:
            self.mark_timed_out_provider_calls(
                attempt_recorders.get(attempt_number, []),
                error,
            )

        async def _call() -> str:
            nonlocal retry_attempt, wire_attempt
            retry_attempt += 1
            wire_attempt += 1
            kwargs: Dict[str, Any] = {
                "model": target_model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": output_limit,
            }
            if timeout_seconds is not None:
                kwargs["timeout"] = timeout_seconds

            response, _ = await self.execute_provider_request(
                lambda: client.chat.completions.create(**kwargs),
                attempt_number=wire_attempt,
                model=target_model,
                temperature=temp,
                output_token_limit=output_limit,
                timeout_seconds=timeout_seconds,
                estimated_prompt_tokens=estimated_input_tokens,
                usage_callback=self._track_usage_from_response,
                attempt_recorders=attempt_recorders.setdefault(retry_attempt, []),
            )

            content = ""
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    content = choice.message.content or ""
            if not content:
                raise LLMClientError("Empty response returned by Groq model", status_code=500)
            return content

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
        retry_attempt = 0
        wire_attempt = 0
        attempt_recorders: Dict[int, List[Any]] = {}

        def on_attempt_failure(attempt_number: int, error: Exception) -> None:
            self.mark_timed_out_provider_calls(
                attempt_recorders.get(attempt_number, []),
                error,
            )

        async def _call() -> T:
            nonlocal retry_attempt, wire_attempt
            retry_attempt += 1
            kwargs: Dict[str, Any] = {
                "model": target_model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": output_limit,
            }
            if timeout_seconds is not None:
                kwargs["timeout"] = timeout_seconds

            async def request_once(use_json_mode: bool):
                nonlocal wire_attempt
                wire_attempt += 1

                async def send_request():
                    if use_json_mode:
                        return await client.chat.completions.create(
                            **kwargs,
                            response_format={"type": "json_object"},
                        )
                    return await client.chat.completions.create(**kwargs)

                return await self.execute_provider_request(
                    send_request,
                    attempt_number=wire_attempt,
                    retry_number=retry_attempt - 1,
                    is_fallback=not use_json_mode,
                    model=target_model,
                    temperature=temp,
                    output_token_limit=output_limit,
                    timeout_seconds=timeout_seconds,
                    estimated_prompt_tokens=estimated_input_tokens,
                    usage_callback=self._track_usage_from_response,
                    attempt_recorders=attempt_recorders.setdefault(retry_attempt, []),
                )

            # Try JSON mode first where supported. The fallback is a distinct provider request.
            try:
                response, _ = await request_once(use_json_mode=True)
            except Exception as format_err:
                if self._is_unsupported_format_error(format_err):
                    logger.warning(
                        "Groq JSON response format is unsupported (%s); retrying without response_format.",
                        type(format_err).__name__,
                    )
                    response, _ = await request_once(use_json_mode=False)
                else:
                    logger.error(
                        "Groq structured request failed (%s); not retrying as plain text.",
                        type(format_err).__name__,
                    )
                    raise

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

        return await execute_with_retry(
            _call,
            max_retries=max_retries,
            initial_delay=INITIAL_RETRY_DELAY,
            max_delay=MAX_RETRY_DELAY,
            timeout_seconds=timeout_seconds,
            on_attempt_failure=on_attempt_failure,
        )
