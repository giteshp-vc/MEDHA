import asyncio
import random
import re
from typing import Any, Callable, Coroutine, Optional, Tuple, TypeVar

from config.settings import INITIAL_RETRY_DELAY, MAX_RETRIES, MAX_RETRY_DELAY
from utils.logger import get_logger

logger = get_logger("retry")

T = TypeVar("T")

# Keywords indicating non-retryable hard quota or billing exhaustion
QUOTA_EXHAUSTION_KEYWORDS = [
    "daily quota",
    "per day",
    "requests per day",
    "rpd",
    "tpd",
    "tokens per day",
    "quota exhausted",
    "exceeded your current quota",
    "insufficient_quota",
    "monthly quota",
    "billing",
    "credits exhausted",
    "free tier limit",
    "day limit",
    "quota_exceeded",
]

# Per-minute windows are transient even if the provider message also links to billing.
PER_MINUTE_LIMIT_KEYWORDS = [
    "tokens per minute",
    "requests per minute",
    "(tpm)",
    "(rpm)",
    "per minute",
]


class LLMError(Exception):
    """Base exception for LLM operations."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        retryable: bool = False,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after


class LLMClientError(LLMError):
    """Non-retryable 400 Bad Request error."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message, status_code=status_code, retryable=False)


class LLMStructuredOutputError(LLMClientError):
    """A provider response was received but could not be validated against its schema."""

    def __init__(self, schema_name: str, raw_text: str, validation_error: Exception):
        self.schema_name = schema_name
        self.raw_text = raw_text
        self.validation_error = validation_error
        super().__init__(
            f"Malformed JSON or schema validation mismatch for {schema_name}: {validation_error}",
            status_code=400,
        )


class LLMAuthError(LLMError):
    """Non-retryable 401/403 Authentication or authorization error."""

    def __init__(self, message: str, status_code: int = 401):
        super().__init__(message, status_code=status_code, retryable=False)


class LLMModelNotFoundError(LLMError):
    """Non-retryable 404 Model not found or unavailable error."""

    def __init__(self, message: str, status_code: int = 404):
        super().__init__(message, status_code=status_code, retryable=False)


class LLMQuotaExhaustedError(LLMError):
    """Non-retryable daily or account-level quota exhaustion error."""

    def __init__(self, message: str, status_code: int = 429):
        super().__init__(message, status_code=status_code, retryable=False)


class LLMBudgetExceededError(LLMError):
    """Non-retryable task-level or central LLM call budget exceeded error."""

    def __init__(self, message: str, status_code: int = 429):
        super().__init__(message, status_code=status_code, retryable=False)


class LLMTokenBudgetExceededError(LLMBudgetExceededError):
    """The local application token estimate exceeds its configured task or per-call budget."""


class LLMRateLimitError(LLMError):
    """Retryable 429 transient per-minute / per-token rate limit error."""

    def __init__(self, message: str, status_code: int = 429, retry_after: Optional[float] = None):
        super().__init__(message, status_code=status_code, retryable=True, retry_after=retry_after)


class LLMServerError(LLMError):
    """Retryable 500/502/503/504 Temporary server error."""

    def __init__(self, message: str, status_code: int = 500, retry_after: Optional[float] = None):
        super().__init__(message, status_code=status_code, retryable=True, retry_after=retry_after)


class LLMTimeoutError(LLMError):
    """Retryable network or request timeout error."""

    def __init__(self, message: str):
        super().__init__(message, status_code=408, retryable=True)


def extract_status_code(exc: Exception) -> Optional[int]:
    """Attempt to extract an HTTP status code from various exception types."""
    # Direct status_code or code attribute
    for attr in ("code", "status_code", "status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.isdigit():
            return int(val)

    # Check response attribute (e.g., httpx.HTTPStatusError)
    response = getattr(exc, "response", None)
    if response is not None and hasattr(response, "status_code"):
        return getattr(response, "status_code", None)

    # Check error message strings for known status codes
    msg = str(exc)
    for code in [400, 401, 403, 404, 429, 500, 502, 503, 504]:
        if f" {code} " in f" {msg} ":
            return code

    return None


def extract_retry_after(exc: Exception) -> Optional[float]:
    """Extract Retry-After duration in seconds from response headers or error messages."""
    # Check response headers (httpx or groq/gemini response)
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers and hasattr(headers, "get"):
            header_val = headers.get("retry-after") or headers.get("Retry-After")
            if header_val:
                try:
                    return float(header_val)
                except ValueError:
                    pass

    # Check direct retry_after attribute
    val = getattr(exc, "retry_after", None)
    if isinstance(val, (int, float)):
        return float(val)

    # Check error message patterns (e.g., "try again in 2.5s", "retry after 4s", "retry in 10s")
    msg = str(exc)
    match = re.search(
        r"(?:try again in|retry[- ]after|retry in)\s*([0-9]+(?:\.[0-9]+)?)\s*(s|sec|seconds?|m|min|minutes?|h|hr|hours?)?",
        msg,
        re.IGNORECASE,
    )
    if match:
        duration = float(match.group(1))
        unit = (match.group(2) or "s").lower()
        if unit.startswith("m"):
            duration *= 60.0
        elif unit.startswith("h"):
            duration *= 3600.0
        return duration

    return None


def classify_error(exc: Exception) -> Tuple[bool, LLMError]:
    """Classify an exception into specific LLM error categories and determine retryability.

    Crucially distinguishes transient per-minute rate limits from non-retryable daily quota exhaustion.
    """
    # Check if already classified
    if isinstance(exc, LLMError):
        return exc.retryable, exc

    # Check for timeout exceptions
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True, LLMTimeoutError(f"Request timed out: {exc}")

    exc_name = exc.__class__.__name__.lower()
    if "timeout" in exc_name:
        return True, LLMTimeoutError(f"Network timeout: {exc}")

    # Transport failures (e.g. groq.APIConnectionError, httpx.ConnectError) never reached the model.
    lower_text = str(exc).lower()
    if (
        "connection" in exc_name
        or exc_name in ("connecterror", "readerror", "writeerror", "remoteprotocolerror", "networkerror")
        or "connection error" in lower_text
        or "connection reset" in lower_text
        or "connection aborted" in lower_text
    ):
        return True, LLMServerError(f"Network connection error (retryable): {exc}", status_code=503)

    # A single request above the provider's per-minute cap can never succeed, so retrying wastes budget.
    if extract_status_code(exc) == 413 or "request too large" in lower_text:
        return False, LLMClientError(f"Request too large for provider limits (413): {exc}", status_code=413)

    code = extract_status_code(exc)
    retry_after = extract_retry_after(exc)
    msg = str(exc)
    lower_msg = msg.lower()

    # 1. Check for daily/monthly quota exhaustion (NON-RETRYABLE)
    # If retry-after exceeds 60 seconds (e.g. hours or days), treat as quota exhaustion
    if retry_after is not None and retry_after > 60.0:
        return False, LLMQuotaExhaustedError(
            f"Daily quota limit exhausted (retry-after: {retry_after:.1f}s): {msg}",
            status_code=429,
        )

    if any(kw in lower_msg for kw in PER_MINUTE_LIMIT_KEYWORDS):
        return True, LLMRateLimitError(
            f"Transient rate limit (429): {msg}",
            status_code=429,
            retry_after=retry_after,
        )

    for kw in QUOTA_EXHAUSTION_KEYWORDS:
        if kw in lower_msg:
            return False, LLMQuotaExhaustedError(
                f"Daily/account quota limit exhausted: {msg}",
                status_code=code or 429,
            )

    # 2. Check for authentication or permission errors (NON-RETRYABLE)
    if code in (401, 403) or "unauthorized" in lower_msg or "invalid api key" in lower_msg or "authentication" in lower_msg:
        return False, LLMAuthError(f"Authentication/Permission error ({code or 401}): {msg}", status_code=code or 401)

    # 3. Check for model not found errors (NON-RETRYABLE)
    if code == 404 or "model_not_found" in lower_msg or "model not found" in lower_msg or "does not exist" in lower_msg:
        return False, LLMModelNotFoundError(f"Model not found or unavailable (404): {msg}", status_code=404)

    # 4. Check for bad requests (NON-RETRYABLE)
    if code == 400:
        return False, LLMClientError(f"Invalid request (400): {msg}", status_code=400)

    # 5. Check for transient rate limit (RETRYABLE)
    if code == 429 or "rate limit" in lower_msg or "rate_limit" in lower_msg or "too many requests" in lower_msg or "tpm" in lower_msg or "rpm" in lower_msg or "resource exhausted" in lower_msg:
        return True, LLMRateLimitError(
            f"Transient rate limit (429): {msg}",
            status_code=429,
            retry_after=retry_after,
        )

    # 6. Check for server failures (RETRYABLE)
    if code in (500, 502, 503, 504) or "unavailable" in lower_msg or "internal server" in lower_msg or "bad gateway" in lower_msg:
        return True, LLMServerError(f"Server failure ({code or 500}): {msg}", status_code=code or 500, retry_after=retry_after)

    # Default: non-retryable generic error
    return False, LLMError(f"Unclassified LLM error: {msg}", status_code=code, retryable=False)


async def execute_with_retry(
    coro_fn: Callable[[], Coroutine[Any, Any, T]],
    max_retries: int = MAX_RETRIES,
    initial_delay: float = INITIAL_RETRY_DELAY,
    max_delay: float = MAX_RETRY_DELAY,
    timeout_seconds: Optional[float] = None,
    on_attempt_failure: Optional[Callable[[int, Exception], None]] = None,
) -> T:
    """Execute an async operation with exponential backoff, random jitter, and error classification.

    Respects Retry-After metadata when available, and fails immediately on non-retryable errors
    (such as quota exhaustion, authentication failures, and invalid models).
    """
    attempt = 0

    while True:
        try:
            if timeout_seconds and timeout_seconds > 0:
                return await asyncio.wait_for(coro_fn(), timeout=timeout_seconds)
            else:
                return await coro_fn()

        except Exception as exc:
            attempt += 1
            is_retryable, classified_err = classify_error(exc)
            if on_attempt_failure is not None:
                try:
                    on_attempt_failure(attempt, classified_err)
                except Exception:
                    # Telemetry callbacks must never change provider retry behavior.
                    pass

            if not is_retryable or attempt > max_retries:
                logger.error(
                    f"Execution failed (attempt {attempt}/{max_retries + 1}, "
                    f"retryable={is_retryable}): {classified_err}"
                )
                raise classified_err from exc

            # Determine delay: respect Retry-After if available, bounded by max_delay
            retry_after = getattr(classified_err, "retry_after", None)
            if retry_after is not None and retry_after > 0:
                delay = min(max_delay, float(retry_after))
            else:
                base_backoff = min(max_delay, initial_delay * (2 ** (attempt - 1)))
                jitter = random.uniform(0, 0.5 * base_backoff)
                delay = base_backoff + jitter

            logger.warning(
                f"Retryable error on attempt {attempt}/{max_retries + 1} "
                f"({classified_err.__class__.__name__}: {classified_err}). "
                f"Backing off for {delay:.2f}s..."
            )
            await asyncio.sleep(delay)
