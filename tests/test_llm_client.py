import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest
from pydantic import BaseModel

from benchmark.baseline import SingleAgentBaseline
from core.llm_client import (
    BaseLLMClient,
    GeminiClient,
    GroqClient,
    create_llm_client,
)
from utils.retry import (
    LLMAuthError,
    LLMBudgetExceededError,
    LLMClientError,
    LLMError,
    LLMModelNotFoundError,
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    LLMServerError,
    LLMTokenBudgetExceededError,
    LLMTimeoutError,
)
from utils.token_tracker import TokenTracker


def test_groq_per_minute_limit_with_billing_link_is_retryable():
    from utils.retry import classify_error

    err = RuntimeError(
        "Error code: 429 - Rate limit reached for model `m` on tokens per minute (TPM): Limit 8000, "
        "Used 5581, Requested 2613. Please try again in 1.455s. Need more tokens? Upgrade to Dev Tier "
        "today at https://console.groq.com/settings/billing"
    )
    retryable, classified = classify_error(err)
    assert retryable is True
    assert isinstance(classified, LLMRateLimitError)
    assert classified.retry_after == pytest.approx(1.455)


class APIConnectionError(Exception):
    pass


@pytest.mark.parametrize(
    "error",
    [APIConnectionError("Connection error."), RuntimeError("Connection reset by peer")],
)
def test_network_connection_errors_are_retryable(error):
    from utils.retry import classify_error

    retryable, classified = classify_error(error)
    assert retryable is True
    assert isinstance(classified, LLMServerError)


def test_request_too_large_is_not_retried():
    from utils.retry import classify_error

    err = RuntimeError(
        "Error code: 413 - Request too large for model `m` on tokens per minute (TPM): Limit 8000, "
        "Requested 8151, please reduce your message size and try again."
    )
    retryable, classified = classify_error(err)
    assert retryable is False
    assert isinstance(classified, LLMClientError)
    assert classified.status_code == 413


@pytest.mark.parametrize(
    "tpm, input_tokens, requested, expected",
    [(0, 5000, 4096, 4096), (8000, 4000, 4096, 3200), (8000, 1000, 2048, 2048), (8000, 7900, 4096, 256), (8000, 7900, 48, 48)],
)
def test_groq_output_limit_fits_provider_tpm(monkeypatch, tpm, input_tokens, requested, expected):
    import core.providers.groq_client as groq_module

    monkeypatch.setattr(groq_module, "PROVIDER_TOKENS_PER_MINUTE", tpm)
    assert GroqClient._fit_output_limit(input_tokens, requested) == expected


def test_daily_token_limit_remains_quota_exhaustion():
    from utils.retry import classify_error

    err = RuntimeError("Error code: 429 - Rate limit reached on tokens per day (TPD): Limit 500000")
    retryable, classified = classify_error(err)
    assert retryable is False
    assert isinstance(classified, LLMQuotaExhaustedError)


def test_reviewer_instruction_follows_verification_mode(monkeypatch):
    from agents.reviewer_agent import (
        GENERAL_KNOWLEDGE_ADDENDUM,
        REVIEWER_SYSTEM_INSTRUCTION,
        reviewer_system_instruction,
    )
    from config import settings

    monkeypatch.setattr(settings, "VERIFICATION_MODE", "strict")
    assert reviewer_system_instruction() == REVIEWER_SYSTEM_INSTRUCTION
    monkeypatch.setattr(settings, "VERIFICATION_MODE", "general")
    assert reviewer_system_instruction().endswith(GENERAL_KNOWLEDGE_ADDENDUM)


def test_console_text_replaces_typographic_unicode():
    import main

    assert main._console_text("needs\u2011revision \u2014 0.0\u202f% \u201cok\u201d \u2265 1\u2026") == (
        'needs-revision - 0.0 % "ok" >= 1...'
    )
    assert main._console_text(None) == ""


def test_console_prints_only_direct_answer_unless_full():
    import main

    report = (
        "## Direct Answer\n1. Catches regressions early.\n2. Enables safe refactoring.\n\n---\n\n"
        "## Evidence & Verification Audit\n- Support rate: 100%\n"
    )
    assert main._answer_for_console(report, full=False) == (
        "1. Catches regressions early.\n2. Enables safe refactoring."
    )
    assert "Evidence & Verification Audit" in main._answer_for_console(report, full=True)
    failure_report = "# Execution Failure Report\n\n### Status: FAILED"
    assert main._answer_for_console(failure_report, full=False) == failure_report


def test_benchmark_refuses_general_verification_mode(monkeypatch):
    import main
    from config import settings

    monkeypatch.setattr(main, "validate_settings", lambda: True)
    monkeypatch.setattr(settings, "VERIFICATION_MODE", "general")
    with pytest.raises(SystemExit) as exit_info:
        asyncio.run(main.run_benchmark_cli())
    assert exit_info.value.code == 2


class SampleOutput(BaseModel):
    title: str
    score: int


def _create_mock_gemini_response(
    text: str,
    prompt_tokens: int = 10,
    candidates_tokens: int = 20,
    has_usage: bool = True,
):
    response = MagicMock()
    response.text = text
    response.parsed = None
    if has_usage:
        usage = MagicMock()
        usage.prompt_token_count = prompt_tokens
        usage.candidates_token_count = candidates_tokens
        usage.total_token_count = prompt_tokens + candidates_tokens
        usage.cached_content_token_count = 0
        response.usage_metadata = usage
    else:
        response.usage_metadata = None
    return response


def _create_mock_groq_response(
    content: str,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    has_usage: bool = True,
):
    response = MagicMock()
    choice = MagicMock()
    msg = MagicMock()
    msg.content = content
    choice.message = msg
    response.choices = [choice]
    if has_usage:
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        usage.total_tokens = prompt_tokens + completion_tokens
        response.usage = usage
    else:
        response.usage = None
    return response


# =====================================================================
# Provider Factory & Configuration Tests
# =====================================================================


def test_provider_factory_selection():
    """Verify create_llm_client creates the appropriate client based on provider."""
    mock_groq = MagicMock()
    mock_gemini = MagicMock()

    client_groq = create_llm_client(provider="groq", api_key="test_gsk", client=mock_groq)
    assert isinstance(client_groq, GroqClient)
    assert client_groq.default_model == "openai/gpt-oss-20b"

    client_gemini = create_llm_client(provider="gemini", api_key="test_gem", client=mock_gemini)
    assert isinstance(client_gemini, GeminiClient)
    assert client_gemini.default_model == "gemini-3.8-flash"

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client(provider="unsupported_vendor")


@pytest.mark.asyncio
async def test_groq_missing_api_key_raises(monkeypatch):
    """Verify GroqClient raises LLMClientError when API key is missing."""
    monkeypatch.setattr("config.settings.GROQ_API_KEY", None)
    client = GroqClient(api_key="", client=None)
    with pytest.raises(LLMClientError, match="GROQ_API_KEY is not configured"):
        await client.generate_text("Hello")


@pytest.mark.asyncio
async def test_gemini_missing_api_key_raises(monkeypatch):
    """Verify GeminiClient raises LLMClientError when API key is missing."""
    monkeypatch.setattr("config.settings.GEMINI_API_KEY", None)
    client = GeminiClient(api_key="", client=None)
    with pytest.raises(LLMClientError, match="GEMINI_API_KEY is not configured"):
        await client.generate_text("Hello")


# =====================================================================
# Groq Provider Tests
# =====================================================================


@pytest.mark.asyncio
async def test_groq_generate_text_success():
    """Verify GroqClient.generate_text returns response and tracks token usage."""
    tracker = TokenTracker(name="groq_test")
    mock_sdk = MagicMock()
    mock_sdk.chat.completions.create = AsyncMock(
        return_value=_create_mock_groq_response(
            "Hello from Groq", prompt_tokens=12, completion_tokens=18
        )
    )

    client = GroqClient(api_key="mock_key", client=mock_sdk, token_tracker=tracker)
    result = await client.generate_text("Hi")

    assert result == "Hello from Groq"
    summary = await tracker.get_summary()
    assert summary["prompt_tokens"] == 12
    assert summary["candidate_tokens"] == 18
    assert summary["total_tokens"] == 30
    assert summary["successful_requests"] == 1


@pytest.mark.asyncio
async def test_groq_generate_text_missing_usage_safe():
    """Verify GroqClient handles responses gracefully when usage metadata is None."""
    tracker = TokenTracker(name="groq_no_usage")
    mock_sdk = MagicMock()
    mock_sdk.chat.completions.create = AsyncMock(
        return_value=_create_mock_groq_response("Safe response", has_usage=False)
    )

    client = GroqClient(api_key="mock_key", client=mock_sdk, token_tracker=tracker)
    result = await client.generate_text("Say hi")

    assert result == "Safe response"
    summary = await tracker.get_summary()
    assert summary["successful_requests"] == 1


@pytest.mark.asyncio
async def test_groq_generate_structured_success():
    """Verify GroqClient.generate_structured parses valid Pydantic models."""
    tracker = TokenTracker(name="groq_struct_test")
    mock_sdk = MagicMock()
    mock_sdk.chat.completions.create = AsyncMock(
        return_value=_create_mock_groq_response(
            '{"title": "Groq Architecture", "score": 98}', prompt_tokens=20, completion_tokens=15
        )
    )

    client = GroqClient(api_key="mock_key", client=mock_sdk, token_tracker=tracker)
    result = await client.generate_structured("Generate score", response_schema=SampleOutput)

    assert isinstance(result, SampleOutput)
    assert result.title == "Groq Architecture"
    assert result.score == 98


@pytest.mark.asyncio
async def test_groq_generate_structured_malformed_json_raises():
    """Verify GroqClient raises LLMClientError when model returns malformed JSON."""
    tracker = TokenTracker(name="groq_malformed")
    mock_sdk = MagicMock()
    mock_sdk.chat.completions.create = AsyncMock(
        return_value=_create_mock_groq_response("This is not JSON at all")
    )

    client = GroqClient(api_key="mock_key", client=mock_sdk, token_tracker=tracker)
    with pytest.raises(LLMClientError, match="Malformed JSON or schema validation"):
        await client.generate_structured("Generate score", response_schema=SampleOutput)


@pytest.mark.asyncio
async def test_groq_generate_structured_format_fallback():
    """Verify GroqClient falls back to standard completion if json_object format is unsupported."""
    mock_sdk = MagicMock()
    success_resp = _create_mock_groq_response('{"title": "Fallback", "score": 85}')
    mock_sdk.chat.completions.create = AsyncMock(
        side_effect=[Exception("Model does not support response_format json_object"), success_resp]
    )

    client = GroqClient(api_key="mock_key", client=mock_sdk)
    result = await client.generate_structured("Generate", response_schema=SampleOutput)

    assert isinstance(result, SampleOutput)
    assert result.title == "Fallback"
    assert result.score == 85
    assert mock_sdk.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_groq_quota_429_does_not_fallback_to_plain_text():
    """CRITICAL TEST: Groq HTTP 429 quota exhaustion must NOT trigger json format fallback and must raise LLMQuotaExhaustedError."""
    mock_sdk = MagicMock()
    # Groq rate limit on tokens per day (TPD)
    quota_err = Exception("Rate limit reached for model openai/gpt-oss-20b on tokens per day (TPD): Limit 500000 429")
    mock_sdk.chat.completions.create = AsyncMock(side_effect=quota_err)

    client = GroqClient(api_key="mock_key", client=mock_sdk)

    with pytest.raises(LLMQuotaExhaustedError):
        await client.generate_structured("Generate", response_schema=SampleOutput, max_retries=1)

    # Must NOT attempt a fallback plain-text call (call count must be exactly 1)
    assert mock_sdk.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_groq_auth_401_does_not_fallback_to_plain_text():
    """Verify Groq HTTP 401 Unauthorized does not trigger format fallback and raises LLMAuthError immediately."""
    mock_sdk = MagicMock()
    auth_err = Exception("401 Unauthorized: Invalid API Key")
    mock_sdk.chat.completions.create = AsyncMock(side_effect=auth_err)

    client = GroqClient(api_key="mock_key", client=mock_sdk)

    with pytest.raises(LLMAuthError):
        await client.generate_structured("Generate", response_schema=SampleOutput, max_retries=1)

    assert mock_sdk.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_groq_server_500_does_not_fallback_to_plain_text():
    """Verify Groq 500 Internal Server error does NOT trigger format fallback and retries properly."""
    mock_sdk = MagicMock()
    server_err = Exception("500 Internal Server Error: temporary backend failure")
    success_resp = _create_mock_groq_response('{"title": "Recovered", "score": 90}')
    mock_sdk.chat.completions.create = AsyncMock(side_effect=[server_err, success_resp])

    client = GroqClient(api_key="mock_key", client=mock_sdk)
    result = await client.generate_structured("Generate", response_schema=SampleOutput, max_retries=1)

    assert result.title == "Recovered"
    # Both calls should have passed response_format={"type": "json_object"}
    assert mock_sdk.chat.completions.create.call_count == 2
    for call in mock_sdk.chat.completions.create.call_args_list:
        assert call.kwargs.get("response_format") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_llm_call_budget_enforcement():
    """Verify that exceeding MAX_LLM_CALLS_PER_TASK raises LLMBudgetExceededError centrally."""
    tracker = TokenTracker(name="budget_test")
    # Simulate tracker having made 25 requests
    for _ in range(25):
        tracker.add_usage_sync(prompt_tokens=10, candidate_tokens=10)

    mock_sdk = MagicMock()
    client = GroqClient(api_key="mock_key", client=mock_sdk, token_tracker=tracker)

    with pytest.raises(LLMBudgetExceededError, match="Central LLM call budget exceeded"):
        await client.generate_text("Should fail budget check")

    with pytest.raises(LLMBudgetExceededError, match="Central LLM call budget exceeded"):
        await client.generate_structured("Should fail budget check", response_schema=SampleOutput)

    # API call must never have been initiated
    assert mock_sdk.chat.completions.create.call_count == 0


@pytest.mark.asyncio
async def test_concurrent_call_budget_safety():
    """Verify that under simultaneous requests from concurrent agents,

    in-flight reservations prevent exceeding MAX_LLM_CALLS_PER_TASK due to a race condition.
    """
    from config.settings import MAX_LLM_CALLS_PER_TASK

    tracker = TokenTracker(name="concurrent_race_test")
    # Simulate tracker having made (MAX_LLM_CALLS_PER_TASK - 3) calls
    pre_fill = max(0, MAX_LLM_CALLS_PER_TASK - 3)
    for _ in range(pre_fill):
        tracker.add_usage_sync(prompt_tokens=10, candidate_tokens=10)

    mock_sdk = MagicMock()

    async def slow_create(**kwargs):
        # Small delay to ensure all concurrent coroutines are in flight simultaneously
        await asyncio.sleep(0.05)
        return _create_mock_groq_response("OK response")

    mock_sdk.chat.completions.create = AsyncMock(side_effect=slow_create)
    client = GroqClient(api_key="mock_key", client=mock_sdk, token_tracker=tracker)

    # Launch 10 simultaneous calls when only 3 slots remain
    tasks = [client.generate_text(f"Prompt {i}") for i in range(10)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successes = [r for r in results if isinstance(r, str)]
    budget_errors = [r for r in results if isinstance(r, LLMBudgetExceededError)]

    # Exactly 3 calls must succeed, and 7 must be rejected by budget protection
    assert len(successes) == 3
    assert len(budget_errors) == 7

    # Total executed calls on mock SDK must be exactly 3
    assert mock_sdk.chat.completions.create.call_count == 3

    # Total requests recorded in tracker cannot exceed MAX_LLM_CALLS_PER_TASK
    summary = await tracker.get_summary()
    assert summary["successful_requests"] == MAX_LLM_CALLS_PER_TASK
    assert summary["in_flight_requests"] == 0


@pytest.mark.asyncio
async def test_token_reservations_are_atomic_under_concurrent_calls():
    tracker = TokenTracker(
        max_tokens_per_task=100,
        max_input_tokens_per_call=100,
        max_output_tokens_per_call=20,
        essential_token_reserves={"synthesis": 20},
    )
    entered = 0
    both_entered = asyncio.Event()
    release = asyncio.Event()

    async def reserved_call():
        nonlocal entered
        try:
            async with tracker.track_call(estimated_input_tokens=20, max_output_tokens=20):
                entered += 1
                if entered == 2:
                    both_entered.set()
                await release.wait()
                await tracker.add_usage(prompt_tokens=7, candidate_tokens=3, total_tokens=10)
                return "accepted"
        except LLMTokenBudgetExceededError:
            return "denied"

    async with tracker.task_budget():
        with tracker.operation("specialist"):
            tasks = [asyncio.create_task(reserved_call()) for _ in range(3)]
            await both_entered.wait()
            release.set()
            outcomes = await asyncio.gather(*tasks)
        summary = await tracker.get_summary()

    assert outcomes.count("accepted") == 2
    assert outcomes.count("denied") == 1
    assert summary["local_task_token_budget"]["accounted_tokens"] == 20
    assert summary["local_task_token_budget"]["reserved_tokens"] == 0
    assert summary["local_task_token_budget"]["provider_daily_quota_known"] is False


@pytest.mark.asyncio
async def test_token_reservation_reconciles_to_actual_usage():
    tracker = TokenTracker(
        max_tokens_per_task=100,
        max_input_tokens_per_call=100,
        max_output_tokens_per_call=20,
        essential_token_reserves={},
    )

    async with tracker.task_budget():
        with tracker.operation("specialist"):
            async with tracker.track_call(estimated_input_tokens=40, max_output_tokens=20):
                await tracker.add_usage(prompt_tokens=9, candidate_tokens=6, total_tokens=15)
        summary = await tracker.get_summary()

    assert summary["total_tokens"] == 15
    assert summary["local_task_token_budget"]["accounted_tokens"] == 15
    assert summary["local_task_token_budget"]["remaining_tokens"] == 85


@pytest.mark.asyncio
async def test_token_budget_exhaustion_prevents_provider_dispatch():
    tracker = TokenTracker(
        max_tokens_per_task=10,
        max_input_tokens_per_call=100,
        max_output_tokens_per_call=2048,
        essential_token_reserves={},
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_create_mock_gemini_response("Should not be dispatched")
    )
    client = GeminiClient(api_key="mock_key", client=mock_client, token_tracker=tracker)

    async with tracker.task_budget():
        with tracker.operation("specialist"):
            with pytest.raises(LLMTokenBudgetExceededError, match="Local task token budget"):
                await client.generate_text("A small prompt", max_tokens=2048)
        summary = await tracker.get_summary()

    assert mock_client.aio.models.generate_content.call_count == 0
    assert summary["failed_requests"] == 0


@pytest.mark.asyncio
async def test_nonessential_calls_cannot_consume_synthesis_reserve():
    tracker = TokenTracker(
        max_tokens_per_task=100,
        max_input_tokens_per_call=100,
        max_output_tokens_per_call=20,
        essential_token_reserves={
            "claim_extraction": 20,
            "verification": 20,
            "synthesis": 30,
        },
    )

    async with tracker.task_budget():
        with tracker.operation("specialist"):
            with pytest.raises(LLMTokenBudgetExceededError):
                async with tracker.track_call(estimated_input_tokens=40, max_output_tokens=20):
                    pytest.fail("Nonessential request should not enter the provider call.")
        with tracker.operation("synthesis"):
            async with tracker.track_call(estimated_input_tokens=10, max_output_tokens=20):
                await tracker.add_usage(prompt_tokens=10, candidate_tokens=10, total_tokens=20)
        summary = await tracker.get_summary()

    assert summary["local_task_token_budget"]["accounted_tokens"] == 20


@pytest.mark.asyncio
async def test_per_call_input_limit_prevents_provider_dispatch():
    tracker = TokenTracker(
        max_tokens_per_task=100,
        max_input_tokens_per_call=5,
        max_output_tokens_per_call=20,
        essential_token_reserves={},
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock()
    client = GeminiClient(api_key="mock_key", client=mock_client, token_tracker=tracker)

    with pytest.raises(LLMTokenBudgetExceededError, match="MAX_INPUT_TOKENS_PER_CALL"):
        await client.generate_text("x" * 100)

    mock_client.aio.models.generate_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_usage_metadata_consumes_the_estimated_reservation():
    tracker = TokenTracker(
        max_tokens_per_task=100,
        max_input_tokens_per_call=100,
        max_output_tokens_per_call=20,
        essential_token_reserves={},
    )

    async with tracker.task_budget():
        with tracker.operation("specialist"):
            async with tracker.track_call(estimated_input_tokens=15, max_output_tokens=5):
                await tracker.add_usage()
        summary = await tracker.get_summary()

    assert summary["local_task_token_budget"]["accounted_tokens"] == 20
    assert summary["local_task_token_budget"]["reserved_tokens"] == 0
    assert summary["local_task_token_budget"]["estimated_tokens_without_usage"] == 20
    assert summary["local_task_token_budget"]["usage_diagnostics"]


@pytest.mark.asyncio
async def test_gemini_response_without_usage_metadata_preserves_local_estimate():
    tracker = TokenTracker(
        max_tokens_per_task=5000,
        max_input_tokens_per_call=1000,
        max_output_tokens_per_call=2048,
        essential_token_reserves={},
    )
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_create_mock_gemini_response("Response without usage", has_usage=False)
    )
    client = GeminiClient(api_key="mock_key", client=mock_client, token_tracker=tracker)

    async with tracker.task_budget():
        with tracker.operation("specialist"):
            assert await client.generate_text("Short prompt", max_tokens=2048) == "Response without usage"
        summary = await tracker.get_summary()

    budget = summary["local_task_token_budget"]
    assert budget["accounted_tokens"] > 0
    assert budget["estimated_tokens_without_usage"] == budget["accounted_tokens"]
    assert "usage metadata was missing or empty" in budget["usage_diagnostics"][0]


@pytest.mark.asyncio
async def test_failed_dispatched_call_without_usage_metadata_keeps_estimate():
    tracker = TokenTracker(
        max_tokens_per_task=100,
        max_input_tokens_per_call=100,
        max_output_tokens_per_call=20,
        essential_token_reserves={},
    )

    async with tracker.task_budget():
        with tracker.operation("specialist"):
            with pytest.raises(RuntimeError, match="connection lost"):
                async with tracker.track_call(estimated_input_tokens=15, max_output_tokens=5):
                    raise RuntimeError("connection lost after dispatch")
        summary = await tracker.get_summary()

    budget = summary["local_task_token_budget"]
    assert budget["accounted_tokens"] == 20
    assert budget["estimated_tokens_without_usage"] == 20
    assert "conservatively charged its 20-token estimate" in budget["usage_diagnostics"][0]


@pytest.mark.asyncio
async def test_actual_usage_over_local_budget_is_diagnosed_and_accounted():
    tracker = TokenTracker(
        max_tokens_per_task=50,
        max_input_tokens_per_call=100,
        max_output_tokens_per_call=20,
        essential_token_reserves={},
    )

    async with tracker.task_budget():
        with tracker.operation("specialist"):
            with pytest.raises(LLMTokenBudgetExceededError, match="Actual provider usage"):
                async with tracker.track_call(estimated_input_tokens=5, max_output_tokens=5):
                    await tracker.add_usage(prompt_tokens=40, candidate_tokens=30, total_tokens=70)
        summary = await tracker.get_summary()

    budget = summary["local_task_token_budget"]
    assert budget["accounted_tokens"] == 70
    assert budget["budget_violations"]
    assert "Provider daily quota is not observable" in budget["budget_violations"][0]
    assert summary["total_tokens"] == 70


@pytest.mark.asyncio
async def test_call_budget_atomic_counting_no_double_counting():
    """Verify that in-flight, completed, and failed requests transition cleanly without double-counting."""
    tracker = TokenTracker(name="no_double_counting")
    mock_sdk = MagicMock()

    in_flight_observed = []

    async def observable_create(**kwargs):
        # Sample in-flight requests while the call is executing
        summary = await tracker.get_summary()
        in_flight_observed.append(summary["in_flight_requests"])
        return _create_mock_groq_response("OK")

    mock_sdk.chat.completions.create = AsyncMock(side_effect=observable_create)
    client = GroqClient(api_key="mock_key", client=mock_sdk, token_tracker=tracker)

    # 1. Execute a successful call
    res = await client.generate_text("Test 1")
    assert res == "OK"

    # In flight was 1 during execution
    assert 1 in in_flight_observed

    # After execution: exactly 1 successful request, 0 in-flight, 0 failed
    summary_after = await tracker.get_summary()
    assert summary_after["successful_requests"] == 1
    assert summary_after["in_flight_requests"] == 0
    assert summary_after["failed_requests"] == 0
    assert summary_after["total_requests"] == 1

    # 2. Execute a failing call (exhausting retries)
    mock_sdk.chat.completions.create = AsyncMock(side_effect=RuntimeError("Permanent API crash"))
    with pytest.raises((LLMError, RuntimeError)):
        await client.generate_text("Test fail", max_retries=1)

    summary_after_fail = await tracker.get_summary()
    assert summary_after_fail["successful_requests"] == 1
    assert summary_after_fail["in_flight_requests"] == 0
    assert summary_after_fail["failed_requests"] == 1
    assert summary_after_fail["total_requests"] == 2


# =====================================================================
# Gemini Provider Tests (Preserved)
# =====================================================================


@pytest.mark.asyncio
async def test_gemini_generate_text_success():
    """Verify GeminiClient.generate_text calls client and updates token tracker."""
    tracker = TokenTracker(name="test")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_create_mock_gemini_response("Hello from Gemini", prompt_tokens=15, candidates_tokens=25)
    )

    client = GeminiClient(api_key="mock_key", client=mock_client, token_tracker=tracker)
    result = await client.generate_text("Hi")

    assert result == "Hello from Gemini"
    summary = await tracker.get_summary()
    assert summary["prompt_tokens"] == 15
    assert summary["candidate_tokens"] == 25
    assert summary["total_tokens"] == 40
    assert summary["successful_requests"] == 1


@pytest.mark.asyncio
async def test_gemini_generate_structured_success():
    """Verify GeminiClient.generate_structured parses Pydantic model correctly."""
    tracker = TokenTracker(name="test")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_create_mock_gemini_response('{"title": "Test Plan", "score": 95}')
    )

    client = GeminiClient(api_key="mock_key", client=mock_client, token_tracker=tracker)
    result = await client.generate_structured("Generate plan", response_schema=SampleOutput)

    assert isinstance(result, SampleOutput)
    assert result.title == "Test Plan"
    assert result.score == 95


@pytest.mark.asyncio
async def test_gemini_missing_usage_safe():
    """Verify GeminiClient gracefully handles missing usage_metadata."""
    tracker = TokenTracker(name="test")
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_create_mock_gemini_response("No usage text", has_usage=False)
    )

    client = GeminiClient(api_key="mock_key", client=mock_client, token_tracker=tracker)
    result = await client.generate_text("Test")

    assert result == "No usage text"
    summary = await tracker.get_summary()
    assert summary["successful_requests"] == 1


# =====================================================================
# Retry & Error Classification Tests
# =====================================================================


@pytest.mark.asyncio
async def test_non_retryable_auth_error():
    """Verify 401 Unauthorized is classified as non-retryable and fails immediately."""
    mock_client = MagicMock()

    class MockAuthError(Exception):
        code = 401
        message = "Unauthorized: Invalid API key"

    mock_client.aio.models.generate_content = AsyncMock(side_effect=MockAuthError("401 Invalid Key"))
    client = GeminiClient(api_key="bad_key", client=mock_client)

    with pytest.raises(LLMAuthError):
        await client.generate_text("test", max_retries=3)

    assert mock_client.aio.models.generate_content.call_count == 1


@pytest.mark.asyncio
async def test_non_retryable_model_not_found():
    """Verify 404 Model Not Found is non-retryable and fails immediately."""
    mock_client = MagicMock()

    class Mock404Error(Exception):
        code = 404
        message = "Model gemini-fake not found"

    mock_client.aio.models.generate_content = AsyncMock(side_effect=Mock404Error("404 Model not found"))
    client = GeminiClient(api_key="mock_key", client=mock_client)

    with pytest.raises(LLMModelNotFoundError):
        await client.generate_text("test", max_retries=3)

    assert mock_client.aio.models.generate_content.call_count == 1


@pytest.mark.asyncio
async def test_non_retryable_daily_quota_exhaustion():
    """Verify daily quota exhaustion does not retry and raises LLMQuotaExhaustedError immediately."""
    mock_client = MagicMock()
    quota_exc = Exception("429 ResourceExhausted: You have exceeded your daily quota for requests per day (RPD).")
    mock_client.aio.models.generate_content = AsyncMock(side_effect=quota_exc)
    client = GeminiClient(api_key="mock_key", client=mock_client)

    with pytest.raises(LLMQuotaExhaustedError):
        await client.generate_text("test", max_retries=3)

    assert mock_client.aio.models.generate_content.call_count == 1


@pytest.mark.asyncio
async def test_non_retryable_excessive_retry_after():
    """Verify Retry-After exceeding 60s (e.g., 18h daily quota) is treated as quota exhaustion and not retried."""
    mock_client = MagicMock()

    class MockHeaderExc(Exception):
        code = 429
        retry_after = 64800.0  # 18 hours

    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=MockHeaderExc("429 Too Many Requests: retry after 64800s")
    )
    client = GeminiClient(api_key="mock_key", client=mock_client)

    with pytest.raises(LLMQuotaExhaustedError):
        await client.generate_text("test", max_retries=3)

    assert mock_client.aio.models.generate_content.call_count == 1


@pytest.mark.asyncio
async def test_retryable_rate_limit_recovers():
    """Verify transient per-minute rate limit (retry_after <= 60s) retries and recovers."""
    mock_client = MagicMock()

    class MockRateLimit(Exception):
        code = 429
        retry_after = 0.05

    success_resp = _create_mock_gemini_response("Recovered after transient rate limit")
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=[MockRateLimit("429 RPM limit reached, retry in 0.05s"), success_resp]
    )

    client = GeminiClient(api_key="mock_key", client=mock_client)
    result = await client.generate_text("test", max_retries=2)

    assert result == "Recovered after transient rate limit"
    assert mock_client.aio.models.generate_content.call_count == 2


@pytest.mark.asyncio
async def test_retryable_server_error_recovers():
    """Verify temporary 503 server failure retries and recovers."""
    mock_client = MagicMock()

    class MockServerError(Exception):
        code = 503
        message = "Service temporarily unavailable"

    success_resp = _create_mock_gemini_response("Recovered from server error")
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=[MockServerError("503 Backend failure"), success_resp]
    )

    client = GeminiClient(api_key="mock_key", client=mock_client)
    result = await client.generate_text("test", max_retries=2)

    assert result == "Recovered from server error"
    assert mock_client.aio.models.generate_content.call_count == 2


@pytest.mark.asyncio
async def test_single_agent_baseline_run():
    """Verify SingleAgentBaseline runs and records output and latency."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_create_mock_gemini_response("Architecture baseline answer", 10, 20)
    )
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)
    baseline = SingleAgentBaseline(llm_client=gemini_client)

    result = await baseline.run("Design an architecture")

    assert result.success is True
    assert result.answer == "Architecture baseline answer"
    assert result.duration_seconds >= 0.0
    assert result.model is not None
