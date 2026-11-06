import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from core.providers.gemini_client import GeminiClient
from core.providers.groq_client import GroqClient
from benchmark.baseline import SingleAgentBaseline
from models.provider_call import AccountingContext, ProviderCallStatus, ProviderUsageSource
from models.schemas import TaskStatus
from models.task_models import SubTask, TaskPlan, TaskResult
from utils.retry import LLMError, LLMTimeoutError, LLMTokenBudgetExceededError
from utils.token_tracker import ProviderCallRecorder, TokenTracker


class _StructuredAnswer(BaseModel):
    value: str


def _response(
    text="mock response",
    *,
    prompt_tokens=None,
    candidate_tokens=None,
    total_tokens=None,
    model_version=None,
    model=None,
):
    usage = None
    if any(value is not None for value in (prompt_tokens, candidate_tokens, total_tokens)):
        usage = SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=candidate_tokens,
            total_token_count=total_tokens,
            cached_content_token_count=None,
        )
    return SimpleNamespace(
        text=text,
        parsed=None,
        usage_metadata=usage,
        model_version=model_version,
        model=model,
    )


def _mock_gemini_client(side_effect=None, return_value=None):
    sdk_client = MagicMock()
    sdk_client.aio.models.generate_content = AsyncMock(
        side_effect=side_effect,
        return_value=return_value or _response(),
    )
    tracker = TokenTracker(name="telemetry_test", essential_token_reserves={})
    client = GeminiClient(
        api_key="mock-secret-never-record",
        default_model="mock-model",
        token_tracker=tracker,
        client=sdk_client,
    )
    return client, tracker, sdk_client


async def _run_scoped_text_call(
    client,
    tracker,
    task_id="trial-task",
    prompt="Do not persist this sensitive prompt",
    **kwargs,
):
    async with tracker.accounting_session(
        AccountingContext(
            experiment_id="experiment-1",
            task_block_id="block-1",
            trial_id=task_id,
            treatment_id="MOCK_ARM",
            execution_id=f"exec-{task_id}",
            scope_kind="trial_execution",
        )
    ) as scope:
        async with tracker.task_budget(essential_reserves={}):
            with tracker.operation("synthesis"):
                with tracker.call_context(
                    execution_id=f"exec-{task_id}",
                    agent_id="agent-synth",
                    agent_role="Final Synthesizer",
                ):
                    answer = await client.generate_text(
                        prompt=prompt,
                        temperature=0.73,
                        max_tokens=61,
                        **kwargs,
                    )
        return answer, scope.snapshot()


@pytest.mark.asyncio
async def test_successful_provider_call_records_effective_config_and_actual_usage(monkeypatch):
    monkeypatch.setattr("core.providers.base.MODEL_VERSION", "mock-v7")
    client, tracker, sdk = _mock_gemini_client(
        return_value=_response("safe answer", prompt_tokens=13, candidate_tokens=7, total_tokens=20)
    )

    answer, snapshot = await _run_scoped_text_call(client, tracker)

    assert answer == "safe answer"
    assert len(snapshot.provider_calls) == 1
    record = snapshot.provider_calls[0]
    assert record.call_id
    assert record.status == ProviderCallStatus.SUCCEEDED
    assert record.attempt_number == 1
    assert record.provider == "gemini"
    assert record.model == "mock-model"
    assert record.model_version is None
    assert record.operator_model_version == "mock-v7"
    assert record.model_version_source == "unavailable"
    assert record.model_identifier_policy == "provider_may_resolve_mutable_alias"
    assert record.timeout_seconds is None
    assert record.timeout_source == "provider_sdk_default"
    assert record.temperature == 0.73
    assert record.output_token_limit == 61
    assert record.prompt_tokens == 13
    assert record.candidate_tokens == 7
    assert record.total_tokens == 20
    assert record.usage_source == ProviderUsageSource.PROVIDER_REPORTED
    assert record.execution_id == "exec-trial-task"
    assert record.trial_id == "trial-task"
    assert record.stage == "synthesis"
    assert record.agent_id == "agent-synth"
    assert record.agent_role == "Final Synthesizer"
    assert sdk.aio.models.generate_content.await_args.kwargs["config"].temperature == 0.73
    assert sdk.aio.models.generate_content.await_args.kwargs["config"].max_output_tokens == 61


@pytest.mark.asyncio
async def test_provider_response_revision_and_response_model_are_captured():
    client, tracker, _ = _mock_gemini_client(
        return_value=_response("safe answer", model_version="gemini-revision-7", model="mock-model-7")
    )

    _, snapshot = await _run_scoped_text_call(client, tracker)

    record = snapshot.provider_calls[0]
    assert record.model_version == "gemini-revision-7"
    assert record.model_version_source == "provider_response"
    assert record.response_model == "mock-model-7"


@pytest.mark.asyncio
async def test_explicit_timeout_is_recorded_at_provider_call_boundary(monkeypatch):
    monkeypatch.setattr("core.providers.gemini_client.PROVIDER_REQUEST_TIMEOUT_SECONDS", 8.5)
    client, tracker, _ = _mock_gemini_client(return_value=_response("bounded response"))

    _, snapshot = await _run_scoped_text_call(client, tracker)

    record = snapshot.provider_calls[0]
    assert record.timeout_seconds == 8.5
    assert record.timeout_source == "explicit_request"


@pytest.mark.asyncio
async def test_provider_error_is_recorded_without_secret_or_request_content():
    client, tracker, _ = _mock_gemini_client(side_effect=RuntimeError("Authorization: Bearer mock-secret"))

    async with tracker.accounting_session(AccountingContext(trial_id="failed-trial")) as scope:
        async with tracker.task_budget(essential_reserves={}):
            with pytest.raises(LLMError):
                await client.generate_text(
                    prompt="Sensitive task content",
                    max_retries=0,
                )

    record = scope.provider_calls[0]
    serialized = json.dumps(record.model_dump(mode="json"))
    assert record.status == ProviderCallStatus.FAILED
    assert record.error_category == "LLMError"
    assert "mock-secret" not in serialized
    assert "Sensitive task content" not in serialized
    assert "Authorization" not in serialized


@pytest.mark.asyncio
async def test_provider_daily_quota_is_classified_separately_from_local_budget():
    class MockQuotaError(Exception):
        code = 429

    client, tracker, _ = _mock_gemini_client(
        side_effect=MockQuotaError("tokens per day quota exhausted")
    )
    async with tracker.accounting_session(AccountingContext(trial_id="quota-trial")) as scope:
        async with tracker.task_budget(essential_reserves={}):
            with pytest.raises(Exception):
                await client.generate_text("quota request", max_retries=0)

    assert len(scope.provider_calls) == 1
    assert scope.provider_calls[0].error_category == "LLMQuotaExhaustedError"
    assert scope.provider_calls[0].status == ProviderCallStatus.FAILED


@pytest.mark.asyncio
async def test_local_trial_budget_rejection_is_not_a_provider_quota_call():
    sdk_client = MagicMock()
    sdk_client.aio.models.generate_content = AsyncMock()
    tracker = TokenTracker(
        name="local_budget_test",
        max_input_tokens_per_call=2,
        essential_token_reserves={},
    )
    client = GeminiClient(
        api_key="mock-key",
        default_model="mock-model",
        token_tracker=tracker,
        client=sdk_client,
    )

    async with tracker.accounting_session(AccountingContext(trial_id="local-budget")) as scope:
        async with tracker.task_budget(essential_reserves={}):
            with pytest.raises(LLMTokenBudgetExceededError):
                await client.generate_text("this request exceeds the local input limit", max_retries=0)

    sdk_client.aio.models.generate_content.assert_not_awaited()
    assert scope.provider_calls == []


@pytest.mark.asyncio
async def test_timeout_is_recorded_as_timed_out(monkeypatch):
    async def hang(**kwargs):
        await asyncio.Event().wait()

    client, tracker, _ = _mock_gemini_client(side_effect=hang)
    monkeypatch.setattr("core.providers.gemini_client.INITIAL_RETRY_DELAY", 0)

    async with tracker.accounting_session(AccountingContext(trial_id="timeout-trial")) as scope:
        async with tracker.task_budget(essential_reserves={}):
            with pytest.raises(LLMTimeoutError):
                await client.generate_text(
                    prompt="timeout",
                    timeout_seconds=0.01,
                    max_retries=0,
                )

    assert len(scope.provider_calls) == 1
    assert scope.provider_calls[0].status == ProviderCallStatus.TIMED_OUT


@pytest.mark.asyncio
async def test_retry_attempts_are_recorded_separately(monkeypatch):
    class MockServerError(Exception):
        code = 503

    client, tracker, sdk = _mock_gemini_client(
        side_effect=[MockServerError("503 temporary unavailable"), _response("recovered")]
    )
    monkeypatch.setattr("core.providers.gemini_client.INITIAL_RETRY_DELAY", 0)

    answer, snapshot = await _run_scoped_text_call(client, tracker, max_retries=1)

    assert answer == "recovered"
    assert sdk.aio.models.generate_content.await_count == 2
    assert [record.attempt_number for record in snapshot.provider_calls] == [1, 2]
    assert [record.retry_number for record in snapshot.provider_calls] == [0, 1]
    assert all(not record.is_fallback for record in snapshot.provider_calls)
    assert [record.status for record in snapshot.provider_calls] == [
        ProviderCallStatus.FAILED,
        ProviderCallStatus.SUCCEEDED,
    ]


@pytest.mark.asyncio
async def test_missing_usage_and_model_version_remain_unknown(monkeypatch):
    monkeypatch.setattr("core.providers.base.MODEL_VERSION", "unknown")
    client, tracker, _ = _mock_gemini_client(return_value=_response("usage unavailable"))

    _, snapshot = await _run_scoped_text_call(client, tracker, task_id="estimated-trial")

    record = snapshot.provider_calls[0]
    assert record.model_version is None
    assert record.prompt_tokens is None
    assert record.candidate_tokens is None
    assert record.total_tokens is None
    assert record.estimated_prompt_tokens is not None
    assert record.estimated_candidate_tokens == tracker.estimate_tokens("usage unavailable")
    assert record.estimated_total_tokens == (
        record.estimated_prompt_tokens + record.estimated_candidate_tokens
    )
    assert record.usage_source == ProviderUsageSource.LOCALLY_ESTIMATED
    assert snapshot.token_usage["usage_source"] == "estimated"


@pytest.mark.asyncio
async def test_sequential_accounting_sessions_do_not_accumulate_previous_trial_usage():
    client, tracker, _ = _mock_gemini_client(
        side_effect=[
            _response("trial one", prompt_tokens=11, candidate_tokens=3, total_tokens=14),
            _response("trial two", prompt_tokens=22, candidate_tokens=5, total_tokens=27),
        ]
    )

    _, first = await _run_scoped_text_call(client, tracker, task_id="trial-one")
    _, second = await _run_scoped_text_call(client, tracker, task_id="trial-two")

    assert first.token_usage["total_tokens"] == 14
    assert second.token_usage["total_tokens"] == 27
    assert [record.trial_id for record in first.provider_calls] == ["trial-one"]
    assert [record.trial_id for record in second.provider_calls] == ["trial-two"]
    assert tracker.get_summary_sync()["total_tokens"] == 0


@pytest.mark.asyncio
async def test_concurrent_accounting_sessions_are_isolated():
    async def respond(model, contents, config):
        await asyncio.sleep(0.01)
        tokens = 10 if "trial-a" in contents else 30
        return _response(contents, prompt_tokens=tokens, candidate_tokens=1, total_tokens=tokens + 1)

    client, tracker, _ = _mock_gemini_client(side_effect=respond)

    async def run_trial(trial_id):
        return await _run_scoped_text_call(
            client,
            tracker,
            task_id=trial_id,
            prompt=trial_id,
        )

    results = await asyncio.gather(run_trial("trial-a"), run_trial("trial-b"))
    snapshots = {snapshot.provider_calls[0].trial_id: snapshot for _, snapshot in results}

    assert snapshots["trial-a"].token_usage["total_tokens"] == 11
    assert snapshots["trial-b"].token_usage["total_tokens"] == 31
    assert snapshots["trial-a"].provider_calls[0].execution_id == "exec-trial-a"
    assert snapshots["trial-b"].provider_calls[0].execution_id == "exec-trial-b"
    assert tracker.get_summary_sync()["total_tokens"] == 0


@pytest.mark.asyncio
async def test_partial_trial_snapshot_preserves_successful_and_failed_call_records():
    client, tracker, sdk = _mock_gemini_client(
        side_effect=[_response("first succeeded", prompt_tokens=8, candidate_tokens=2, total_tokens=10), RuntimeError("later failure")]
    )
    async with tracker.accounting_session(AccountingContext(trial_id="partial")) as scope:
        async with tracker.task_budget(essential_reserves={}):
            assert await client.generate_text("first", max_retries=0) == "first succeeded"
            with pytest.raises(LLMError):
                await client.generate_text("second", max_retries=0)

    assert sdk.aio.models.generate_content.await_count == 2
    assert [record.status for record in scope.provider_calls] == [
        ProviderCallStatus.SUCCEEDED,
        ProviderCallStatus.FAILED,
    ]
    assert scope.snapshot().token_usage["total_tokens"] == 10
    assert scope.snapshot().token_usage["failed_requests"] == 1


@pytest.mark.asyncio
async def test_telemetry_finalizer_failure_does_not_change_successful_response(monkeypatch):
    client, tracker, _ = _mock_gemini_client(return_value=_response("still successful"))

    def broken_finish(self, status, error=None):
        raise RuntimeError("telemetry finalizer failed")

    monkeypatch.setattr(ProviderCallRecorder, "finish", broken_finish)

    answer, snapshot = await _run_scoped_text_call(client, tracker, task_id="telemetry-fault")

    assert answer == "still successful"
    assert snapshot.provider_calls[0].status == ProviderCallStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_normal_provider_calls_work_without_benchmark_accounting_scope():
    client, tracker, _ = _mock_gemini_client(return_value=_response("normal MEDHA call"))

    answer = await client.generate_text("normal execution", max_retries=0)

    assert answer == "normal MEDHA call"
    assert len(tracker.get_provider_call_records()) == 1


@pytest.mark.asyncio
async def test_sequential_baseline_runs_use_client_tracker_and_independent_accounting():
    client, tracker, _ = _mock_gemini_client(
        side_effect=[
            _response("baseline one", prompt_tokens=9, candidate_tokens=2, total_tokens=11),
            _response("baseline two", prompt_tokens=19, candidate_tokens=4, total_tokens=23),
        ]
    )
    baseline = SingleAgentBaseline(llm_client=client)

    first = await baseline.run(
        "baseline one",
        accounting_context=AccountingContext(trial_id="baseline-trial-1"),
    )
    second = await baseline.run(
        "baseline two",
        accounting_context=AccountingContext(trial_id="baseline-trial-2"),
    )

    assert baseline.tracker is tracker
    assert first.accounting.context.trial_id == "baseline-trial-1"
    assert second.accounting.context.trial_id == "baseline-trial-2"
    assert first.token_usage["total_tokens"] == 11
    assert second.token_usage["total_tokens"] == 23
    assert first.accounting.provider_calls[0].trial_id == "baseline-trial-1"
    assert second.accounting.provider_calls[0].trial_id == "baseline-trial-2"
    assert tracker.get_summary_sync()["total_tokens"] == 0


@pytest.mark.asyncio
async def test_failed_baseline_persists_sanitized_provider_call_record(monkeypatch):
    client, _, _ = _mock_gemini_client(
        side_effect=RuntimeError("Authorization: Bearer baseline-secret")
    )
    saved_records = []
    monkeypatch.setattr(
        "benchmark.baseline.ExecutionRecord.save",
        lambda record, filepath=None: saved_records.append(record.to_dict()) or filepath,
    )
    baseline = SingleAgentBaseline(llm_client=client)

    result = await baseline.run("Baseline task")

    assert result.success is False
    assert "baseline-secret" not in result.error
    assert result.provider_call_records[0].status == ProviderCallStatus.FAILED
    assert saved_records[0]["status"] == "FAILED"
    serialized = json.dumps(saved_records[0]["metadata"]["provider_calls"])
    assert "baseline-secret" not in serialized
    assert "Authorization" not in serialized
    assert "baseline-secret" not in json.dumps(saved_records[0]["errors"])


@pytest.mark.asyncio
async def test_plan_preparation_cost_is_separate_from_planned_trial_execution(
    monkeypatch,
):
    from core.orchestrator import MedhaOrchestrator
    from benchmark.treatments import get_treatment_preset

    task = "Shared task block with frozen plan"
    plan = TaskPlan(
        task="Synthesized plan identity",
        subtasks=[SubTask(id="T1", role="Analyst", objective="Analyze task")],
    )
    client, tracker, _ = _mock_gemini_client(
        return_value=_response("plan", prompt_tokens=15, candidate_tokens=4, total_tokens=19)
    )
    orchestrator = MedhaOrchestrator(llm_client=client)

    async def create_plan(self, task_description, max_subtasks=4):
        await self.client.generate_text(
            task_description,
            model="mock-model",
            temperature=0.2,
            max_tokens=50,
        )
        return plan

    monkeypatch.setattr("core.orchestrator.TaskPlanner.create_plan", create_plan)
    monkeypatch.setattr(
        "core.orchestrator.ExecutionEngine.execute_plan",
        AsyncMock(return_value=[
            TaskResult(
                subtask_id="T1",
                agent_id="agent_T1",
                role="Analyst",
                status=TaskStatus.COMPLETED,
            )
        ]),
    )
    monkeypatch.setattr("core.orchestrator.AdversarialReviewer.review_claims", AsyncMock(return_value=[]))
    monkeypatch.setattr("core.orchestrator.RepairEngine.execute_repair_loop", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        "agents.synthesizer_agent.SynthesizerAgent.synthesize",
        AsyncMock(return_value="Mocked planned trial synthesis"),
    )
    monkeypatch.setattr("core.orchestrator.ExecutionRecord.save", lambda self, filepath=None: filepath)

    frozen = await orchestrator.prepare_frozen_plan(
        task,
        accounting_context=AccountingContext(experiment_id="exp-1", task_block_id="block-1"),
    )
    result = await orchestrator.run_planned_task(
        task,
        frozen,
        get_treatment_preset("INDEPENDENT_PARALLEL").policy,
        accounting_context=AccountingContext(
            experiment_id="exp-1",
            task_block_id="block-1",
            trial_id="trial-b",
            treatment_id="INDEPENDENT_PARALLEL",
        ),
    )

    assert frozen.preparation_accounting.context.scope_kind == "plan_preparation"
    assert frozen.preparation_accounting.context.trial_id is None
    assert frozen.preparation_accounting.context.treatment_id is None
    assert frozen.preparation_accounting.token_usage["total_tokens"] == 19
    assert len(frozen.preparation_accounting.provider_calls) == 1
    assert frozen.preparation_accounting.provider_calls[0].stage == "planning"
    assert result.accounting.context.scope_kind == "trial_execution"
    assert result.accounting.context.trial_id == "trial-b"
    assert result.accounting.token_usage["total_tokens"] == 0
    assert result.accounting.provider_calls == []
    assert result.execution_id != frozen.preparation_accounting.context.execution_id
    assert tracker.get_summary_sync()["total_tokens"] == 0


@pytest.mark.asyncio
async def test_failed_plan_preparation_persists_calls_collected_before_failure(monkeypatch):
    from core.orchestrator import MedhaOrchestrator

    client, _, _ = _mock_gemini_client(
        return_value=_response("planner response", prompt_tokens=4, candidate_tokens=2, total_tokens=6)
    )
    saved_records = []
    monkeypatch.setattr(
        "core.orchestrator.ExecutionRecord.save",
        lambda record, filepath=None: saved_records.append(record.to_dict()) or filepath,
    )

    async def create_plan(self, task_description, max_subtasks=4):
        await self.client.generate_text(task_description, max_retries=0)
        raise ValueError("synthetic plan validation failure")

    monkeypatch.setattr("core.orchestrator.TaskPlanner.create_plan", create_plan)
    orchestrator = MedhaOrchestrator(llm_client=client)

    with pytest.raises(ValueError, match="synthetic plan validation failure") as raised:
        await orchestrator.prepare_frozen_plan("Task whose planning fails")

    assert raised.value.preparation_accounting.token_usage["total_tokens"] == 6
    assert len(raised.value.preparation_accounting.provider_calls) == 1
    assert saved_records[0]["status"] == "FAILED"
    assert len(saved_records[0]["metadata"]["provider_calls"]) == 1
    assert saved_records[0]["metadata"]["provider_calls"][0]["stage"] == "planning"
    assert "synthetic plan validation failure" not in json.dumps(
        saved_records[0]["metadata"]["provider_calls"]
    )


@pytest.mark.asyncio
async def test_groq_structured_format_fallback_records_each_wire_attempt():
    sdk_client = MagicMock()
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"value":"accepted"}'))],
        usage=usage,
    )
    sdk_client.chat.completions.create = AsyncMock(
        side_effect=[RuntimeError("response_format unsupported by mock model"), response]
    )
    tracker = TokenTracker(name="groq_telemetry_test", essential_token_reserves={})
    client = GroqClient(
        api_key="mock-secret-never-record",
        default_model="mock-groq-model",
        token_tracker=tracker,
        client=sdk_client,
    )

    async with tracker.accounting_session(AccountingContext(trial_id="groq-fallback")) as scope:
        async with tracker.task_budget(essential_reserves={}):
            output = await client.generate_structured(
                prompt="safe structured request",
                response_schema=_StructuredAnswer,
                temperature=0.4,
                max_tokens=48,
                max_retries=0,
            )

    assert output.value == "accepted"
    assert sdk_client.chat.completions.create.await_count == 2
    assert [record.attempt_number for record in scope.provider_calls] == [1, 2]
    assert [record.retry_number for record in scope.provider_calls] == [0, 0]
    assert [record.is_fallback for record in scope.provider_calls] == [False, True]
    assert [record.status for record in scope.provider_calls] == [
        ProviderCallStatus.FAILED,
        ProviderCallStatus.SUCCEEDED,
    ]
    assert scope.provider_calls[1].temperature == 0.4
    assert scope.provider_calls[1].output_token_limit == 48
    assert scope.provider_calls[1].usage_source == ProviderUsageSource.PROVIDER_REPORTED
    serialized = json.dumps([record.model_dump(mode="json") for record in scope.provider_calls])
    assert "mock-secret" not in serialized
    assert "safe structured request" not in serialized


@pytest.mark.asyncio
async def test_groq_json_validate_failed_falls_back_to_plain_text():
    sdk_client = MagicMock()
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"value":"accepted"}'))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )
    json_mode_rejection = RuntimeError(
        "Error code: 400 - {'error': {'message': \"Failed to validate JSON. Please adjust your prompt.\", "
        "'type': 'invalid_request_error', 'code': 'json_validate_failed', 'failed_generation': ''}}"
    )
    sdk_client.chat.completions.create = AsyncMock(side_effect=[json_mode_rejection, response])
    tracker = TokenTracker(name="groq_json_validate_test", essential_token_reserves={})
    client = GroqClient(
        api_key="mock-secret-never-record",
        default_model="mock-groq-model",
        token_tracker=tracker,
        client=sdk_client,
    )

    async with tracker.task_budget(essential_reserves={}):
        output = await client.generate_structured(
            prompt="safe structured request",
            response_schema=_StructuredAnswer,
            max_retries=0,
        )

    assert output.value == "accepted"
    assert sdk_client.chat.completions.create.await_count == 2
    assert "response_format" in sdk_client.chat.completions.create.await_args_list[0].kwargs
    assert "response_format" not in sdk_client.chat.completions.create.await_args_list[1].kwargs


@pytest.mark.asyncio
async def test_normal_medha_execution_persists_provider_calls_without_benchmark_scope(monkeypatch):
    from core.orchestrator import MedhaOrchestrator

    task = "Normal non-benchmark execution"
    plan = TaskPlan(
        task="Synthesized plan identity",
        subtasks=[SubTask(id="T1", role="Analyst", objective="Analyze task")],
    )
    client, _, _ = _mock_gemini_client(
        return_value=_response("planner response", prompt_tokens=6, candidate_tokens=2, total_tokens=8)
    )
    orchestrator = MedhaOrchestrator(llm_client=client)

    async def create_plan(self, task_description, max_subtasks=4):
        await self.client.generate_text(task_description, temperature=0.2, max_tokens=40)
        return plan

    saved_records = []
    monkeypatch.setattr("core.orchestrator.TaskPlanner.create_plan", create_plan)
    monkeypatch.setattr(
        "core.orchestrator.ExecutionEngine.execute_plan",
        AsyncMock(return_value=[
            TaskResult(
                subtask_id="T1",
                agent_id="agent_T1",
                role="Analyst",
                status=TaskStatus.COMPLETED,
            )
        ]),
    )
    monkeypatch.setattr(
        "agents.synthesizer_agent.SynthesizerAgent.synthesize",
        AsyncMock(return_value="Mocked normal synthesis"),
    )
    monkeypatch.setattr(
        "core.orchestrator.ExecutionRecord.save",
        lambda record, filepath=None: saved_records.append(record.to_dict()) or filepath,
    )

    result = await orchestrator.run_task(task)

    assert result.status == "SUCCESS"
    assert len(result.provider_call_records) == 1
    call = result.provider_call_records[0]
    assert call.execution_id == result.execution_id
    assert call.stage == "planning"
    assert saved_records[0]["metadata"]["provider_calls"][0]["call_id"] == call.call_id


@pytest.mark.asyncio
async def test_provider_call_history_is_bounded_and_reports_dropped_records(monkeypatch):
    monkeypatch.setattr("utils.token_tracker.MAX_RETAINED_PROVIDER_CALLS", 2)
    client, tracker, _ = _mock_gemini_client(
        side_effect=[_response("one"), _response("two"), _response("three")]
    )

    async with tracker.accounting_session(AccountingContext(trial_id="bounded")) as scope:
        async with tracker.task_budget(essential_reserves={}):
            for index in range(3):
                await client.generate_text(f"request-{index}", max_retries=0)
        snapshot = scope.snapshot()

    assert len(snapshot.provider_calls) == 2
    assert snapshot.dropped_provider_call_records == 1
    assert len(tracker.get_provider_call_records()) == 2
    assert tracker._dropped_provider_call_records == 1