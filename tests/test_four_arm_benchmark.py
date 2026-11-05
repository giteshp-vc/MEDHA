import csv
import json
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.evaluator import BenchmarkEvaluator
from benchmark.experiment import (
    BenchmarkArmResult,
    BenchmarkConfigurationSnapshot,
    BenchmarkEvaluationResult,
    BenchmarkExperiment,
    BenchmarkTaskBlock,
    BenchmarkTrial,
    MetricMeasurement,
    WILLIAMS_SEQUENCES,
    balanced_sequence,
)
from benchmark.four_arm import _cost_for
from benchmark.four_arm import FourArmBenchmark
from models.provider_call import ProviderCallRecord
from models.frozen_plan import FrozenTaskPlan
from models.provider_call import (
    AccountingContext,
    AccountingSnapshot,
    ProviderCallRecord,
    ProviderCallStatus,
    ProviderUsageSource,
)
from models.task_models import SubTask, TaskPlan


def _snapshot(
    context,
    *,
    total_tokens=12,
    dropped=0,
    model="mock-model",
    model_version="v1",
    model_version_source="provider_response",
    timeout_seconds=30,
    timeout_source="explicit_request",
):
    call = ProviderCallRecord(
        experiment_id=context.experiment_id,
        task_block_id=context.task_block_id,
        trial_id=context.trial_id,
        treatment_id=context.treatment_id,
        execution_id=context.execution_id or "exec-mock",
        attempt_number=1,
        stage=(
            "baseline" if context.scope_kind == "baseline_execution"
            else "planning" if context.scope_kind == "plan_preparation"
            else "specialist_execution"
        ),
        provider="groq",
        model=model,
        model_version=model_version,
        model_version_source=model_version_source,
        temperature=0.2,
        input_token_limit=1000,
        output_token_limit=100,
        timeout_seconds=timeout_seconds,
        timeout_source=timeout_source,
        prompt_tokens=8,
        candidate_tokens=4,
        total_tokens=12,
        usage_source=ProviderUsageSource.PROVIDER_REPORTED,
        status=ProviderCallStatus.SUCCEEDED,
    )
    return AccountingSnapshot(
        context=context,
        duration_seconds=0.01,
        token_usage={
            "prompt_tokens": 8,
            "candidate_tokens": 4,
            "total_tokens": total_tokens,
            "accounted_total_tokens": total_tokens,
            "usage_source": "provider",
            "total_requests": 1 + dropped,
            "failed_requests": 0,
        },
        provider_calls=[call],
        dropped_provider_call_records=dropped,
    )


class FakeBaseline:
    def __init__(self, config=None, effective_model="mock-model", missing_provider_metadata=False):
        self.calls = []
        self.config = config or _configuration()
        self.effective_model = effective_model
        self.missing_provider_metadata = missing_provider_metadata

    def benchmark_configuration(self):
        return self.config

    async def run(self, task, accounting_context=None):
        self.calls.append(accounting_context)
        context = AccountingContext.model_validate({
            **accounting_context.model_dump(exclude_none=True),
            "execution_id": "baseline-exec",
            "scope_kind": "baseline_execution",
        })
        return SimpleNamespace(
            answer="Answer with evidence and source references.",
            execution_id="baseline-exec",
            duration_seconds=0.1,
            success=True,
            error=None,
            token_usage={},
            accounting=_snapshot(
                context,
                model=self.effective_model,
                model_version=None if self.missing_provider_metadata else "v1",
                model_version_source="unavailable" if self.missing_provider_metadata else "provider_response",
                timeout_seconds=None if self.missing_provider_metadata else 30,
                timeout_source="provider_sdk_default" if self.missing_provider_metadata else "explicit_request",
            ),
        )


def _configuration():
    return {
        "provider": "groq",
        "model": "mock-model",
        "model_version": "v1",
        "temperature": 0.2,
        "input_token_limit": 1000,
        "output_token_limit": 100,
        "task_token_budget": 10000,
        "call_budget": 10,
    }


class FakeOrchestrator:
    def __init__(self, *, planning_error=None, interrupt_arm=None, missing_provider_metadata=False):
        self.prepare_calls = []
        self.planned_calls = []
        self.copy_ids = []
        self.planning_error = planning_error
        self.interrupt_arm = interrupt_arm
        self.missing_provider_metadata = missing_provider_metadata

    def _snapshot(self, context, **kwargs):
        if self.missing_provider_metadata:
            kwargs.update(
                model_version=None,
                model_version_source="unavailable",
                timeout_seconds=None,
                timeout_source="provider_sdk_default",
            )
        return _snapshot(context, **kwargs)

    async def prepare_frozen_plan(self, task, accounting_context=None):
        self.prepare_calls.append((task, accounting_context))
        if self.planning_error is not None:
            self.planning_error.preparation_accounting = _snapshot(AccountingContext(
                experiment_id=accounting_context.experiment_id,
                task_block_id=accounting_context.task_block_id,
                execution_id="failed-plan-exec",
                scope_kind="plan_preparation",
            ))
            raise self.planning_error
        plan = TaskPlan(
            task="Stable task plan",
            subtasks=[SubTask(id="T1", role="Analyst", objective="Analyze")],
        )
        return FrozenTaskPlan.from_plan(
            task,
            plan,
            preparation_accounting=self._snapshot(AccountingContext(
                experiment_id=accounting_context.experiment_id,
                task_block_id=accounting_context.task_block_id,
                execution_id="plan-exec",
                scope_kind="plan_preparation",
            )),
        )

    async def run_planned_task(self, task, frozen_plan, policy, accounting_context=None):
        if accounting_context.treatment_id == self.interrupt_arm:
            interrupted = asyncio.CancelledError()
            interrupted.accounting_snapshot = _snapshot(accounting_context.model_copy(update={
                "execution_id": f"partial-{accounting_context.trial_id}",
                "scope_kind": "trial_execution",
            }))
            raise interrupted
        copy = frozen_plan.load_plan(task)
        self.copy_ids.append(id(copy))
        original_role = copy.subtasks[0].role
        copy.subtasks[0].role = f"mutated-{accounting_context.treatment_id}"
        assert original_role == "Analyst"
        self.planned_calls.append((frozen_plan.plan_hash, policy, accounting_context))
        context = AccountingContext.model_validate({
            **accounting_context.model_dump(exclude_none=True),
            "execution_id": f"exec-{accounting_context.trial_id}",
            "scope_kind": "trial_execution",
        })
        return SimpleNamespace(
            final_answer="Answer with evidence and source references.",
            execution_id=context.execution_id,
            status="SUCCESS",
            duration_seconds=0.2,
            token_usage={},
            error=None,
            verification_report={
                "total_claims": 2,
                "accepted_claims": [{"claim_id": "c1"}],
                "rejected_claims": [{"claim_id": "c2"}],
                "support_rate": 0.5,
                "unsupported_rate": 0.5,
                "repair_success_rate": 0.25,
            },
            accounting=self._snapshot(
                context,
                dropped=int(accounting_context.treatment_id == "DEPENDENCY_AWARE"),
            ),
        )


class FakeMedhaRunner:
    def __init__(self, *, config=None, planning_error=None, interrupt_arm=None, missing_provider_metadata=False):
        self.orchestrator = FakeOrchestrator(
            planning_error=planning_error,
            interrupt_arm=interrupt_arm,
            missing_provider_metadata=missing_provider_metadata,
        )
        self.config = config or _configuration()

    def benchmark_configuration(self):
        return self.config


@pytest.mark.parametrize("block,repetition,repetitions,expected", [
    (0, 0, 1, WILLIAMS_SEQUENCES[0]),
    (1, 0, 1, WILLIAMS_SEQUENCES[1]),
    (0, 1, 2, WILLIAMS_SEQUENCES[1]),
    (1, 1, 2, WILLIAMS_SEQUENCES[3]),
])
def test_balanced_sequence_rotates_deterministically(block, repetition, repetitions, expected):
    assert balanced_sequence(block, repetition, repetitions) == expected


def test_williams_sequences_balance_positions_and_directed_carryovers():
    position_counts = {arm: [0, 0, 0, 0] for arm in ("BASELINE", "INDEPENDENT_PARALLEL", "DEPENDENCY_AWARE", "FULL_MEDHA")}
    carryovers = []
    for sequence in WILLIAMS_SEQUENCES:
        for position, arm in enumerate(sequence):
            position_counts[arm][position] += 1
        carryovers.extend(zip(sequence, sequence[1:]))

    assert all(counts == [1, 1, 1, 1] for counts in position_counts.values())
    assert len(carryovers) == 12
    assert len(set(carryovers)) == 12


def test_smoke_fixture_operational_aggregates_include_incompatible_attempts():
    fixture_path = Path(__file__).parent / "fixtures" / "four_arm_smoke_minimal.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    trials = []
    for repetition, item in enumerate(fixture["trials"], start=1):
        provider_calls = [
            ProviderCallRecord(
                attempt_number=index + 1,
                provider=fixture["configuration"]["provider"],
                model=fixture["configuration"]["model"],
                model_version=fixture["configuration"]["model_version"],
                model_version_source="unavailable",
                temperature=0.2,
                input_token_limit=12000,
                output_token_limit=2048,
                timeout_seconds=fixture["configuration"]["timeout_seconds"],
                timeout_source="provider_sdk_default",
                status=(
                    ProviderCallStatus.FAILED
                    if index < item["failed_provider_request_count"]
                    else ProviderCallStatus.SUCCEEDED
                ),
            )
            for index in range(item["provider_request_count"])
        ]
        context = AccountingContext(
            experiment_id="smoke-minimal",
            task_block_id="block_0001",
            trial_id=f"smoke-minimal:{item['arm_id']}",
            treatment_id=item["arm_id"],
            execution_id=f"exec-{item['arm_id']}",
            scope_kind="trial_execution",
        )
        accounting = AccountingSnapshot(
            context=context,
            duration_seconds=item["duration_seconds"],
            token_usage={
                "prompt_tokens": item["input_tokens"],
                "candidate_tokens": item["output_tokens"],
                "total_tokens": item["provider_reported_total_tokens"],
                "estimated_tokens_without_usage": item["estimated_total_tokens"],
                "accounted_total_tokens": item["total_tokens"],
                "usage_source": item["usage_source"],
            },
            provider_calls=provider_calls,
        )
        compatible = fixture["configuration"]["all_arms_compatible"]
        result = BenchmarkArmResult(
            experiment_id="smoke-minimal",
            task_block_id="block_0001",
            trial_id=context.trial_id,
            execution_id=context.execution_id,
            arm_id=item["arm_id"],
            task_id="smoke-case",
            repetition=1,
            start_timestamp="2026-11-05T00:00:00+00:00",
            end_timestamp="2026-11-05T00:00:01+00:00",
            duration_seconds=item["duration_seconds"],
            status=item["status"],
            success=item["success"],
            quality_eligible=item["success"],
            metrics={
                "answer_completeness": MetricMeasurement(
                    value=0.0 if item["success"] else 1.0,
                    eligible_sample_count=int(item["success"]),
                    attempt_count=1,
                    missing_count=int(not item["success"]),
                    missingness_reason=None if item["success"] else "output_ineligible",
                ),
                "verification_support_rate": MetricMeasurement(
                    value=None,
                    attempt_count=1,
                    missing_count=1,
                    missingness_reason="not_applicable" if item["arm_id"] == "BASELINE" else "verification_unavailable",
                ),
            },
            provider_call_records=provider_calls,
            accounting=accounting,
            trial_execution_cost={"amount_usd": None, "status": "unknown"},
            telemetry_complete=True,
            configuration=BenchmarkConfigurationSnapshot(
                provider_configuration={
                    "provider": fixture["configuration"]["provider"],
                    "model": fixture["configuration"]["model"],
                },
                compatible=compatible,
                compatibility_status="INCOMPATIBLE",
                compatibility_assessments={
                    "provider_model_revision": {"status": "unavailable"},
                    "request_timeout_seconds": {"status": "unavailable"},
                },
                compatibility_limitations=["revision and default timeout are unavailable"],
                compatibility_differences={"fixture": "marked incompatible per source artifact"},
            ),
        )
        trial_id = result.trial_id
        trials.append(BenchmarkTrial(
            trial_id=trial_id,
            task_block_id="block_0001",
            repetition=repetition,
            arm_id=item["arm_id"],
            planned_order=[item["arm_id"]],
            actual_order=[item["arm_id"]],
            result=result,
        ))

    experiment = BenchmarkExperiment(
        experiment_id="smoke-minimal",
        created_at="2026-11-05T00:00:00+00:00",
        repetitions=1,
        task_blocks=[BenchmarkTaskBlock(
            task_block_id="block_0001",
            task_id="smoke-case",
            task_text="sanitized task reference",
            trials=trials,
        )],
    )
    aggregate_runner = FourArmBenchmark.__new__(FourArmBenchmark)
    operational, comparative, legacy = aggregate_runner._aggregate(experiment)
    total_operational = {
        name: sum(arm[name] for arm in operational.values())
        for name in ("attempt_count", "successful_completion_count", "failure_or_interruption_count")
    }

    assert total_operational == {
        "attempt_count": 4,
        "successful_completion_count": 1,
        "failure_or_interruption_count": 3,
    }
    assert operational["INDEPENDENT_PARALLEL"]["provider_request_count"]["value"] == 11
    assert operational["INDEPENDENT_PARALLEL"]["failed_provider_request_count"]["value"] == 2
    assert operational["FULL_MEDHA"]["provider_request_count"]["value"] == 4
    assert operational["DEPENDENCY_AWARE"]["provider_request_count"]["value"] == 12
    assert sum(arm["provider_request_count"]["value"] for arm in operational.values()) == 28
    assert sum(arm["failed_provider_request_count"]["value"] for arm in operational.values()) == 5
    assert sum(arm["tokens"]["total_tokens"]["value"] for arm in operational.values()) == 97173
    assert sum(arm["tokens"]["estimated_tokens"]["value"] for arm in operational.values()) == 21721
    assert sum(arm["tokens"]["provider_reported_total_tokens"]["value"] for arm in operational.values()) == 75452
    assert operational["BASELINE"]["success_rate"] == 1.0
    assert comparative["BASELINE"]["success_rate"] is None
    assert comparative["BASELINE"]["answer_quality_metrics"]["answer_completeness"]["value"] is None
    assert comparative["BASELINE"]["answer_quality_metrics"]["answer_completeness"]["missingness_reasons"] == {
        "configuration_incompatible": 1
    }
    assert comparative["BASELINE"]["answer_quality_metrics"]["answer_completeness"]["attempt_count"] == 1
    assert comparative["BASELINE"]["answer_quality_metrics"]["answer_completeness"]["missing_count"] == 1
    assert legacy["BASELINE"]["success_rate"] == 1.0
    assert legacy["INDEPENDENT_PARALLEL"]["failure_or_interruption_count"] == 1
    assert operational["BASELINE"]["trial_execution_cost"]["amount_usd"] is None
    assert operational["BASELINE"]["trial_execution_cost"]["unknown_cost_count"] == 1


@pytest.mark.parametrize("usage_source", ["estimated", "mixed"])
def test_cost_does_not_price_estimated_or_mixed_usage_as_zero(usage_source):
    cost = _cost_for(
        {
            "usage_source": usage_source,
            "prompt_tokens": 0,
            "candidate_tokens": 0,
            "total_tokens": 0,
            "estimated_tokens_without_usage": 75,
            "accounted_total_tokens": 75,
        },
        {"provider": "groq", "model": "mock-model"},
        {"groq:mock-model": {
            "input_per_million_usd": 1.0,
            "output_per_million_usd": 2.0,
        }},
    )
    assert cost["status"] == usage_source
    assert cost["amount_usd"] is None


def test_missing_model_revision_and_sdk_default_timeout_are_nonblocking_limitations():
    baseline = {**_configuration(), "model_version": "unknown", "provider_request_timeout_seconds": None}
    medha = {**_configuration(), "model_version": None, "provider_request_timeout_seconds": None}

    differences = FourArmBenchmark._configuration_differences({
        "BASELINE": baseline,
        "MEDHA": medha,
    })
    assessments, limitations = FourArmBenchmark._configuration_assessments({
        "BASELINE": baseline,
        "MEDHA": medha,
    })

    assert differences == {}
    assert assessments["provider_model_revision"]["status"] == "unavailable"
    assert assessments["request_timeout_mode"]["status"] == "known_equal"
    assert assessments["request_timeout_seconds"]["status"] == "unavailable"
    assert assessments["plan_preparation_for_baseline"]["status"] == "not_applicable"
    assert any("mutable aliases" in item for item in limitations)
    assert any("provider-managed" in item for item in limitations)


@pytest.mark.parametrize(
    ("field", "baseline_value", "medha_value"),
    [
        ("provider", "groq", "gemini"),
        ("model", "model-a", "model-b"),
        ("temperature", 0.2, 0.4),
        ("output_token_limit", 2048, 1024),
        ("provider_request_timeout_seconds", 10.0, 20.0),
    ],
)
def test_known_provider_and_generation_configuration_differences_are_rejected(
    field,
    baseline_value,
    medha_value,
):
    baseline = {**_configuration(), field: baseline_value}
    medha = {**_configuration(), field: medha_value}

    differences = FourArmBenchmark._configuration_differences({
        "BASELINE": baseline,
        "MEDHA": medha,
    })

    assert field in differences


def test_operational_aggregates_preserve_partial_telemetry_denominators():
    context = AccountingContext(
        experiment_id="partial-telemetry",
        task_block_id="block-1",
        trial_id="known",
        execution_id="exec-known",
        scope_kind="trial_execution",
    )
    known_accounting = AccountingSnapshot(
        context=context,
        duration_seconds=0.1,
        token_usage={
            "prompt_tokens": 9,
            "candidate_tokens": 4,
            "total_tokens": 13,
            "accounted_total_tokens": 13,
            "estimated_tokens_without_usage": 0,
            "usage_source": "provider",
        },
    )
    common = {
        "experiment_id": "partial-telemetry",
        "task_block_id": "block-1",
        "arm_id": "BASELINE",
        "task_id": "case",
        "repetition": 1,
        "start_timestamp": "start",
        "end_timestamp": "end",
        "status": "SUCCESS",
        "success": True,
        "quality_eligible": True,
        "configuration": BenchmarkConfigurationSnapshot(compatible=True),
        "trial_execution_cost": {"amount_usd": None, "status": "unknown"},
    }
    known = BenchmarkArmResult(
        **common,
        trial_id="known",
        execution_id="exec-known",
        duration_seconds=0.1,
        accounting=known_accounting,
    )
    missing = BenchmarkArmResult(
        **{**common, "status": "FAILED", "success": False, "quality_eligible": False},
        trial_id="missing",
        execution_id="exec-missing",
        duration_seconds=0.2,
    )

    operational = FourArmBenchmark._aggregate_operational([known, missing])

    assert operational["attempt_count"] == 2
    assert operational["success_rate"] == 0.5
    assert operational["provider_request_count"] == {
        "value": 0,
        "attempt_count": 2,
        "available_attempt_count": 1,
        "missing_count": 1,
        "missingness_reason": "provider-call records unavailable",
    }
    assert operational["tokens"]["input_tokens"]["value"] == 9
    assert operational["tokens"]["input_tokens"]["available_attempt_count"] == 1
    assert operational["tokens"]["input_tokens"]["missing_count"] == 1

    empty = FourArmBenchmark._aggregate_operational([])
    assert empty["attempt_count"] == 0
    assert empty["provider_request_count"]["value"] is None
    assert empty["tokens"]["total_tokens"]["value"] is None
    assert empty["latency"]["average_seconds"] is None


@pytest.mark.asyncio
async def test_four_arm_experiment_reuses_plan_isolates_trials_and_exports(tmp_path):
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text(json.dumps([{
        "id": "case-1",
        "domain": "systems",
        "prompt": "Analyze the mocked system.",
        "reference_criteria": ["Answer includes evidence and source references"],
        "known_pitfalls": [],
    }]), encoding="utf-8")
    baseline = FakeBaseline()
    medha = FakeMedhaRunner()
    evaluator = BenchmarkEvaluator(
        dataset_path=dataset_path,
        baseline_runner=baseline,
        medha_runner=medha,
        pricing={"groq:mock-model": {
            "input_per_million_usd": 1.0,
            "output_per_million_usd": 2.0,
        }},
    )

    result = await evaluator.run_experiment(
        max_cases=1,
        repetitions=1,
        experiment_id="four-arm-test",
        output_directory=tmp_path / "exports",
    )

    block = result.experiment.task_blocks[0]
    assert len(medha.orchestrator.prepare_calls) == 1
    assert len(baseline.calls) == 1
    assert len(medha.orchestrator.planned_calls) == 3
    assert len(set(medha.orchestrator.copy_ids)) == 3
    assert {call[0] for call in medha.orchestrator.planned_calls} == {block.plan_hash}
    policies = [policy for _, policy, _ in medha.orchestrator.planned_calls]
    assert [policy.deliberation_enabled for policy in policies] == [False, True, False]
    assert [policy.execution.honor_dependencies for policy in policies] == [False, True, True]
    assert [policy.execution.route_messages for policy in policies] == [False, True, True]
    assert all(policy.verification_enabled and policy.synthesis_enabled for policy in policies)
    assert len({call[2].trial_id for call in medha.orchestrator.planned_calls}) == 3
    assert len({call[2].treatment_id for call in medha.orchestrator.planned_calls}) == 3

    schedule = block.schedules[0]
    assert schedule.planned_order == list(WILLIAMS_SEQUENCES[0])
    assert schedule.actual_order == schedule.planned_order
    assert len(block.trials) == 4
    assert len({trial.trial_id for trial in block.trials}) == 4
    assert all(trial.result.plan_hash == block.plan_hash for trial in block.trials if trial.arm_id != "BASELINE")
    assert next(t.result for t in block.trials if t.arm_id == "BASELINE").plan_hash is None
    baseline_result = next(t.result for t in block.trials if t.arm_id == "BASELINE")
    assert baseline_result.verification_metrics == {
        "support_rate": None,
        "unsupported_rate": None,
        "repair_success_rate": None,
    }
    assert baseline_result.metrics["verification_support_rate"].missingness_reason == "verification not executed"
    assert baseline_result.metrics["factual_correctness"].value is None
    dependency_trial = next(t.result for t in block.trials if t.arm_id == "DEPENDENCY_AWARE")
    assert dependency_trial.dropped_provider_call_records == 1
    assert not dependency_trial.telemetry_complete
    assert result.aggregate_results["BASELINE"]["verification_metrics"]["support_rate"] is None
    assert result.experiment.scoring_policy["factual_correctness"] is None
    assert result.aggregate_results["DEPENDENCY_AWARE"]["provider_request_count"] == 2
    assert result.operational_aggregates["DEPENDENCY_AWARE"]["retry_and_fallback_counts_are_lower_bounds"] is True

    plan_amount = (
        block.plan_preparation_accounting.token_usage["prompt_tokens"] * 1.0
        + block.plan_preparation_accounting.token_usage["candidate_tokens"] * 2.0
    ) / 1_000_000
    trial_amount = 8 * 1.0 / 1_000_000 + 4 * 2.0 / 1_000_000
    assert result.actual_experiment_cost["amount_usd"] == pytest.approx(plan_amount + trial_amount * 4)
    for arm in ("INDEPENDENT_PARALLEL", "DEPENDENCY_AWARE", "FULL_MEDHA"):
        trial = next(item.result for item in block.trials if item.arm_id == arm)
        assert trial.end_to_end_cost["amount_usd"] == pytest.approx(plan_amount + trial_amount)

    json_path = tmp_path / "exports" / "four-arm-test.json"
    csv_path = tmp_path / "exports" / "four-arm-test.csv"
    assert json_path.exists() and csv_path.exists()
    decoded = json.loads(json_path.read_text(encoding="utf-8"))
    assert decoded["experiment"]["task_blocks"][0]["plan_hash"] == block.plan_hash
    assert len(decoded["experiment"]["task_blocks"][0]["trials"]) == 4
    round_tripped = BenchmarkEvaluationResult.model_validate(decoded)
    assert round_tripped.experiment.experiment_id == "four-arm-test"
    assert len(round_tripped.experiment.task_blocks[0].trials) == 4
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    assert len(rows) == 4
    assert len({row["trial_id"] for row in rows}) == 4
    assert next(row for row in rows if row["arm_id"] == "BASELINE")["verification_support_rate"] == "null"
    assert rows[0]["total_tokens"] == "12"
    assert rows[0]["usage_source"] == "provider"
    assert rows[0]["answer_completeness_eligible_count"] == "1"

    with pytest.raises(FileExistsError):
        await evaluator.run_experiment(
            max_cases=1,
            repetitions=1,
            experiment_id="four-arm-test",
            output_directory=tmp_path / "exports",
        )


@pytest.mark.asyncio
async def test_experiment_rejects_invalid_limits_and_unsafe_ids(tmp_path):
    evaluator = BenchmarkEvaluator(
        dataset_path=tmp_path / "missing.json",
        baseline_runner=FakeBaseline(),
        medha_runner=FakeMedhaRunner(),
    )
    with pytest.raises(ValueError, match="repetitions"):
        await evaluator.run_experiment(repetitions=0)
    with pytest.raises(ValueError, match="max_cases"):
        await evaluator.run_experiment(max_cases=0)
    with pytest.raises(ValueError, match="experiment_id"):
        await evaluator.run_experiment(experiment_id="../unsafe")


@pytest.mark.asyncio
async def test_multiple_blocks_and_repetitions_rotate_full_balanced_cycle(tmp_path):
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text(json.dumps([
        {"id": "case-1", "prompt": "Task one"},
        {"id": "case-2", "prompt": "Task two"},
    ]), encoding="utf-8")
    evaluator = BenchmarkEvaluator(
        dataset_path=dataset_path,
        baseline_runner=FakeBaseline(),
        medha_runner=FakeMedhaRunner(),
    )

    result = await evaluator.run_experiment(
        repetitions=2,
        experiment_id="rotation-test",
        output_directory=tmp_path / "exports",
    )

    assert [
        schedule.sequence_index
        for block in result.experiment.task_blocks
        for schedule in block.schedules
    ] == [0, 1, 2, 3]
    assert [
        schedule.planned_order
        for block in result.experiment.task_blocks
        for schedule in block.schedules
    ] == [list(WILLIAMS_SEQUENCES[index]) for index in (0, 1, 2, 3)]
    trials = [trial for block in result.experiment.task_blocks for trial in block.trials]
    assert len(trials) == 16
    assert len({trial.trial_id for trial in trials}) == 16
    assert result.experiment.schedule_cycle_complete is True
    assert result.actual_experiment_cost["amount_usd"] is None
    assert result.actual_experiment_cost["status"] == "unknown"


@pytest.mark.asyncio
async def test_configuration_mismatch_is_preserved_but_not_aggregated(tmp_path):
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text('[{"id":"mismatch","prompt":"Task"}]', encoding="utf-8")
    baseline_config = _configuration()
    medha_config = {**_configuration(), "model": "different-model"}
    evaluator = BenchmarkEvaluator(
        dataset_path=dataset_path,
        baseline_runner=FakeBaseline(config=baseline_config),
        medha_runner=FakeMedhaRunner(config=medha_config),
    )

    result = await evaluator.run_experiment(
        experiment_id="config-mismatch",
        output_directory=tmp_path / "exports",
    )

    assert sum(len(block.trials) for block in result.experiment.task_blocks) == 4
    assert result.experiment.configuration_snapshots["BASELINE"].compatible is False
    assert "model" in result.experiment.configuration_snapshots["BASELINE"].compatibility_differences
    assert result.aggregate_results["BASELINE"]["incompatible_attempt_count"] == 1
    assert result.aggregate_results["BASELINE"]["success_rate"] == 1.0
    assert result.comparative_aggregates["BASELINE"]["success_rate"] is None
    assert result.aggregate_results["FULL_MEDHA"]["incompatible_attempt_count"] == 1


@pytest.mark.asyncio
async def test_unavailable_revision_and_default_timeout_are_nonblocking_in_final_experiment(tmp_path):
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text('[{"id":"metadata-unavailable","prompt":"Task"}]', encoding="utf-8")
    config = {
        **_configuration(),
        "model_version": "unknown",
        "provider_request_timeout_seconds": None,
    }
    evaluator = BenchmarkEvaluator(
        dataset_path=dataset_path,
        baseline_runner=FakeBaseline(config=config, missing_provider_metadata=True),
        medha_runner=FakeMedhaRunner(config=config, missing_provider_metadata=True),
    )

    result = await evaluator.run_experiment(
        experiment_id="metadata-unavailable",
        output_directory=tmp_path / "exports",
    )

    for snapshot in result.experiment.configuration_snapshots.values():
        assert snapshot.compatible is True
        assert snapshot.compatibility_status == "COMPATIBLE_WITH_LIMITATIONS"
        assert "mutable provider-managed alias" in snapshot.model_revision_policy
        assert snapshot.compatibility_assessments["provider_model_revision"]["status"] == "unavailable"
        assert snapshot.compatibility_assessments["request_timeout_seconds"]["status"] == "unavailable"
    assert result.comparative_aggregates["BASELINE"]["compatible_attempt_count"] == 1


@pytest.mark.asyncio
async def test_effective_provider_configuration_mismatch_is_detected(tmp_path):
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text('[{"id":"effective-mismatch","prompt":"Task"}]', encoding="utf-8")
    evaluator = BenchmarkEvaluator(
        dataset_path=dataset_path,
        baseline_runner=FakeBaseline(effective_model="unexpected-model"),
        medha_runner=FakeMedhaRunner(),
    )

    result = await evaluator.run_experiment(
        experiment_id="effective-mismatch",
        output_directory=tmp_path / "exports",
    )

    differences = result.experiment.configuration_snapshots["BASELINE"].compatibility_differences
    assert differences["BASELINE.effective_call.model"] == {
        "configured": "mock-model",
        "effective": "unexpected-model",
    }
    assert result.aggregate_results["BASELINE"]["success_rate"] == 1.0
    assert result.comparative_aggregates["BASELINE"]["success_rate"] is None


@pytest.mark.asyncio
async def test_interruption_preserves_completed_trials_and_does_not_run_later_arms(tmp_path):
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text('[{"id":"interrupt","prompt":"Task"}]', encoding="utf-8")
    evaluator = BenchmarkEvaluator(
        dataset_path=dataset_path,
        baseline_runner=FakeBaseline(),
        medha_runner=FakeMedhaRunner(interrupt_arm="FULL_MEDHA"),
    )

    result = await evaluator.run_experiment(
        experiment_id="interrupted-run",
        output_directory=tmp_path / "exports",
    )

    block = result.experiment.task_blocks[0]
    assert [trial.arm_id for trial in block.trials] == [
        "BASELINE", "INDEPENDENT_PARALLEL", "FULL_MEDHA"
    ]
    assert [trial.result.status for trial in block.trials] == ["SUCCESS", "SUCCESS", "INTERRUPTED"]
    assert len(block.trials[-1].result.provider_call_records) == 1
    assert block.trials[-1].result.accounting.context.trial_id == block.trials[-1].trial_id
    assert block.schedules[0].actual_order == [
        "BASELINE", "INDEPENDENT_PARALLEL", "FULL_MEDHA"
    ]
    assert block.schedules[0].interrupted is True
    assert result.experiment.schedule_cycle_complete is False
    json_path = tmp_path / "exports" / "interrupted-run.json"
    csv_path = tmp_path / "exports" / "interrupted-run.csv"
    assert json_path.exists() and csv_path.exists()
    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    persisted_trials = persisted["experiment"]["task_blocks"][0]["trials"]
    assert [trial["arm_id"] for trial in persisted_trials] == [
        "BASELINE", "INDEPENDENT_PARALLEL", "FULL_MEDHA"
    ]


@pytest.mark.asyncio
async def test_plan_failure_still_runs_baseline_only_in_actual_order(tmp_path):
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text('[{"id":"planning-failure","prompt":"Task"}]', encoding="utf-8")
    baseline = FakeBaseline()
    evaluator = BenchmarkEvaluator(
        dataset_path=dataset_path,
        baseline_runner=baseline,
        medha_runner=FakeMedhaRunner(planning_error=RuntimeError("planner failed")),
    )

    result = await evaluator.run_experiment(
        experiment_id="planning-failure",
        output_directory=tmp_path / "exports",
    )

    block = result.experiment.task_blocks[0]
    assert len(baseline.calls) == 1
    assert block.plan_hash is None
    assert block.plan_preparation_error_category == "RuntimeError"
    assert len(block.plan_preparation_accounting.provider_calls) == 1
    assert [trial.arm_id for trial in block.trials] == ["BASELINE"]
    assert block.schedules[0].planned_order == list(WILLIAMS_SEQUENCES[0])
    assert block.schedules[0].actual_order == ["BASELINE"]


@pytest.mark.parametrize("arguments, message", [
    (["--benchmark", "--benchmark-repetitions", "0"], "must be positive"),
    (["--benchmark", "--benchmark-cases", "0"], "must be positive"),
    (["--benchmark", "--experiment-id", "../unsafe"], "safe filename component"),
    (["--benchmark", "--benchmark-output", "..\\outside"], "parent-directory traversal"),
    (["--benchmark-repetitions", "2"], "require --benchmark"),
])
def test_cli_rejects_invalid_benchmark_arguments(monkeypatch, arguments, message):
    import main

    monkeypatch.setattr(sys, "argv", ["main.py", *arguments])
    with pytest.raises(SystemExit) as error:
        main.main()
    assert error.value.code == 2