from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from benchmark.evaluator import BenchmarkEvaluator
from benchmark.treatments import BenchmarkArm, TreatmentPreset, get_treatment_preset
from benchmark.metrics import (
    SystemMetrics,
    TaskEvaluationResult,
    compute_system_metrics,
)
from core.orchestrator import MedhaResult
from benchmark.metrics import BenchmarkConfiguration
from models.execution_policy import ExecutionPolicy, OrchestrationPolicy


def test_compute_system_metrics_calculation():
    """Verify statistical calculation of system performance metrics."""
    results = [
        TaskEvaluationResult(
            task_id="t1",
            domain="Tech",
            mode="MEDHA",
            prompt="Prompt 1",
            answer="Answer 1",
            success=True,
            duration_seconds=10.0,
            token_usage={"total_tokens": 1000, "total_requests": 3},
            criteria_coverage_score=0.8,
            verification_stats={
                "support_rate": 0.85,
                "unsupported_rate": 0.15,
                "repair_success_rate": 0.5,
                "total_claims": 5,
                "rejected_claims_count": 1,
            },
        ),
        TaskEvaluationResult(
            task_id="t2",
            domain="Tech",
            mode="MEDHA",
            prompt="Prompt 2",
            answer="Answer 2",
            success=True,
            duration_seconds=20.0,
            token_usage={"total_tokens": 2000, "total_requests": 5},
            criteria_coverage_score=0.9,
            verification_stats={
                "support_rate": 0.95,
                "unsupported_rate": 0.05,
                "repair_success_rate": 1.0,
                "total_claims": 5,
                "rejected_claims_count": 0,
            },
        ),
    ]

    metrics = compute_system_metrics("Test Mode", results)

    assert metrics.total_tasks == 2
    assert metrics.task_completion_rate == 1.0
    assert metrics.avg_latency_seconds == 15.0
    assert metrics.total_tokens == 3000
    assert metrics.avg_tokens_per_task == 1500.0
    assert metrics.total_api_requests == 8
    assert metrics.reference_criteria_coverage == 0.85
    assert metrics.claim_support_rate == 0.90
    assert metrics.estimated_cost_usd == 0.0
    assert metrics.cost_status == "unknown"


@pytest.mark.parametrize(
    ("arm", "uses_plan", "specialists", "dependencies", "upstream", "messages", "deliberation", "verification", "synthesis"),
    [
        (BenchmarkArm.BASELINE, False, False, False, False, False, False, False, False),
        (BenchmarkArm.INDEPENDENT_PARALLEL, True, True, False, False, False, False, True, True),
        (BenchmarkArm.DEPENDENCY_AWARE, True, True, True, True, True, False, True, True),
        (BenchmarkArm.FULL_MEDHA, True, True, True, True, True, True, True, True),
    ],
)
def test_treatment_presets_match_four_arm_contract(
    arm,
    uses_plan,
    specialists,
    dependencies,
    upstream,
    messages,
    deliberation,
    verification,
    synthesis,
):
    preset = get_treatment_preset(arm)

    assert preset.arm == arm
    assert preset.uses_plan is uses_plan
    assert preset.specialists_enabled is specialists
    assert preset.policy.execution.honor_dependencies is dependencies
    assert preset.policy.execution.inject_upstream_results is upstream
    assert preset.policy.execution.route_messages is messages
    assert preset.policy.deliberation_enabled is deliberation
    assert preset.policy.verification_enabled is verification
    assert preset.policy.synthesis_enabled is synthesis


def test_treatment_presets_reject_invalid_arm_combinations():
    invalid_policy = OrchestrationPolicy(
        execution=ExecutionPolicy(
            honor_dependencies=False,
            inject_upstream_results=False,
            route_messages=False,
        ),
        deliberation_enabled=False,
        verification_enabled=True,
        synthesis_enabled=True,
    )

    with pytest.raises(ValueError, match="do not match benchmark arm"):
        TreatmentPreset(
            arm=BenchmarkArm.INDEPENDENT_PARALLEL,
            uses_plan=False,
            specialists_enabled=True,
            policy=invalid_policy,
        )

    with pytest.raises(ValueError, match="Upstream results require dependency scheduling"):
        ExecutionPolicy(honor_dependencies=False, inject_upstream_results=True)

    with pytest.raises(ValueError, match="Deliberation requires"):
        OrchestrationPolicy(
            execution=ExecutionPolicy(
                honor_dependencies=True,
                inject_upstream_results=True,
                route_messages=False,
            ),
            deliberation_enabled=True,
        )

    with pytest.raises(ValueError, match="supported treatment combination"):
        OrchestrationPolicy(
            execution=ExecutionPolicy(),
            deliberation_enabled=False,
            verification_enabled=False,
            synthesis_enabled=False,
        )


def test_system_metrics_keep_unreviewed_support_rate_undefined():
    result = TaskEvaluationResult(
        task_id="unreviewed",
        domain="Tech",
        mode="MEDHA",
        prompt="Prompt",
        answer="Answer",
        success=False,
        duration_seconds=1.0,
        verification_stats={"support_rate": None, "unsupported_rate": None},
    )

    metrics = compute_system_metrics("MEDHA", [result])

    assert metrics.claim_support_rate is None
    assert metrics.unsupported_claim_rate is None


def test_failed_trial_counts_for_completion_but_not_aggregate_quality():
    successful = TaskEvaluationResult(
        task_id="success",
        domain="Tech",
        mode="MEDHA",
        prompt="Prompt",
        answer="Successful answer",
        success=True,
        completion_status="SUCCESS",
        quality_eligible=True,
        duration_seconds=1.0,
        criteria_coverage_score=0.8,
        answer_metrics={"answer_completeness": 0.8},
    )
    failed = TaskEvaluationResult(
        task_id="failed",
        domain="Tech",
        mode="MEDHA",
        prompt="Prompt",
        answer="Retained answer from a failed run",
        success=False,
        completion_status="FAILED",
        quality_eligible=False,
        duration_seconds=2.0,
        criteria_coverage_score=1.0,
        answer_metrics={"answer_completeness": 1.0},
    )

    metrics = compute_system_metrics("MEDHA", [successful, failed])

    assert metrics.task_completion_rate == 0.5
    assert metrics.failure_count == 1
    assert metrics.reference_criteria_coverage == 0.8
    assert metrics.answer_metrics["answer_completeness"] == 0.8
    assert metrics.metric_sample_counts["answer_completeness"] == 1


def test_evaluate_answer_criteria_matching():
    """Verify evaluation matches reference criteria and identifies pitfalls."""
    evaluator = BenchmarkEvaluator()

    ref_criteria = [
      "Leader leases or heartbeat read verification required to prevent stale reads",
      "Network partitions can isolate leader from quorum",
    ]
    pitfalls = ["Operates without quorum under partitions"]

    answer_good = (
        "In Raft, leader leases and heartbeat read verification are required to prevent "
        "stale reads when an isolated leader is separated from quorum by network partitions."
    )

    res = evaluator.evaluate_answer(
        task_id="test_t1",
        domain="Tech",
        mode="MEDHA",
        prompt="Explain Raft partitions",
        answer=answer_good,
        duration=5.0,
        token_usage={"total_tokens": 300},
        reference_criteria=ref_criteria,
        known_pitfalls=pitfalls,
        completion_status="SUCCESS",
    )

    assert res.success is True
    assert len(res.matched_criteria) == 2
    assert len(res.missed_criteria) == 0
    assert len(res.identified_pitfalls) == 0
    assert res.criteria_coverage_score == 1.0
    assert res.answer_metrics["answer_completeness"] == 1.0
    assert res.answer_metrics["task_completion"] == 1.0
    assert res.answer_metrics["factual_correctness"] is None
    assert res.evaluator_reasoning


def test_pitfall_avoidance_is_missing_when_case_has_no_pitfall_denominator():
    result = BenchmarkEvaluator().evaluate_answer(
        task_id="no-pitfalls",
        domain="Tech",
        mode="BASELINE",
        prompt="Task",
        answer="A successful response.",
        duration=1.0,
        token_usage={},
        reference_criteria=[],
        known_pitfalls=[],
        completion_status="SUCCESS",
    )

    assert result.answer_metrics["pitfall_avoidance"] is None


def test_nonempty_answer_without_explicit_completion_status_is_not_success():
    result = BenchmarkEvaluator().evaluate_answer(
        task_id="unknown-status",
        domain="Tech",
        mode="MEDHA",
        prompt="Prompt",
        answer="A detailed answer that matches the criteria.",
        duration=1.0,
        token_usage={},
        reference_criteria=["Detailed answer matches criteria"],
        known_pitfalls=[],
    )

    assert result.completion_status == "UNKNOWN"
    assert result.success is False
    assert result.quality_eligible is False
    assert result.answer


def test_symmetric_evaluator_uses_same_metric_shape_for_both_modes():
    evaluator = BenchmarkEvaluator()
    kwargs = {
        "task_id": "same_task",
        "domain": "Tech",
        "prompt": "Explain Raft",
        "answer": "Leader leases and quorum are discussed with an assumption and evidence source.",
        "duration": 1.0,
        "token_usage": {},
        "reference_criteria": ["Leader leases are required"],
        "known_pitfalls": [],
        "completion_status": "SUCCESS",
    }

    baseline = evaluator.evaluate_answer(mode="BASELINE", **kwargs)
    medha = evaluator.evaluate_answer(mode="MEDHA", **kwargs)

    assert set(baseline.answer_metrics) == set(medha.answer_metrics)
    assert baseline.answer_metrics["answer_completeness"] == medha.answer_metrics["answer_completeness"]
    assert baseline.answer_metrics["evidence_quality"] == medha.answer_metrics["evidence_quality"]


def _benchmark_runner(answer, config=None, failures=None, status="SUCCESS", success=True):
    class Runner:
        def __init__(self):
            self.calls = 0

        def benchmark_configuration(self):
            return config or {}

        async def run(self, task):
            self.calls += 1
            if failures and self.calls in failures:
                raise RuntimeError(f"trial failure {self.calls}")
            result = MagicMock()
            result.answer = answer
            result.final_answer = answer
            result.status = status
            result.success = success
            result.duration_seconds = 1.0
            result.token_usage = {
                "total_tokens": 100,
                "accounted_total_tokens": 100,
                "usage_source": "provider",
                "total_requests": 1,
            }
            result.error = None
            result.verification_report = {}
            result.claims = []
            return result

    return Runner()


@pytest.mark.asyncio
async def test_configuration_mismatch_is_rejected():
    baseline = _benchmark_runner("baseline", {"provider": "groq", "model": "model-a", "temperature": 0.2})
    medha = _benchmark_runner("medha", {"provider": "groq", "model": "model-b", "temperature": 0.2})
    evaluator = BenchmarkEvaluator(baseline_runner=baseline, medha_runner=medha)

    with pytest.raises(ValueError, match="incompatible"):
        await evaluator.run_benchmark(max_cases=1)


@pytest.mark.asyncio
async def test_repetitions_alternate_order_and_preserve_failed_trials(tmp_path):
    config = {
        "provider": "groq",
        "model": "same-model",
        "model_version": "v1",
        "temperature": 0.2,
        "input_token_limit": 1000,
        "output_token_limit": 100,
        "task_token_budget": 10000,
        "call_budget": 10,
    }
    baseline = _benchmark_runner("baseline", config, failures={2})
    medha = _benchmark_runner("medha", config)
    evaluator = BenchmarkEvaluator(
        dataset_path=tmp_path / "dataset.json",
        baseline_runner=baseline,
        medha_runner=medha,
    )
    evaluator.dataset_path.write_text(
        '[{"id":"t1","domain":"Tech","prompt":"Prompt","reference_criteria":[],"known_pitfalls":[]}]',
        encoding="utf-8",
    )
    evaluator.save_json_report = lambda data: BenchmarkEvaluator.save_json_report(
        evaluator, data, tmp_path / "report.json"
    )
    evaluator.save_csv_report = lambda b, m: BenchmarkEvaluator.save_csv_report(
        evaluator, b, m, tmp_path / "report.csv"
    )

    data = await evaluator.run_benchmark(max_cases=1, repetitions=3, experiment_id="exp_fixed")

    assert [trial["execution_order"] for trial in data["trial_records"]] == [
        ["BASELINE", "MEDHA"],
        ["MEDHA", "BASELINE"],
        ["BASELINE", "MEDHA"],
    ]
    assert len(data["trials"]) == 6
    failed = [trial for trial in data["trials"] if not trial["success"]]
    assert len(failed) == 1
    assert "trial failure 2" in failed[0]["error"]
    assert data["baseline_metrics"]["failure_count"] == 1
    assert data["experiment_id"] == "exp_fixed"
    assert data["trials"][0]["configuration"]["model"] == "same-model"
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.csv").exists()


def test_cost_labels_actual_estimated_and_unknown():
    evaluator = BenchmarkEvaluator(
        pricing={"groq:same-model": {"input_per_million_usd": 1.0, "output_per_million_usd": 2.0}}
    )
    common = {
        "task_id": "cost",
        "domain": "Tech",
        "mode": "BASELINE",
        "prompt": "Prompt",
        "answer": "Answer",
        "duration": 1.0,
        "reference_criteria": [],
        "known_pitfalls": [],
        "configuration": {"provider": "groq", "model": "same-model"},
        "completion_status": "SUCCESS",
    }
    actual = evaluator.evaluate_answer(
        **common,
        token_usage={"prompt_tokens": 100, "candidate_tokens": 50, "total_tokens": 150, "usage_source": "provider"},
    )
    estimated = evaluator.evaluate_answer(
        **common,
        token_usage={"prompt_tokens": 100, "candidate_tokens": 50, "accounted_total_tokens": 150, "usage_source": "estimated"},
    )
    unknown = evaluator.evaluate_answer(**common, token_usage={})

    assert actual.cost["status"] == "actual"
    assert estimated.cost["status"] == "estimated"
    assert unknown.cost["status"] == "unknown"
    assert unknown.cost["amount_usd"] is None


@pytest.mark.asyncio
async def test_benchmark_evaluator_run(tmp_path):
    """Verify BenchmarkEvaluator executes benchmark suite and exports CSV & JSON."""
    mock_baseline = MagicMock()
    mock_b_result = MagicMock()
    mock_b_result.answer = "Baseline Raft answer discussing leader leases and partitions"
    mock_b_result.duration_seconds = 2.0
    mock_b_result.token_usage = {"total_tokens": 200, "total_requests": 1}
    mock_b_result.error = None
    mock_baseline.run = AsyncMock(return_value=mock_b_result)

    mock_medha = MagicMock()
    mock_m_result = MedhaResult(
        execution_id="exec_bench_test",
        task="Test task",
        final_answer="MEDHA verified answer discussing leader leases, heartbeat verification, and quorum partitions",
        status="SUCCESS",
        duration_seconds=8.0,
        token_usage={"total_tokens": 1200, "total_requests": 6},
        claims=[{"claim_id": "c1", "claim_text": "Claim 1", "status": "SUPPORTED"}],
        verification_report={
            "total_claims": 2,
            "accepted_claims": [{"claim_id": "c1"}],
            "rejected_claims": [],
            "support_rate": 1.0,
            "unsupported_rate": 0.0,
            "repair_success_rate": 1.0,
        },
    )
    mock_medha.run = AsyncMock(return_value=mock_m_result)

    dataset_file = tmp_path / "test_dataset.json"
    dataset_file.write_text(
        """[
            {
                "id": "t1",
                "domain": "Tech",
                "prompt": "Evaluate Raft consensus",
                "reference_criteria": ["Leader leases or heartbeat read verification required to prevent stale reads"],
                "known_pitfalls": []
            }
        ]""",
        encoding="utf-8",
    )

    evaluator = BenchmarkEvaluator(
        dataset_path=dataset_file,
        baseline_runner=mock_baseline,
        medha_runner=mock_medha,
    )

    # Override report target paths to tmp_path
    json_path = tmp_path / "benchmark_report.json"
    csv_path = tmp_path / "benchmark_results.csv"

    evaluator.save_json_report = lambda data: evaluator.save_json_report.__wrapped__(evaluator, data, json_path) if hasattr(evaluator.save_json_report, "__wrapped__") else BenchmarkEvaluator.save_json_report(evaluator, data, json_path)
    evaluator.save_csv_report = lambda b, m: BenchmarkEvaluator.save_csv_report(evaluator, b, m, csv_path)

    comparison = await evaluator.run_benchmark(max_cases=1)

    assert "baseline_metrics" in comparison
    assert "medha_metrics" in comparison
    assert json_path.exists()
    assert csv_path.exists()

    md_summary = BenchmarkEvaluator.generate_markdown_summary(comparison)
    assert "Summary Metrics Comparison" in md_summary
    assert "Single-Agent Baseline" in md_summary
    assert "MEDHA Multi-Agent" in md_summary


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_status", "eligible"),
    [
        ("FAILED", "FAILED", False),
        ("CANCELLED", "CANCELLED", False),
        ("INTERRUPTED", "INTERRUPTED", False),
        ("PARTIAL_FAILURE", "PARTIAL_FAILURE", False),
        ("SUCCESS", "SUCCESS", True),
    ],
)
async def test_medha_status_controls_completion_and_quality_eligibility(
    tmp_path,
    status,
    expected_status,
    eligible,
):
    config = {
        "provider": "groq",
        "model": "same-model",
        "model_version": "v1",
        "temperature": 0.2,
        "input_token_limit": 1000,
        "output_token_limit": 100,
        "task_token_budget": 10000,
        "call_budget": 10,
    }
    answer = "Leader leases and heartbeat read verification prevent stale reads."
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        '[{"id":"status-case","domain":"Tech","prompt":"Prompt",'
        '"reference_criteria":["Leader leases prevent stale reads"],"known_pitfalls":[]}]',
        encoding="utf-8",
    )
    evaluator = BenchmarkEvaluator(
        dataset_path=dataset_path,
        baseline_runner=_benchmark_runner(answer, config),
        medha_runner=_benchmark_runner(answer, config, status=status),
    )
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    evaluator.save_json_report = lambda data: BenchmarkEvaluator.save_json_report(
        evaluator, data, json_path
    )
    evaluator.save_csv_report = lambda baseline, medha: BenchmarkEvaluator.save_csv_report(
        evaluator, baseline, medha, csv_path
    )

    report = await evaluator.run_benchmark(max_cases=1, experiment_id=f"exp_{status}")
    medha_trial = next(trial for trial in report["trials"] if trial["mode"] == "MEDHA")

    assert medha_trial["completion_status"] == expected_status
    assert medha_trial["success"] is eligible
    assert medha_trial["quality_eligible"] is eligible
    assert medha_trial["answer"] == answer
    assert medha_trial["criteria_coverage_score"] == 1.0
    assert report["medha_metrics"]["failure_count"] == (0 if eligible else 1)
    assert report["medha_metrics"]["reference_criteria_coverage"] == (1.0 if eligible else 0.0)
    assert report["medha_metrics"]["answer_metrics"]["answer_completeness"] == (
        1.0 if eligible else None
    )
    csv_text = csv_path.read_text(encoding="utf-8")
    assert expected_status in csv_text
    assert answer in csv_text


@pytest.mark.asyncio
async def test_success_with_empty_answer_is_an_unsuccessful_empty_output_trial(tmp_path):
    config = {
        "provider": "groq",
        "model": "same-model",
        "model_version": "v1",
        "temperature": 0.2,
        "input_token_limit": 1000,
        "output_token_limit": 100,
        "task_token_budget": 10000,
        "call_budget": 10,
    }
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        '[{"id":"empty-case","domain":"Tech","prompt":"Prompt",'
        '"reference_criteria":["Leader leases prevent stale reads"],"known_pitfalls":[]}]',
        encoding="utf-8",
    )
    evaluator = BenchmarkEvaluator(
        dataset_path=dataset_path,
        baseline_runner=_benchmark_runner("baseline answer", config),
        medha_runner=_benchmark_runner("", config, status="SUCCESS"),
    )
    evaluator.save_json_report = lambda data: None
    evaluator.save_csv_report = lambda baseline, medha: None

    report = await evaluator.run_benchmark(max_cases=1)
    medha_trial = next(trial for trial in report["trials"] if trial["mode"] == "MEDHA")

    assert medha_trial["completion_status"] == "EMPTY_OUTPUT"
    assert medha_trial["success"] is False
    assert medha_trial["quality_eligible"] is False
    assert report["medha_metrics"]["failure_count"] == 1


@pytest.mark.asyncio
async def test_baseline_failure_is_not_inferred_success_from_nonempty_answer(tmp_path):
    config = {
        "provider": "groq",
        "model": "same-model",
        "model_version": "v1",
        "temperature": 0.2,
        "input_token_limit": 1000,
        "output_token_limit": 100,
        "task_token_budget": 10000,
        "call_budget": 10,
    }
    answer = "Leader leases prevent stale reads, but this run failed."
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        '[{"id":"baseline-failure","domain":"Tech","prompt":"Prompt",'
        '"reference_criteria":["Leader leases prevent stale reads"],"known_pitfalls":[]}]',
        encoding="utf-8",
    )
    evaluator = BenchmarkEvaluator(
        dataset_path=dataset_path,
        baseline_runner=_benchmark_runner(answer, config, success=False),
        medha_runner=_benchmark_runner("Valid response", config),
    )
    evaluator.save_json_report = lambda data: None
    evaluator.save_csv_report = lambda baseline, medha: None

    report = await evaluator.run_benchmark(max_cases=1)
    baseline_trial = next(trial for trial in report["trials"] if trial["mode"] == "BASELINE")

    assert baseline_trial["completion_status"] == "FAILED"
    assert baseline_trial["success"] is False
    assert baseline_trial["quality_eligible"] is False
    assert baseline_trial["answer"] == answer
    assert baseline_trial["criteria_coverage_score"] == 1.0
    assert report["baseline_metrics"]["failure_count"] == 1
    assert report["baseline_metrics"]["reference_criteria_coverage"] == 0.0

