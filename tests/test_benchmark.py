from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from benchmark.evaluator import BenchmarkEvaluator
from benchmark.metrics import (
    SystemMetrics,
    TaskEvaluationResult,
    compute_system_metrics,
)
from core.orchestrator import MedhaResult


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
    assert metrics.estimated_cost_usd > 0.0


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
    )

    assert res.success is True
    assert len(res.matched_criteria) == 2
    assert len(res.missed_criteria) == 0
    assert len(res.identified_pitfalls) == 0
    assert res.criteria_coverage_score == 1.0


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
        status="COMPLETED",
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

