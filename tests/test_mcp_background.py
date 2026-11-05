
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.orchestrator import MedhaResult
from mcp_server.registry import ExecutionRegistry
from mcp_server.tools import (
    get_execution_result_impl,
    get_execution_status_impl,
    get_verification_report_impl,
    start_medha_task_impl,
    run_medha_task_impl,
)


def create_result():
    return MedhaResult(
        execution_id="exec_background_test",
        task="Test background execution",
        final_answer="Background task completed",
        status="SUCCESS",
        verification_report={
            "total_claims": 1,
            "accepted_claims": [{"claim_id": "c1"}],
            "rejected_claims": [],
            "support_rate": 1.0,
        },
        claims=[{"claim_id": "c1", "status": "SUPPORTED"}],
        evidence=[{"evidence_id": "e1"}],
    )


async def wait_for_status(registry, job_id, expected):
    for _ in range(100):
        result = await registry.get_job(job_id, include_result=True)

        if result["status"] == expected:
            return result

        await asyncio.sleep(0.01)

    raise AssertionError(f"Job did not reach {expected}.")


@pytest.mark.asyncio
async def test_background_submission_and_result(tmp_path):
    registry = ExecutionRegistry(logs_dir=tmp_path)

    mock_orchestrator = MagicMock()
    mock_orchestrator.run_task = AsyncMock(return_value=create_result())

    submitted = await start_medha_task_impl(
        "Test background task",
        registry=registry,
        orchestrator_factory=lambda: mock_orchestrator,
    )

    assert submitted["job_id"].startswith("mcp_")

    completed = await wait_for_status(
        registry,
        submitted["job_id"],
        "COMPLETED",
    )

    assert completed["medha_execution_id"] == "exec_background_test"

    result = await get_execution_result_impl(
        submitted["job_id"],
        registry=registry,
    )

    assert result["result"]["final_answer"] == "Background task completed"


@pytest.mark.asyncio
async def test_background_status(tmp_path):
    registry = ExecutionRegistry(logs_dir=tmp_path)

    mock_orchestrator = MagicMock()
    mock_orchestrator.run_task = AsyncMock(return_value=create_result())

    submitted = await start_medha_task_impl(
        "Status test",
        registry=registry,
        orchestrator_factory=lambda: mock_orchestrator,
    )

    await wait_for_status(registry, submitted["job_id"], "COMPLETED")

    status = await get_execution_status_impl(
        submitted["job_id"],
        registry=registry,
    )

    assert status["job_id"] == submitted["job_id"]
    assert status["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_verification_report_by_job_id(tmp_path):
    registry = ExecutionRegistry(logs_dir=tmp_path)

    mock_orchestrator = MagicMock()
    mock_orchestrator.run_task = AsyncMock(return_value=create_result())

    submitted = await start_medha_task_impl(
        "Verification test",
        registry=registry,
        orchestrator_factory=lambda: mock_orchestrator,
    )

    await wait_for_status(registry, submitted["job_id"], "COMPLETED")

    report = await get_verification_report_impl(
        submitted["job_id"],
        registry=registry,
    )

    assert report["total_claims"] == 1
    assert report["support_rate"] == 1.0
    assert len(report["all_claims"]) == 1
    assert len(report["evidence"]) == 1


@pytest.mark.asyncio
async def test_invalid_background_task(tmp_path):
    registry = ExecutionRegistry(logs_dir=tmp_path)

    result = await start_medha_task_impl(
        "   ",
        registry=registry,
    )

    assert result["status"] == "REJECTED"
    assert "error" in result


@pytest.mark.asyncio
async def test_capacity_limit_and_concurrent_submissions(tmp_path):
    registry = ExecutionRegistry(logs_dir=tmp_path, max_active_jobs=2)
    started = asyncio.Event()
    release = asyncio.Event()

    async def runner():
        started.set()
        await release.wait()
        return create_result()

    submissions = await asyncio.gather(
        *[registry.submit(f"Task {index}", runner) for index in range(2)]
    )
    assert len({item["job_id"] for item in submissions}) == 2
    await started.wait()

    with pytest.raises(RuntimeError, match="capacity reached"):
        await registry.submit("Task over capacity", runner)

    release.set()
    for submission in submissions:
        await wait_for_status(registry, submission["job_id"], "COMPLETED")


@pytest.mark.asyncio
async def test_failed_background_execution_is_persisted(tmp_path):
    registry = ExecutionRegistry(logs_dir=tmp_path)

    async def runner():
        raise RuntimeError("background failure")

    submitted = await registry.submit("Failure test", runner)
    failed = await wait_for_status(registry, submitted["job_id"], "FAILED")

    assert "RuntimeError: background failure" in failed["error"]
    recovered = ExecutionRegistry(logs_dir=tmp_path)
    recovered_job = await recovered.get_job(submitted["job_id"])
    assert recovered_job["status"] == "FAILED"
    assert "background failure" in recovered_job["error"]


@pytest.mark.asyncio
async def test_cancellation_is_graceful_and_isolated(tmp_path):
    registry = ExecutionRegistry(logs_dir=tmp_path)
    started = asyncio.Event()

    async def runner():
        started.set()
        await asyncio.Event().wait()

    submitted = await registry.submit("Cancel test", runner)
    await started.wait()
    cancelled = await registry.cancel(submitted["job_id"])

    assert cancelled["status"] == "CANCELLED"
    final = await wait_for_status(registry, submitted["job_id"], "CANCELLED")
    assert "cancelled" in final["error"].lower()


@pytest.mark.asyncio
async def test_completed_result_and_interrupted_job_recover_after_restart(tmp_path):
    registry = ExecutionRegistry(logs_dir=tmp_path)

    async def runner():
        return create_result()

    completed = await registry.submit("Recovery test", runner)
    await wait_for_status(registry, completed["job_id"], "COMPLETED")

    restarted = ExecutionRegistry(logs_dir=tmp_path)
    recovered = await restarted.get_job(completed["job_id"], include_result=True)
    assert recovered["status"] == "COMPLETED"
    assert recovered["result"]["execution_id"] == "exec_background_test"

    active_path = tmp_path / "mcp_execution_mcp_active.json"
    active_path.write_text(
        json.dumps({
            "job_id": "mcp_active",
            "task": "Interrupted task",
            "status": "RUNNING",
            "created_at": "2026-11-05T00:00:00+00:00",
            "started_at": "2026-11-05T00:00:01+00:00",
            "updated_at": "2026-11-05T00:00:01+00:00",
            "result": None,
            "error": None,
        }),
        encoding="utf-8",
    )
    interrupted = await restarted.get_job("mcp_active")
    assert interrupted["status"] == "INTERRUPTED"
    assert "not automatically resumed" in interrupted["error"]


@pytest.mark.asyncio
async def test_verification_report_returns_complete_payload(tmp_path):
    registry = ExecutionRegistry(logs_dir=tmp_path)
    result = create_result()
    result.verification_report.update({
        "unverifiable_claims": [{"claim_id": "c2"}],
        "unreviewed_claims": [{"claim_id": "c3"}],
        "needs_revision_claims": [{"claim_id": "c4"}],
        "repair_history": [{"claim_id": "c4", "status": "NEEDS_REVISION"}],
    })
    result.review_decisions = [{"claim_id": "c1", "decision": "ACCEPT"}]

    async def runner():
        return result

    submitted = await registry.submit("Report test", runner)
    await wait_for_status(registry, submitted["job_id"], "COMPLETED")
    report = await get_verification_report_impl(submitted["job_id"], registry=registry)

    assert report["accepted_claims"]
    assert report["unverifiable_claims"]
    assert report["unreviewed_claims"]
    assert report["needs_revision_claims"]
    assert report["repair_history"]
    assert report["review_decisions"]
    assert report["evidence"]


@pytest.mark.asyncio
async def test_tool_option_validation(tmp_path):
    registry = ExecutionRegistry(logs_dir=tmp_path)
    invalid_background = await start_medha_task_impl(
        "valid task",
        max_concurrency=0,
        registry=registry,
    )
    invalid_sync = await run_medha_task_impl(
        "valid task",
        max_repair_iterations=-1,
    )

    assert invalid_background["status"] == "REJECTED"
    assert invalid_sync["status"] == "FAILED"
