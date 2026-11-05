
import asyncio
import pytest

from mcp_server.registry import ExecutionRegistry


@pytest.mark.asyncio
async def test_submit_and_complete(tmp_path):
    registry = ExecutionRegistry(logs_dir=tmp_path)

    async def runner():
        await asyncio.sleep(0.01)
        return {
            "execution_id": "exec_test_1",
            "status": "SUCCESS",
            "final_answer": "Test completed",
        }

    submitted = await registry.submit("Test MEDHA task", runner)

    assert submitted["status"] == "QUEUED"
    assert submitted["job_id"].startswith("mcp_")

    for _ in range(50):
        result = await registry.get_job(submitted["job_id"])

        if result["status"] == "COMPLETED":
            break

        await asyncio.sleep(0.01)

    assert result["status"] == "COMPLETED"
    assert result["medha_execution_id"] == "exec_test_1"


@pytest.mark.asyncio
async def test_failed_execution(tmp_path):
    registry = ExecutionRegistry(logs_dir=tmp_path)

    async def runner():
        raise RuntimeError("Simulated failure")

    submitted = await registry.submit("Failure test", runner)

    for _ in range(50):
        result = await registry.get_job(submitted["job_id"])

        if result["status"] == "FAILED":
            break

        await asyncio.sleep(0.01)

    assert result["status"] == "FAILED"
    assert "Simulated failure" in result["error"]


@pytest.mark.asyncio
async def test_persistent_result_recovery(tmp_path):
    registry = ExecutionRegistry(logs_dir=tmp_path)

    async def runner():
        return {
            "execution_id": "exec_persist",
            "status": "SUCCESS",
            "final_answer": "Persisted output",
        }

    submitted = await registry.submit("Persistence test", runner)

    for _ in range(50):
        result = await registry.get_job(
            submitted["job_id"],
            include_result=True,
        )

        if result["status"] == "COMPLETED":
            break

        await asyncio.sleep(0.01)

    new_registry = ExecutionRegistry(logs_dir=tmp_path)

    recovered = await new_registry.get_job(
        submitted["job_id"],
        include_result=True,
    )

    assert recovered["status"] == "COMPLETED"
    assert recovered["result"]["final_answer"] == "Persisted output"


@pytest.mark.asyncio
async def test_interrupted_execution_recovery(tmp_path):
    registry = ExecutionRegistry(logs_dir=tmp_path)

    async def runner():
        await asyncio.sleep(10)

    submitted = await registry.submit("Interrupted task", runner)

    for _ in range(50):
        result = await registry.get_job(submitted["job_id"])

        if result["status"] == "RUNNING":
            break

        await asyncio.sleep(0.01)

    # Simulate a new server process.
    new_registry = ExecutionRegistry(logs_dir=tmp_path)

    recovered = await new_registry.get_job(submitted["job_id"])

    assert recovered["status"] == "INTERRUPTED"

    # Clean up the original background task.
    original_job = registry._jobs[submitted["job_id"]]
    original_job._task_handle.cancel()

    with pytest.raises(asyncio.CancelledError):
        await original_job._task_handle


@pytest.mark.asyncio
async def test_active_job_limit(tmp_path):
    registry = ExecutionRegistry(
        logs_dir=tmp_path,
        max_active_jobs=1,
    )

    gate = asyncio.Event()

    async def runner():
        await gate.wait()
        return {"status": "SUCCESS"}

    await registry.submit("First task", runner)

    with pytest.raises(RuntimeError, match="capacity reached"):
        await registry.submit("Second task", runner)

    gate.set()

    await asyncio.sleep(0.05)
