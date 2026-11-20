import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from core.orchestrator import MedhaOrchestrator, MedhaResult
from mcp_server.server import (
    get_execution_status,
    get_verification_report,
    run_baseline,
    run_medha_task,
)
from mcp_server.tools import (
    cache_execution_result,
    get_execution_status_impl,
    get_verification_report_impl,
    run_baseline_impl,
    run_medha_task_impl,
)


def _create_sample_medha_result(exec_id: str = "exec_test_mcp") -> MedhaResult:
    return MedhaResult(
        execution_id=exec_id,
        task="Test task for MCP",
        final_answer="Final synthesized response for MCP",
        status="COMPLETED",
        storage_mode="persistent",
        duration_seconds=5.2,
        token_usage={"total_tokens": 500},
        generated_agents=[{"agent_id": "a1", "role": "Architect", "status": "COMPLETED"}],
        task_results=[{"subtask_id": "T1", "role": "Architect", "output": "Done"}],
        claims=[{"claim_id": "c1", "claim_text": "Claim 1", "status": "SUPPORTED"}],
        evidence=[{"evidence_id": "e1", "supporting_content": "Ev 1"}],
        review_decisions=[{"claim_id": "c1", "decision": "ACCEPT"}],
        repair_history=[],
        unresolved_questions=["Open risk 1"],
        verification_report={
            "total_claims": 1,
            "accepted_claims": [{"claim_id": "c1"}],
            "rejected_claims": [],
            "support_rate": 1.0,
            "repair_success_rate": 0.0,
        },
        workflow_stages=[{"stage": "COMPLETED"}],
    )


@pytest.mark.asyncio
async def test_run_medha_task_impl():
    """Verify run_medha_task_impl coordinates with orchestrator and returns summary."""
    mock_orch = MagicMock()
    mock_result = _create_sample_medha_result("exec_123")
    mock_orch.run_task = AsyncMock(return_value=mock_result)

    summary = await run_medha_task_impl("Design system", orchestrator=mock_orch)

    assert summary["execution_id"] == "exec_123"
    assert summary["status"] == "COMPLETED"
    assert summary["final_answer"] == "Final synthesized response for MCP"
    assert summary["verification_status"]["support_rate"] == 1.0


@pytest.mark.asyncio
async def test_ephemeral_foreground_result_is_not_cached_or_written(monkeypatch, tmp_path):
    monkeypatch.setattr("mcp_server.tools.MEDHA_STORAGE_MODE", "ephemeral")
    monkeypatch.setattr("mcp_server.tools.LOGS_DIR", tmp_path)
    mock_orchestrator = MagicMock()
    result = _create_sample_medha_result("exec_ephemeral_foreground")
    result.storage_mode = "ephemeral"
    mock_orchestrator.run_task = AsyncMock(return_value=result)

    response = await run_medha_task_impl(
        "Synthetic private MCP task",
        orchestrator=mock_orchestrator,
    )

    assert response["status"] == "COMPLETED"
    assert response["storage_mode"] == "ephemeral"
    assert await get_execution_status_impl("exec_ephemeral_foreground") == {
        "error": "Execution ID 'exec_ephemeral_foreground' not found."
    }
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_get_execution_status_and_verification_report():
    """Verify get_execution_status and get_verification_report query cached executions."""
    sample_res = _create_sample_medha_result("exec_query_test")
    cache_execution_result(sample_res)

    status_data = await get_execution_status_impl("exec_query_test")
    assert status_data["execution_id"] == "exec_query_test"
    assert status_data["status"] == "COMPLETED"
    assert len(status_data["agent_statuses"]) == 1

    report_data = await get_verification_report_impl("exec_query_test")
    assert report_data["execution_id"] == "exec_query_test"
    assert report_data["total_claims"] == 1
    assert report_data["support_rate"] == 1.0


@pytest.mark.asyncio
async def test_run_baseline_impl():
    """Verify run_baseline_impl executes single-agent baseline."""
    mock_baseline = MagicMock()
    baseline_result = MagicMock()
    baseline_result.model_dump.return_value = {
        "task": "Test baseline",
        "answer": "Direct answer from single agent",
        "duration_seconds": 1.5,
        "token_usage": {"total_tokens": 100},
        "model": "gemini-3.8-flash",
    }
    mock_baseline.run = AsyncMock(return_value=baseline_result)

    res = await run_baseline_impl("Test baseline", baseline_runner=mock_baseline)
    assert res["answer"] == "Direct answer from single agent"


@pytest.mark.asyncio
async def test_mcp_tool_functions_return_valid_json():
    """Verify top-level MCP tool wrapper functions return valid JSON strings."""
    sample_res = _create_sample_medha_result("exec_mcp_wrapper")
    cache_execution_result(sample_res)

    status_json = await get_execution_status("exec_mcp_wrapper")
    status_dict = json.loads(status_json)
    assert status_dict["execution_id"] == "exec_mcp_wrapper"

    report_json = await get_verification_report("exec_mcp_wrapper")
    report_dict = json.loads(report_json)
    assert report_dict["execution_id"] == "exec_mcp_wrapper"
    assert report_dict["total_claims"] == 1

    # Missing ID returns error in JSON
    missing_json = await get_execution_status("non_existent_id_404")
    missing_dict = json.loads(missing_json)
    assert "error" in missing_dict

