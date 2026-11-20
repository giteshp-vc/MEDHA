import asyncio
import json
import logging
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import settings
from core.orchestrator import MedhaOrchestrator, MedhaResult
from core.providers.gemini_client import GeminiClient
from models.provider_call import ProviderCallRecord, ProviderCallStatus
from storage.workspace import ExecutionWorkspace, ExecutionWorkspaceManager, _pid_alive
from utils.logger import ExecutionRecord
from utils.privacy import storage_mode_context
from config.logging_config import CredentialRedactingFormatter
from utils.token_tracker import TokenTracker
from main import run_single_task


def test_ephemeral_workspaces_are_isolated_and_removed(tmp_path):
    manager = ExecutionWorkspaceManager(
        mode="ephemeral",
        ephemeral_root=tmp_path / "ephemeral",
        persistent_root=tmp_path / "persistent",
        ttl_seconds=60,
    )
    first = manager.create("exec-one")
    second = manager.create("exec-two")

    assert first.path != second.path
    assert first.path.parent == second.path.parent
    assert first.subdirectory("uploads").is_dir()
    state = json.loads((first.path / first.STATE_FILENAME).read_text(encoding="utf-8"))
    assert state["execution_id"] == "exec-one"
    assert "task" not in state

    first_result = first.close()
    second_result = second.close()
    assert first_result.verified is True
    assert second_result.verified is True
    assert not first.path.exists()
    assert not second.path.exists()


def test_ephemeral_cleanup_is_idempotent(tmp_path):
    workspace = ExecutionWorkspaceManager(
        mode="ephemeral",
        ephemeral_root=tmp_path / "ephemeral",
        persistent_root=tmp_path / "persistent",
    ).create("exec-one")

    first = workspace.close()
    second = workspace.close()

    assert first.verified is True
    assert second.verified is True
    assert not workspace.path.exists()


def test_cleanup_reports_resource_close_failure_even_when_directory_is_removed(tmp_path):
    workspace = ExecutionWorkspaceManager(
        mode="ephemeral",
        ephemeral_root=tmp_path / "ephemeral",
        persistent_root=tmp_path / "persistent",
    ).create("exec-close-failure")

    def fail_closer():
        raise RuntimeError("private close detail")

    workspace.register_closer(fail_closer)
    result = workspace.close()

    assert result.removed is True
    assert result.verified is False
    assert result.error == "RuntimeError"
    assert not workspace.path.exists()


def test_persistent_workspace_is_explicit_and_retained(tmp_path):
    manager = ExecutionWorkspaceManager(
        mode="persistent",
        ephemeral_root=tmp_path / "ephemeral",
        persistent_root=tmp_path / "persistent",
        ttl_seconds=60,
    )
    workspace = manager.create("exec-persistent")
    result = workspace.close()

    state = json.loads((workspace.path / workspace.STATE_FILENAME).read_text(encoding="utf-8"))
    assert result.verified is True
    assert workspace.path.is_dir()
    assert state["state"] == "COMPLETED"


def test_startup_sweep_removes_abandoned_ephemeral_workspace(tmp_path):
    root = tmp_path / "ephemeral"
    manager = ExecutionWorkspaceManager(
        mode="ephemeral",
        ephemeral_root=root,
        persistent_root=tmp_path / "persistent",
        ttl_seconds=60,
    )
    abandoned = manager.create("exec-abandoned")
    state_path = abandoned.path / abandoned.STATE_FILENAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["owner_pid"] = 2_000_000_000
    state_path.write_text(json.dumps(state), encoding="utf-8")

    report = manager.cleanup_abandoned()

    assert report["removed"] == 1
    assert not abandoned.path.exists()
    abandoned._closed = True


def test_startup_sweep_removes_expired_ephemeral_workspace_even_if_pid_alive(tmp_path):
    manager = ExecutionWorkspaceManager(
        mode="ephemeral",
        ephemeral_root=tmp_path / "ephemeral",
        persistent_root=tmp_path / "persistent",
        ttl_seconds=60,
    )
    expired = manager.create("exec-expired-live")
    state_path = expired.path / expired.STATE_FILENAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["expires_at"] = "2000-01-01T00:00:00+00:00"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    report = manager.cleanup_abandoned()

    assert report["removed"] == 1
    assert not expired.path.exists()
    expired._closed = True


def test_windows_safe_process_liveness_check():
    assert _pid_alive(os.getpid()) is True
    assert _pid_alive(2_000_000_000) is False


def test_workspace_rejects_path_traversal(tmp_path):
    workspace = ExecutionWorkspaceManager(
        mode="ephemeral",
        ephemeral_root=tmp_path / "ephemeral",
        persistent_root=tmp_path / "persistent",
    ).create("exec-safe")
    try:
        workspace.subdirectory("..\\outside")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe workspace path was accepted")
    finally:
        workspace.close()


def test_execution_record_is_not_written_in_ephemeral_context(tmp_path):
    record = ExecutionRecord("exec-private", "sensitive task text")
    with storage_mode_context("ephemeral"):
        assert record.save(tmp_path / "should-not-exist.json") is None
    assert not (tmp_path / "should-not-exist.json").exists()


def test_execution_record_cannot_force_persistence_inside_ephemeral_context(tmp_path):
    record = ExecutionRecord("exec-private-override", "sensitive task text")
    target = tmp_path / "must-not-exist.json"

    with storage_mode_context("ephemeral"):
        assert record.save(target, persistent=True) is None

    assert not target.exists()


def test_ephemeral_file_log_formatter_omits_task_content():
    formatter = CredentialRedactingFormatter("%(levelname)s %(message)s")
    record = logging.LogRecord("medha.test", logging.INFO, "", 0, "private task text", (), None)

    with storage_mode_context("ephemeral"):
        rendered = formatter.format(record)

    assert "private task text" not in rendered
    assert "omitted in Ephemeral Mode" in rendered


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [("SUCCESS", "SUCCESS"), ("PARTIAL_FAILURE", "PARTIAL_FAILURE"), ("FAILED", "FAILED"), ("SYNTHESIS_EXCEPTION", "FAILED")],
)
@pytest.mark.asyncio
async def test_ephemeral_outcomes_do_not_write_execution_content(monkeypatch, tmp_path, outcome, expected_status):
    orchestrator = _mock_orchestrator(monkeypatch, tmp_path)
    private_text = "synthetic private execution marker"
    report_path = tmp_path / "logs" / "execution_private.json"
    log_path = tmp_path / "logs" / "medha-private.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    audit_logger = logging.getLogger("medha.test_ephemeral_audit")
    audit_logger.setLevel(logging.ERROR)
    audit_logger.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(CredentialRedactingFormatter("%(levelname)s %(message)s"))
    audit_logger.addHandler(handler)

    async def execute(task, **kwargs):
        record = ExecutionRecord(kwargs["execution_id"], task)
        record.record_error("synthesis", private_text, details=private_text)
        record.log_message({"content": private_text})
        assert record.save(report_path) is None
        try:
            raise RuntimeError(private_text)
        except RuntimeError:
            audit_logger.exception("Synthesis failed for %s", private_text)
        if outcome == "SYNTHESIS_EXCEPTION":
            raise RuntimeError(private_text)
        return MedhaResult(
            execution_id=kwargs["execution_id"],
            task=task,
            final_answer="safe result",
            status=outcome,
        )

    orchestrator._run_task = AsyncMock(side_effect=execute)
    try:
        result = await orchestrator.run_task(private_text)
    finally:
        handler.flush()
        audit_logger.removeHandler(handler)
        handler.close()

    assert result.status == expected_status
    assert result.cleanup_verified is True
    assert not report_path.exists()
    assert private_text not in log_path.read_text(encoding="utf-8")
    assert "Execution detail omitted in Ephemeral Mode." in log_path.read_text(encoding="utf-8")
    assert list((tmp_path / "ephemeral").glob("*")) == []


@pytest.mark.parametrize(
    ("fail_synthesis", "outcome"),
    [(False, "SUCCESS"), (False, "PARTIAL_FAILURE"), (True, "FAILED")],
)
@pytest.mark.asyncio
async def test_ephemeral_cli_success_and_failure_do_not_write_reports(
    monkeypatch, tmp_path, capsys, fail_synthesis, outcome
):
    monkeypatch.setattr(settings, "MEDHA_STORAGE_MODE", "ephemeral")
    monkeypatch.setattr("main.validate_settings", lambda: None)
    monkeypatch.setattr("utils.logger.LOGS_DIR", tmp_path / "logs")
    orchestrator = _mock_orchestrator(monkeypatch, tmp_path)
    diagnostic_snapshots = []
    from main import _print_execution_diagnostics as print_execution_diagnostics

    def capture_diagnostics(snapshot):
        assert list((tmp_path / "ephemeral").glob("*"))
        diagnostic_snapshots.append(snapshot)
        print_execution_diagnostics(snapshot)

    monkeypatch.setattr("main._print_execution_diagnostics", capture_diagnostics)

    async def execute(task, **kwargs):
        record = ExecutionRecord(kwargs["execution_id"], task)
        record.metadata["final_answer"] = "private CLI answer"
        record.save((tmp_path / "logs" / f"execution_{kwargs['execution_id']}.json"))
        if fail_synthesis:
            raise RuntimeError("private synthesis failure")
        return MedhaResult(
            execution_id=kwargs["execution_id"],
            task=task,
            final_answer="CLI answer",
            status=outcome,
            execution_status_reasons=["specialist_subtask_failed"] if outcome == "PARTIAL_FAILURE" else [],
            token_usage={
                "prompt_tokens": 9,
                "candidate_tokens": 4,
                "total_tokens": 13,
                "accounted_total_tokens": 13,
                "usage_source": "provider",
            },
            provider_call_records=[
                ProviderCallRecord(
                    attempt_number=1,
                    retry_number=0,
                    max_retry_attempts=2,
                    provider="groq",
                    model="mock-model",
                    stage="verification",
                    status=ProviderCallStatus.FAILED,
                    error_category="LLMRateLimitError",
                    http_status_code=429,
                    provider_error_code="synthetic-private-code-marker",
                    retry_classification="RETRYABLE",
                    sanitized_error="LLMRateLimitError (status 429)",
                ),
                ProviderCallRecord(
                    attempt_number=2,
                    retry_number=1,
                    max_retry_attempts=2,
                    provider="groq",
                    model="mock-model",
                    stage="verification",
                    status=ProviderCallStatus.SUCCEEDED,
                ),
            ],
        )

    orchestrator._run_task = AsyncMock(side_effect=execute)
    monkeypatch.setattr("main.MedhaOrchestrator", lambda **kwargs: orchestrator)

    await run_single_task("private CLI task")

    output = capsys.readouterr().out
    assert "private synthesis failure" not in output
    assert "private CLI task" not in output
    assert not list((tmp_path / "logs").glob("execution_*.json"))
    assert list((tmp_path / "ephemeral").glob("*")) == []
    assert len(diagnostic_snapshots) == 1
    assert diagnostic_snapshots[0]["status"] == ("FAILED" if fail_synthesis else outcome)
    if fail_synthesis:
        assert "Tokens (existing tracker totals): input=0, output=0, total=0" in output
        assert "succeeded=0, failed=0" in output
        assert "execution_setup_failed:RuntimeError" in output
    else:
        assert "Tokens (existing tracker totals): input=9, output=4, total=13" in output
        assert "succeeded=1, failed=1" in output
        assert "retried_attempts=1" in output
        assert "stage=verification" in output
        assert "http_status=429" in output
        assert "provider_code=N/A" in output
        assert "synthetic-private-code-marker" not in output
        assert "retry_classification=RETRYABLE" in output


def test_persistent_retention_sweep_removes_expired_workspace(tmp_path):
    manager = ExecutionWorkspaceManager(
        mode="persistent",
        ephemeral_root=tmp_path / "ephemeral",
        persistent_root=tmp_path / "persistent",
        ttl_seconds=1,
    )
    workspace = manager.create("exec-expired")
    workspace.close()
    state_path = workspace.path / workspace.STATE_FILENAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["expires_at"] = "2000-01-01T00:00:00+00:00"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    report = manager.cleanup_abandoned()

    assert report["removed"] == 1
    assert not workspace.path.exists()


def _mock_orchestrator(monkeypatch, root, persistent=False):
    monkeypatch.setattr(settings, "MEDHA_EPHEMERAL_ROOT", root / "ephemeral")
    monkeypatch.setattr(settings, "MEDHA_PERSISTENT_ROOT", root / "persistent")
    tracker = TokenTracker(name="workspace_orchestrator_test", essential_token_reserves={})
    client = GeminiClient(api_key="mock-key", client=MagicMock(), token_tracker=tracker)
    return MedhaOrchestrator(
        llm_client=client,
        storage_mode="persistent" if persistent else "ephemeral",
    )


@pytest.mark.asyncio
async def test_successful_retry_records_do_not_downgrade_execution_status(monkeypatch, tmp_path):
    orchestrator = _mock_orchestrator(monkeypatch, tmp_path)
    failed_attempt = ProviderCallRecord(
        attempt_number=1,
        retry_number=0,
        max_retry_attempts=2,
        provider="groq",
        status=ProviderCallStatus.FAILED,
        error_category="LLMServerError",
        http_status_code=503,
        retry_classification="RETRYABLE",
    )
    successful_retry = ProviderCallRecord(
        attempt_number=2,
        retry_number=1,
        max_retry_attempts=2,
        provider="groq",
        status=ProviderCallStatus.SUCCEEDED,
    )
    orchestrator._run_task = AsyncMock(side_effect=lambda task, **kwargs: MedhaResult(
        execution_id=kwargs["execution_id"],
        task=task,
        final_answer="answer",
        provider_call_records=[failed_attempt, successful_retry],
        token_usage={"prompt_tokens": 9, "candidate_tokens": 4, "total_tokens": 13},
    ))

    result = await orchestrator.run_task("retry succeeded")

    assert result.status == "SUCCESS"
    assert result.execution_status_reasons == []
    assert result.token_usage["total_tokens"] == 13
    assert [record.status for record in result.provider_call_records] == [
        ProviderCallStatus.FAILED,
        ProviderCallStatus.SUCCEEDED,
    ]


@pytest.mark.asyncio
async def test_orchestrator_cleans_ephemeral_workspace_after_success(monkeypatch, tmp_path):
    orchestrator = _mock_orchestrator(monkeypatch, tmp_path)
    orchestrator._run_task = AsyncMock(side_effect=lambda task, **kwargs: MedhaResult(
        execution_id=kwargs["execution_id"], task=task, final_answer="answer"
    ))

    result = await orchestrator.run_task("private task")

    assert result.final_answer == "answer"
    assert result.storage_mode == "ephemeral"
    assert result.cleanup_verified is True
    assert list((tmp_path / "ephemeral").glob("*")) == []


@pytest.mark.asyncio
async def test_orchestrator_cleans_ephemeral_workspace_after_exception(monkeypatch, tmp_path):
    orchestrator = _mock_orchestrator(monkeypatch, tmp_path)
    orchestrator._run_task = AsyncMock(side_effect=RuntimeError("private failure detail"))

    result = await orchestrator.run_task("private task")

    assert result.status == "FAILED"
    assert result.cleanup_verified is True
    assert result.token_usage["total_tokens"] == 0
    assert result.provider_call_records == []
    assert "private failure detail" not in result.final_answer
    assert list((tmp_path / "ephemeral").glob("*")) == []


@pytest.mark.asyncio
async def test_orchestrator_cleans_ephemeral_workspace_after_cancellation(monkeypatch, tmp_path):
    orchestrator = _mock_orchestrator(monkeypatch, tmp_path)
    orchestrator._run_task = AsyncMock(side_effect=asyncio.CancelledError())
    snapshots = []

    def capture_diagnostics(snapshot):
        if snapshot["diagnostic_phase"] == "before_cleanup":
            assert list((tmp_path / "ephemeral").glob("*"))
        else:
            assert list((tmp_path / "ephemeral").glob("*")) == []
        snapshots.append(snapshot)

    with pytest.raises(asyncio.CancelledError):
        await orchestrator.run_task("private task", diagnostics_callback=capture_diagnostics)

    assert snapshots[0]["diagnostic_phase"] == "before_cleanup"
    assert snapshots[0]["status"] == "INTERRUPTED"
    assert snapshots[0]["status_reasons"] == ["execution_cancelled"]
    assert snapshots[1]["diagnostic_phase"] == "after_cleanup"
    assert snapshots[1]["cleanup_mode"] == "ephemeral"
    assert snapshots[1]["cleanup_verified"] is True
    assert snapshots[1]["cleanup_removed"] is True
    assert list((tmp_path / "ephemeral").glob("*")) == []


@pytest.mark.asyncio
async def test_orchestrator_cleans_ephemeral_workspace_after_timeout(monkeypatch, tmp_path):
    orchestrator = _mock_orchestrator(monkeypatch, tmp_path)

    async def wait_forever(task, **kwargs):
        await asyncio.Event().wait()

    orchestrator._run_task = AsyncMock(side_effect=wait_forever)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(orchestrator.run_task("private timeout task"), timeout=0.05)

    assert list((tmp_path / "ephemeral").glob("*")) == []


@pytest.mark.asyncio
async def test_orchestrator_retains_only_when_persistent_mode_selected(monkeypatch, tmp_path):
    orchestrator = _mock_orchestrator(monkeypatch, tmp_path, persistent=True)
    snapshots = []
    orchestrator._run_task = AsyncMock(side_effect=lambda task, **kwargs: MedhaResult(
        execution_id=kwargs["execution_id"], task=task, final_answer="answer"
    ))

    def capture_diagnostics(snapshot):
        assert list((tmp_path / "persistent").glob("*"))
        snapshots.append(snapshot)

    result = await orchestrator.run_task("private task", diagnostics_callback=capture_diagnostics)

    assert result.storage_mode == "persistent"
    assert result.cleanup_verified is None
    assert snapshots[0]["status"] == "SUCCESS"
    retained = list((tmp_path / "persistent").glob("*"))
    assert len(retained) == 1
    state = json.loads((retained[0] / ExecutionWorkspace.STATE_FILENAME).read_text(encoding="utf-8"))
    assert state["state"] == "COMPLETED"


@pytest.mark.asyncio
async def test_ephemeral_context_overrides_persistent_orchestrator(monkeypatch, tmp_path):
    orchestrator = _mock_orchestrator(monkeypatch, tmp_path, persistent=True)
    orchestrator._run_task = AsyncMock(side_effect=lambda task, **kwargs: MedhaResult(
        execution_id=kwargs["execution_id"], task=task, final_answer="answer"
    ))

    with storage_mode_context("ephemeral"):
        result = await orchestrator.run_task("private task")

    assert result.storage_mode == "ephemeral"
    assert result.cleanup_verified is True
    assert list((tmp_path / "ephemeral").glob("*")) == []
    assert list((tmp_path / "persistent").glob("*")) == []


@pytest.mark.asyncio
async def test_orchestrator_reports_ephemeral_workspace_cleanup_failure(monkeypatch, tmp_path):
    orchestrator = _mock_orchestrator(monkeypatch, tmp_path)
    orchestrator._run_task = AsyncMock(side_effect=lambda task, **kwargs: MedhaResult(
        execution_id=kwargs["execution_id"], task=task, final_answer="answer"
    ))

    def fail_removal(path):
        raise PermissionError("private filesystem detail")

    monkeypatch.setattr("storage.workspace.shutil.rmtree", fail_removal)

    result = await orchestrator.run_task("private task")

    assert result.status == "PARTIAL_FAILURE"
    assert result.cleanup_verified is False
    assert result.cleanup_error == "PermissionError"
    assert result.error == "Execution completed, but ephemeral workspace cleanup could not be verified."
    assert "ephemeral_workspace_cleanup_unverified" in result.execution_status_reasons
    assert list((tmp_path / "ephemeral").glob("*"))
