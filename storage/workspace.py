from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from config import settings

logger = logging.getLogger("medha.storage")


@dataclass(frozen=True)
class CleanupResult:
    workspace_id: str
    mode: str
    removed: bool
    verified: bool
    error: Optional[str] = None


class ExecutionWorkspace:
    """One execution's private storage namespace and cleanup boundary."""

    STATE_FILENAME = ".medha-workspace.json"

    def __init__(
        self,
        path: Path,
        workspace_id: str,
        execution_id: str,
        mode: str,
        ttl_seconds: int,
    ) -> None:
        if mode not in {"ephemeral", "persistent"}:
            raise ValueError("Storage mode must be 'ephemeral' or 'persistent'.")
        self.path = path.resolve()
        self.workspace_id = workspace_id
        self.execution_id = execution_id
        self.mode = mode
        self.ttl_seconds = ttl_seconds
        self._closed = False
        self.cleanup_result: Optional[CleanupResult] = None
        self._closers: list[Callable[[], Any]] = []
        self._lock = threading.RLock()
        self.path.mkdir(parents=True, exist_ok=False)
        self._restrict_permissions(self.path)
        now = _utc_now()
        self._state = {
            "schema_version": 1,
            "workspace_id": workspace_id,
            "execution_id": execution_id,
            "owner_pid": os.getpid(),
            "storage_mode": mode,
            "state": "ACTIVE",
            "created_at": now,
            "updated_at": now,
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat(),
        }
        self._write_state()

    @staticmethod
    def _restrict_permissions(path: Path) -> None:
        if os.name != "nt":
            try:
                path.chmod(0o700)
            except OSError:
                pass

    @property
    def ephemeral(self) -> bool:
        return self.mode == "ephemeral"

    def subdirectory(self, name: str) -> Path:
        """Create and return a named workspace subdirectory without allowing traversal."""
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Workspace subdirectory must be a single safe path component.")
        target = self.path / name
        target.mkdir(parents=True, exist_ok=True)
        self._restrict_permissions(target)
        return target

    def register_closer(self, closer: Callable[[], Any]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot attach a resource to a closed execution workspace.")
            self._closers.append(closer)

    def heartbeat(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._state["updated_at"] = _utc_now()
            self._state["expires_at"] = (
                datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds)
            ).isoformat()
            self._write_state()

    def _write_state(self) -> None:
        state_path = self.path / self.STATE_FILENAME
        temporary = state_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(self._state, file, separators=(",", ":"))
        os.replace(temporary, state_path)
        if os.name != "nt":
            try:
                state_path.chmod(0o600)
            except OSError:
                pass

    def close(self) -> CleanupResult:
        """Close resources and either erase ephemeral data or retain it by explicit policy."""
        with self._lock:
            if self._closed:
                if self.cleanup_result is not None:
                    return self.cleanup_result
                exists = self.path.exists()
                return CleanupResult(self.workspace_id, self.mode, not exists, not exists)
            self._closed = True
            errors: list[str] = []
            for closer in reversed(self._closers):
                try:
                    closer()
                except Exception as error:
                    errors.append(type(error).__name__)
            self._closers.clear()

            if self.mode == "persistent":
                self._state["state"] = "COMPLETED"
                self._state["updated_at"] = _utc_now()
                self._state["expires_at"] = (
                    datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds)
                ).isoformat()
                try:
                    self._write_state()
                except OSError as error:
                    errors.append(type(error).__name__)
                self.cleanup_result = CleanupResult(
                    self.workspace_id,
                    self.mode,
                    removed=False,
                    verified=self.path.is_dir(),
                    error=", ".join(errors) or None,
                )
                return self.cleanup_result

            try:
                shutil.rmtree(self.path)
            except FileNotFoundError:
                pass
            except OSError as error:
                errors.append(type(error).__name__)
            removed = not self.path.exists()
            verified = removed and not errors
            if not removed and not errors:
                errors.append("WorkspaceStillExists")
            self.cleanup_result = CleanupResult(
                self.workspace_id,
                self.mode,
                removed=removed,
                verified=verified,
                error=", ".join(errors) or None,
            )
            return self.cleanup_result

    def __enter__(self) -> "ExecutionWorkspace":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class ExecutionWorkspaceManager:
    """Creates isolated workspaces and sweeps abandoned or expired workspaces."""

    def __init__(
        self,
        mode: Optional[str] = None,
        ephemeral_root: Optional[Path] = None,
        persistent_root: Optional[Path] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        self.mode = (mode or settings.MEDHA_STORAGE_MODE).strip().lower()
        if self.mode not in {"ephemeral", "persistent"}:
            raise ValueError("MEDHA_STORAGE_MODE must be 'ephemeral' or 'persistent'.")
        self._custom_roots = ephemeral_root is not None or persistent_root is not None
        self.ephemeral_root = Path(ephemeral_root or settings.MEDHA_EPHEMERAL_ROOT).resolve()
        self.persistent_root = Path(persistent_root or settings.MEDHA_PERSISTENT_ROOT).resolve()
        default_ttl = (
            settings.MEDHA_WORKSPACE_TTL
            if self.mode == "ephemeral"
            else settings.MEDHA_PERSISTENT_RETENTION_DAYS * 24 * 60 * 60
        )
        self.ttl_seconds = int(ttl_seconds if ttl_seconds is not None else default_ttl)
        if self.ttl_seconds < 1:
            raise ValueError("Workspace TTL must be at least one second.")
        self._swept = False
        self._watchdog_task: Optional[asyncio.Task] = None
        self._active_workspaces: dict[str, ExecutionWorkspace] = {}
        self.last_cleanup_report = {"removed": 0, "failed": 0, "active_or_unexpired": 0}

    @property
    def root(self) -> Path:
        return self.ephemeral_root if self.mode == "ephemeral" else self.persistent_root

    def create(self, execution_id: str) -> ExecutionWorkspace:
        if not execution_id or not execution_id.strip():
            raise ValueError("Execution ID is required for a workspace.")
        self.root.mkdir(parents=True, exist_ok=True)
        ExecutionWorkspace._restrict_permissions(self.root)
        if not self._swept:
            self.last_cleanup_report = self.cleanup_abandoned()
            self._swept = True
        workspace_id = uuid.uuid4().hex
        workspace = ExecutionWorkspace(
            path=self.root / workspace_id,
            workspace_id=workspace_id,
            execution_id=execution_id,
            mode=self.mode,
            ttl_seconds=self.ttl_seconds,
        )
        self._active_workspaces[workspace_id] = workspace
        workspace.register_closer(lambda: self._active_workspaces.pop(workspace_id, None))
        return workspace

    def cleanup_abandoned(self) -> dict[str, int]:
        """Remove dead-owner ephemeral workspaces and expired persistent workspaces."""
        removed = failed = active = 0
        for root, mode in (
            (self.ephemeral_root, "ephemeral"),
            (self.persistent_root, "persistent"),
        ):
            if not root.is_dir():
                continue
            for child in root.iterdir():
                if (
                    not child.is_dir()
                    or child.is_symlink()
                    or re.fullmatch(r"[0-9a-f]{32}", child.name) is None
                ):
                    continue
                state_path = child / ExecutionWorkspace.STATE_FILENAME
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    try:
                        if time_since_modified(child) >= self.ttl_seconds:
                            shutil.rmtree(child)
                            removed += 1
                    except OSError:
                        failed += 1
                    continue

                owner_alive = _pid_alive(state.get("owner_pid"))
                expires_at = _parse_timestamp(state.get("expires_at"))
                expired = expires_at is None or expires_at <= datetime.now(timezone.utc)
                abandoned = (
                    state.get("state") == "ACTIVE"
                    and (not owner_alive or (mode == "ephemeral" and expired))
                )
                persistent_expired = mode == "persistent" and expired and state.get("state") == "COMPLETED"
                if abandoned or persistent_expired:
                    try:
                        shutil.rmtree(child)
                        removed += 1
                    except OSError:
                        failed += 1
                else:
                    active += 1
        if not self._custom_roots:
            removed += self._cleanup_persistent_execution_logs()
        return {"removed": removed, "failed": failed, "active_or_unexpired": active}

    @staticmethod
    def _cleanup_persistent_execution_logs() -> int:
        """Apply retention only to MEDHA execution artifacts, never unrelated log files."""
        cutoff = datetime.now(timezone.utc).timestamp() - settings.MEDHA_PERSISTENT_RETENTION_DAYS * 86400
        removed = 0
        for pattern in ("execution_*.json", "baseline_*.json", "mcp_execution_*.json"):
            for path in settings.LOGS_DIR.glob(pattern):
                try:
                    if path.is_file() and not path.is_symlink() and path.stat().st_mtime < cutoff:
                        path.unlink()
                        removed += 1
                except OSError:
                    continue
        return removed

    def start_watchdog(self, interval_seconds: Optional[int] = None) -> asyncio.Task:
        if interval_seconds is None:
            interval_seconds = max(1, min(60, self.ttl_seconds // 3))
        if interval_seconds < 1:
            raise ValueError("Watchdog interval must be positive.")
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(
                self._watch(interval_seconds),
                name="medha-workspace-watchdog",
            )
        return self._watchdog_task

    async def stop_watchdog(self) -> None:
        task = self._watchdog_task
        self._watchdog_task = None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _watch(self, interval_seconds: int) -> None:
        try:
            while True:
                await asyncio.sleep(interval_seconds)
                for workspace in list(self._active_workspaces.values()):
                    try:
                        workspace.heartbeat()
                    except OSError:
                        logger.warning("Workspace watchdog heartbeat could not be persisted.")
                report = self.cleanup_abandoned()
                if report["failed"]:
                    logger.warning("Workspace cleanup watchdog could not remove %d expired workspace(s).", report["failed"])
        except asyncio.CancelledError:
            raise


def cleanup_stale_workspaces(storage_mode: Optional[str] = None) -> dict[str, int]:
    """Run the startup sweep without creating a workspace or touching user source files."""
    return ExecutionWorkspaceManager(mode=storage_mode).cleanup_abandoned()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid < 1:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        kernel32.GetExitCodeProcess.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle.restype = ctypes.c_int
        access = 0x00100000 | 0x1000  # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION
        handle = kernel32.OpenProcess(access, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except (ProcessLookupError, OSError):
        return False


def time_since_modified(path: Path) -> float:
    return max(0.0, datetime.now(timezone.utc).timestamp() - path.stat().st_mtime)
