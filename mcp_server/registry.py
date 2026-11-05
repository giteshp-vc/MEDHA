
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from config.settings import LOGS_DIR, MCP_MAX_ACTIVE_JOBS, MCP_MAX_RECORDS
from utils.helpers import generate_id


ACTIVE_STATUSES = {"QUEUED", "RUNNING"}
TERMINAL_STATUSES = {
    "COMPLETED",
    "PARTIAL_FAILURE",
    "FAILED",
    "CANCELLED",
    "INTERRUPTED",
}
VALID_STATUSES = ACTIVE_STATUSES | TERMINAL_STATUSES


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExecutionJob:
    job_id: str
    task: str
    status: str = "QUEUED"
    created_at: str = field(default_factory=utc_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    updated_at: str = field(default_factory=utc_now)
    medha_execution_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    _task_handle: Optional[asyncio.Task] = field(
        default=None, repr=False, compare=False
    )

    def to_dict(self, include_result: bool = True) -> Dict[str, Any]:
        data = {
            "job_id": self.job_id,
            "task": self.task,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "updated_at": self.updated_at,
            "medha_execution_id": self.medha_execution_id,
            "error": self.error,
        }

        if include_result:
            data["result"] = self.result

        return data


class ExecutionRegistry:
    """Tracks MCP task submissions and their MEDHA execution lifecycle."""

    def __init__(
        self,
        logs_dir: Path = LOGS_DIR,
        max_records: int = MCP_MAX_RECORDS,
        max_active_jobs: int = MCP_MAX_ACTIVE_JOBS,
    ):
        if max_records < 1:
            raise ValueError("max_records must be at least 1.")

        if max_active_jobs < 1:
            raise ValueError("max_active_jobs must be at least 1.")

        self.logs_dir = Path(logs_dir)
        self.max_records = max_records
        self.max_active_jobs = max_active_jobs

        self._jobs: Dict[str, ExecutionJob] = {}
        self._lock = asyncio.Lock()
        self._recovered = False

    def _record_path(self, job_id: str) -> Path:
        return self.logs_dir / f"mcp_execution_{job_id}.json"

    def _persist(self, job: ExecutionJob) -> None:
        """Atomically persist a job record."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        target = self._record_path(job.job_id)
        temporary = target.with_suffix(".tmp")

        with temporary.open("w", encoding="utf-8") as file:
            json.dump(job.to_dict(), file, indent=2)

        temporary.replace(target)

    @staticmethod
    def _set_status(job: ExecutionJob, status: str) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Unsupported MCP execution status: {status}")
        if job.status in TERMINAL_STATUSES and job.status != status:
            raise RuntimeError(
                f"Cannot transition terminal job {job.job_id} from {job.status} to {status}."
            )
        job.status = status
        job.updated_at = utc_now()

    def _recover_all_locked(self) -> None:
        """Load persisted jobs and mark active jobs interrupted after restart."""
        if self._recovered:
            return

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        for path in self.logs_dir.glob("mcp_execution_*.json"):
            job_id = path.stem.removeprefix("mcp_execution_")
            if job_id in self._jobs:
                continue
            job = self._load_from_disk(job_id)
            if job is not None:
                self._jobs[job.job_id] = job
        while len(self._jobs) > self.max_records:
            removable = [
                job for job in self._jobs.values()
                if job.status in TERMINAL_STATUSES and job._task_handle is None
            ]
            if not removable:
                break
            oldest = min(removable, key=lambda job: job.created_at)
            self._jobs.pop(oldest.job_id, None)
            try:
                self._record_path(oldest.job_id).unlink(missing_ok=True)
            except OSError:
                pass
        self._recovered = True

    def _serialize_result(self, result: Any) -> Dict[str, Any]:
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")

        if isinstance(result, dict):
            return result

        raise TypeError(
            "Execution runner must return a Pydantic model or dictionary."
        )

    async def _prune_locked(self) -> None:
        """Remove oldest terminal records when memory retention is exceeded."""
        while len(self._jobs) > self.max_records:
            removable = [
                job
                for job in self._jobs.values()
                if job.status in TERMINAL_STATUSES
                and job._task_handle is None
            ]

            if not removable:
                break

            oldest = min(removable, key=lambda job: job.created_at)
            self._jobs.pop(oldest.job_id, None)

            try:
                self._record_path(oldest.job_id).unlink(missing_ok=True)
            except OSError:
                # Keep the registry operational if cleanup fails.
                pass

    async def submit(
        self,
        task: str,
        runner: Callable[[], Awaitable[Any]],
    ) -> Dict[str, Any]:
        """Register a task and schedule its execution in the background."""
        if not task or not task.strip():
            raise ValueError("Task description cannot be empty.")

        if not callable(runner):
            raise TypeError("runner must be callable.")

        async with self._lock:
            self._recover_all_locked()
            active_count = sum(
                job.status in ACTIVE_STATUSES
                for job in self._jobs.values()
            )

            if active_count >= self.max_active_jobs:
                raise RuntimeError(
                    f"MCP execution capacity reached "
                    f"({self.max_active_jobs} active jobs)."
                )

            job_id = generate_id("mcp")
            job = ExecutionJob(job_id=job_id, task=task.strip())

            self._jobs[job_id] = job
            self._persist(job)

            job._task_handle = asyncio.create_task(
                self._execute(job_id, runner),
                name=f"medha-mcp-{job_id}",
            )

            await self._prune_locked()

            return job.to_dict(include_result=False)

    async def _execute(
        self,
        job_id: str,
        runner: Callable[[], Awaitable[Any]],
    ) -> None:
        """Run the MEDHA task and update its lifecycle."""
        async with self._lock:
            job = self._jobs.get(job_id)

            if job is None:
                return

            self._set_status(job, "RUNNING")
            job.started_at = utc_now()
            job.updated_at = job.started_at
            self._persist(job)

        try:
            raw_result = await runner()
            result = self._serialize_result(raw_result)

            medha_status = str(result.get("status", "COMPLETED")).upper()

            if medha_status in {"FAILED"}:
                final_status = "FAILED"
            elif medha_status == "PARTIAL_FAILURE":
                final_status = "PARTIAL_FAILURE"
            else:
                final_status = "COMPLETED"

            async with self._lock:
                job = self._jobs.get(job_id)

                if job is None:
                    return

                if job.status == "CANCELLED":
                    job._task_handle = None
                    job.updated_at = utc_now()
                    self._persist(job)
                    return

                job.result = result
                job.medha_execution_id = result.get("execution_id")
                self._set_status(job, final_status)
                job.finished_at = utc_now()
                job.updated_at = job.finished_at
                job._task_handle = None

                self._persist(job)
                await self._prune_locked()

        except asyncio.CancelledError:
            async with self._lock:
                job = self._jobs.get(job_id)

                if job is not None:
                    self._set_status(job, "CANCELLED")
                    job.error = "Execution was cancelled."
                    job.finished_at = utc_now()
                    job.updated_at = job.finished_at
                    job._task_handle = None
                    self._persist(job)

            raise

        except Exception as exc:
            async with self._lock:
                job = self._jobs.get(job_id)

                if job is not None:
                    if job.status == "CANCELLED":
                        job._task_handle = None
                        job.updated_at = utc_now()
                        self._persist(job)
                        return
                    self._set_status(job, "FAILED")
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.finished_at = utc_now()
                    job.updated_at = job.finished_at
                    job._task_handle = None

                    self._persist(job)
                    await self._prune_locked()

    def _load_from_disk(self, job_id: str) -> Optional[ExecutionJob]:
        """Recover a persisted job record."""
        path = self._record_path(job_id)

        if not path.is_file():
            return None

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            job = ExecutionJob(
                job_id=data["job_id"],
                task=data["task"],
                status=data["status"],
                created_at=data["created_at"],
                started_at=data.get("started_at"),
                finished_at=data.get("finished_at"),
                updated_at=data.get("updated_at", utc_now()),
                medha_execution_id=data.get("medha_execution_id"),
                result=data.get("result"),
                error=data.get("error"),
            )

            # A running task cannot survive a process restart.
            if job.status in ACTIVE_STATUSES:
                job.status = "INTERRUPTED"
                job.error = (
                    "MCP server restarted before this execution completed. "
                    "The original task was not automatically resumed."
                )
                job.finished_at = utc_now()
                job.updated_at = job.finished_at
                self._persist(job)

            return job

        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Unable to recover MCP job '{job_id}': {exc}"
            ) from exc

    async def get_job(
        self,
        job_id: str,
        include_result: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a job from memory or persistent storage."""
        if not job_id or not job_id.strip():
            return None

        async with self._lock:
            self._recover_all_locked()
            job = self._jobs.get(job_id)

            if job is None:
                job = self._load_from_disk(job_id)

                if job is None:
                    return None

                self._jobs[job_id] = job
                await self._prune_locked()

            return job.to_dict(include_result=include_result)

    async def list_jobs(self) -> list[Dict[str, Any]]:
        """List currently retained in-memory jobs."""
        async with self._lock:
            self._recover_all_locked()
            return [
                job.to_dict(include_result=False)
                for job in self._jobs.values()
            ]

    async def cancel(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Cancel a queued or running job without cancelling unrelated jobs."""
        async with self._lock:
            self._recover_all_locked()
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in TERMINAL_STATUSES:
                return job.to_dict(include_result=False)

            self._set_status(job, "CANCELLED")
            job.error = "Execution was cancelled by the MCP client."
            job.finished_at = utc_now()
            task_handle = job._task_handle
            job._task_handle = None
            self._persist(job)
            if task_handle is not None and not task_handle.done():
                task_handle.cancel()
            return job.to_dict(include_result=False)

    async def shutdown(self) -> None:
        """Cancel active jobs and await their cleanup during server shutdown."""
        async with self._lock:
            self._recover_all_locked()
            handles = [
                job._task_handle
                for job in self._jobs.values()
                if job._task_handle is not None and not job._task_handle.done()
            ]
            for handle in handles:
                handle.cancel()
        if handles:
            await asyncio.gather(*handles, return_exceptions=True)


# Shared registry used by the MEDHA MCP server.
execution_registry = ExecutionRegistry()
