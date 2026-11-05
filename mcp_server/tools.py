
import json
from typing import Any, Callable, Dict, Optional

from benchmark.baseline import SingleAgentBaseline
from config.settings import LOGS_DIR
from core.orchestrator import MedhaOrchestrator, MedhaResult
from mcp_server.registry import ExecutionRegistry, execution_registry
from utils.logger import get_logger

logger = get_logger("mcp_tools")

# Cache of completed in-memory executions.
_execution_cache: Dict[str, MedhaResult] = {}


def _validate_execution_options(task: str, max_concurrency: int, max_repair_iterations: int) -> Optional[str]:
    if not task or not task.strip():
        return "Task description cannot be empty."
    if not isinstance(max_concurrency, int) or isinstance(max_concurrency, bool) or max_concurrency < 1:
        return "max_concurrency must be a positive integer."
    if not isinstance(max_repair_iterations, int) or isinstance(max_repair_iterations, bool) or max_repair_iterations < 0:
        return "max_repair_iterations must be a non-negative integer."
    return None


def cache_execution_result(result: MedhaResult) -> None:
    """Store execution result in memory for legacy query tools."""
    _execution_cache[result.execution_id] = result


def get_cached_or_logged_execution(
    execution_id: str,
) -> Optional[Dict[str, Any]]:
    """Retrieve an execution record from cache or execution logs."""
    if execution_id in _execution_cache:
        return _execution_cache[execution_id].model_dump(mode="json")

    log_path = LOGS_DIR / f"execution_{execution_id}.json"

    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, ValueError) as exc:
            logger.warning("Failed to read execution log: %s", exc)

    return None


async def run_medha_task_impl(
    task: str,
    max_concurrency: int = 4,
    max_repair_iterations: int = 2,
    orchestrator: Optional[MedhaOrchestrator] = None,
) -> Dict[str, Any]:
    """Execute a task synchronously through MEDHA."""
    validation_error = _validate_execution_options(task, max_concurrency, max_repair_iterations)
    if validation_error:
        return {"error": validation_error, "status": "FAILED"}

    orch = orchestrator or MedhaOrchestrator(
        max_concurrency=max_concurrency,
        max_repair_iterations=max_repair_iterations,
    )

    result = await orch.run_task(task)
    cache_execution_result(result)

    rep = result.verification_report

    return {
        "execution_id": result.execution_id,
        "task": result.task,
        "final_answer": result.final_answer,
        "status": result.status,
        "duration_seconds": result.duration_seconds,
        "verification_status": {
            "total_claims": rep.get("total_claims", 0),
            "accepted_claims_count": len(rep.get("accepted_claims", [])),
            "rejected_claims_count": len(rep.get("rejected_claims", [])),
            "support_rate": rep.get("support_rate"),
            "repair_success_rate": rep.get("repair_success_rate", 0.0),
        },
        "token_usage": result.token_usage,
        "unresolved_questions": result.unresolved_questions,
    }


async def start_medha_task_impl(
    task: str,
    max_concurrency: int = 4,
    max_repair_iterations: int = 2,
    registry: Optional[ExecutionRegistry] = None,
    orchestrator_factory: Optional[Callable[[], MedhaOrchestrator]] = None,
) -> Dict[str, Any]:
    """Submit a MEDHA task for background execution."""
    validation_error = _validate_execution_options(task, max_concurrency, max_repair_iterations)
    if validation_error:
        return {
            "error": validation_error,
            "status": "REJECTED",
        }

    job_registry = registry or execution_registry

    async def runner():
        orch = (
            orchestrator_factory()
            if orchestrator_factory is not None
            else MedhaOrchestrator(
                max_concurrency=max_concurrency,
                max_repair_iterations=max_repair_iterations,
            )
        )

        result = await orch.run_task(task)
        cache_execution_result(result)
        return result

    try:
        job = await job_registry.submit(task, runner)
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "task": job["task"],
            "created_at": job["created_at"],
            "message": "MEDHA execution accepted for background processing.",
        }

    except (ValueError, RuntimeError, TypeError) as exc:
        logger.warning("Unable to submit MEDHA task: %s", exc)
        return {
            "error": str(exc),
            "status": "REJECTED",
        }


async def get_execution_status_impl(
    execution_id: str,
    registry: Optional[ExecutionRegistry] = None,
) -> Dict[str, Any]:
    """Retrieve live job status or a legacy MEDHA execution status."""
    if not execution_id:
        return {"error": "Execution ID must be provided."}

    job_registry = registry or execution_registry

    # First, check the MCP background-job registry.
    job = await job_registry.get_job(execution_id)

    if job is not None:
        return {
            "job_id": job["job_id"],
            "medha_execution_id": job["medha_execution_id"],
            "status": job["status"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
            "error": job["error"],
        }

    # Preserve compatibility with existing MEDHA execution IDs.
    data = get_cached_or_logged_execution(execution_id)

    if not data:
        return {"error": f"Execution ID '{execution_id}' not found."}

    return {
        "execution_id": execution_id,
        "status": data.get("status", "UNKNOWN"),
        "duration_seconds": data.get("duration_seconds"),
        "agent_statuses": [
            {
                "agent_id": agent.get("agent_id"),
                "role": agent.get("role"),
                "status": agent.get("status"),
            }
            for agent in data.get(
                "generated_agents",
                data.get("agent_records", []),
            )
        ],
        "workflow_stages": data.get("workflow_stages", []),
        "error": data.get("error"),
    }


async def get_execution_result_impl(
    job_id: str,
    registry: Optional[ExecutionRegistry] = None,
) -> Dict[str, Any]:
    """Retrieve the final MEDHA result for a background job."""
    if not job_id:
        return {"error": "Job ID must be provided."}

    job_registry = registry or execution_registry

    job = await job_registry.get_job(job_id, include_result=True)

    if job is None:
        return {"error": f"Job ID '{job_id}' not found."}

    if job["status"] in {"QUEUED", "RUNNING"}:
        return {
            "job_id": job_id,
            "status": job["status"],
            "message": "Execution is still in progress.",
        }

    if job["status"] == "INTERRUPTED":
        return {
            "job_id": job_id,
            "status": "INTERRUPTED",
            "error": job["error"],
        }

    if not job.get("result"):
        return {
            "job_id": job_id,
            "status": job["status"],
            "error": job["error"] or "No execution result is available.",
        }

    return {
        "job_id": job_id,
        "status": job["status"],
        "result": job["result"],
    }


async def get_verification_report_impl(
    execution_id: str,
    registry: Optional[ExecutionRegistry] = None,
) -> Dict[str, Any]:
    """Retrieve detailed verification information using a job or execution ID."""
    if not execution_id:
        return {"error": "Execution ID must be provided."}

    job_registry = registry or execution_registry

    # A job ID gives access to the complete persisted MedhaResult.
    job = await job_registry.get_job(execution_id, include_result=True)

    if job is not None:
        if not job.get("result"):
            return {
                "job_id": execution_id,
                "status": job["status"],
                "error": job["error"] or "Verification report is not available yet.",
            }

        data = job["result"]
    else:
        # Legacy execution-ID lookup.
        data = get_cached_or_logged_execution(execution_id)

    if not data:
        return {"error": f"Execution ID '{execution_id}' not found."}

    report = data.get("verification_report", {})

    return {
        "execution_id": data.get("execution_id", execution_id),
        "job_id": execution_id if job is not None else None,
        "status": data.get("status", job["status"] if job else "UNKNOWN"),
        "total_claims": report.get(
            "total_claims",
            len(data.get("claims", [])),
        ),
        "support_rate": report.get("support_rate"),
        "accepted_claims": report.get("accepted_claims", []),
        "rejected_claims": report.get("rejected_claims", []),
        "unverifiable_claims": report.get("unverifiable_claims", []),
        "unreviewed_claims": report.get("unreviewed_claims", []),
        "needs_revision_claims": report.get("needs_revision_claims", []),
        "all_claims": data.get("claims", []),
        "evidence": data.get("evidence", []),
        "review_decisions": data.get("review_decisions", []),
        "repair_history": report.get(
            "repair_history",
            data.get("repair_history", []),
        ),
        "claim_extraction_errors": data.get("claim_extraction_errors", []),
    }


async def run_baseline_impl(
    task: str,
    model: Optional[str] = None,
    baseline_runner: Optional[SingleAgentBaseline] = None,
) -> Dict[str, Any]:
    """Run the single-agent baseline."""
    if not task or not task.strip():
        return {"error": "Task description cannot be empty."}

    runner = baseline_runner or SingleAgentBaseline(model=model)
    result = await runner.run(task)

    return result.model_dump()
