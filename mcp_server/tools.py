import json
from pathlib import Path
from typing import Any, Dict, Optional

from benchmark.baseline import SingleAgentBaseline
from config.settings import LOGS_DIR
from core.orchestrator import MedhaOrchestrator, MedhaResult
from utils.logger import get_logger

logger = get_logger("mcp_tools")

# Cache of completed in-memory executions
_execution_cache: Dict[str, MedhaResult] = {}


def cache_execution_result(result: MedhaResult) -> None:
    """Store execution result in memory for MCP query tools."""
    _execution_cache[result.execution_id] = result


def get_cached_or_logged_execution(execution_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve an execution record from in-memory cache or saved logs."""
    if execution_id in _execution_cache:
        return _execution_cache[execution_id].model_dump()

    # Try loading from logs/execution_{execution_id}.json
    log_path = LOGS_DIR / f"execution_{execution_id}.json"
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read execution log {log_path}: {e}")

    return None


async def run_medha_task_impl(
    task: str,
    max_concurrency: int = 4,
    max_repair_iterations: int = 2,
    orchestrator: Optional[MedhaOrchestrator] = None,
) -> Dict[str, Any]:
    """Execute a task through MEDHA and return a summary."""
    if not task or not task.strip():
        return {"error": "Task description cannot be empty.", "status": "FAILED"}

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


async def get_execution_status_impl(execution_id: str) -> Dict[str, Any]:
    """Query current status, agents, and workflow progression for an execution ID."""
    if not execution_id:
        return {"error": "Execution ID must be provided."}

    data = get_cached_or_logged_execution(execution_id)
    if not data:
        return {"error": f"Execution ID '{execution_id}' not found."}

    # Format status response
    return {
        "execution_id": execution_id,
        "status": data.get("status", "UNKNOWN"),
        "duration_seconds": data.get("duration_seconds"),
        "agent_statuses": [
            {
                "agent_id": a.get("agent_id"),
                "role": a.get("role"),
                "status": a.get("status"),
            }
            for a in data.get("generated_agents", data.get("agent_records", []))
        ],
        "workflow_stages": data.get("workflow_stages", []),
        "error": data.get("error"),
    }


async def get_verification_report_impl(execution_id: str) -> Dict[str, Any]:
    """Query detailed claim verification and repair report for an execution ID."""
    if not execution_id:
        return {"error": "Execution ID must be provided."}

    data = get_cached_or_logged_execution(execution_id)
    if not data:
        return {"error": f"Execution ID '{execution_id}' not found."}

    rep = data.get("verification_report", {})
    return {
        "execution_id": execution_id,
        "total_claims": rep.get("total_claims", len(data.get("claims", []))),
        "support_rate": rep.get("support_rate"),
        "accepted_claims": rep.get("accepted_claims", []),
        "rejected_claims": rep.get("rejected_claims", []),
        "unverifiable_claims": rep.get("unverifiable_claims", []),
        "repair_history": rep.get("repair_history", data.get("repair_history", [])),
        "evidence_references": data.get("evidence", []),
    }


async def run_baseline_impl(
    task: str,
    model: Optional[str] = None,
    baseline_runner: Optional[SingleAgentBaseline] = None,
) -> Dict[str, Any]:
    """Run single-agent baseline comparison."""
    if not task or not task.strip():
        return {"error": "Task description cannot be empty."}

    runner = baseline_runner or SingleAgentBaseline(model=model)
    result = await runner.run(task)
    return result.model_dump()

