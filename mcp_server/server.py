
import json

from mcp.server.mcpserver import MCPServer

from mcp_server.tools import (
    get_execution_result_impl,
    get_execution_status_impl,
    get_verification_report_impl,
    run_baseline_impl,
    run_medha_task_impl,
    start_medha_task_impl,
)
from mcp_server.registry import execution_registry
from config.settings import MEDHA_STORAGE_MODE
from storage.workspace import cleanup_stale_workspaces

from utils.logger import get_logger

logger = get_logger("mcp_server")

mcp = MCPServer(
    name="MEDHA-MultiAgent-Server",
    version="1.0.0",
    instructions=(
        "MEDHA is an evidence-driven multi-agent intelligence system. "
        "Use start_medha_task for background execution, "
        "get_execution_status to monitor jobs, "
        "get_execution_result to retrieve results, and "
        "get_verification_report to inspect evidence and claims."
    ),
)


@mcp.tool()
async def start_medha_task(
    task: str,
    max_concurrency: int = 4,
    max_repair_iterations: int = 2,
    documents: list[str] | None = None,
    source_urls: list[str] | None = None,
) -> str:
    """Submit a MEDHA task for background execution and return its job ID."""
    logger.info("MCP background task submitted.")
    result = await start_medha_task_impl(
        task=task,
        max_concurrency=max_concurrency,
        max_repair_iterations=max_repair_iterations,
        documents=documents,
        source_urls=source_urls,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_execution_status(execution_id: str) -> str:
    """Get the current status of a MEDHA job or legacy execution ID."""
    result = await get_execution_status_impl(execution_id=execution_id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_execution_result(job_id: str) -> str:
    """Retrieve the final result of a background MEDHA job."""
    result = await get_execution_result_impl(job_id=job_id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_verification_report(execution_id: str) -> str:
    """Retrieve claims, evidence, review decisions, and repair history."""
    result = await get_verification_report_impl(execution_id=execution_id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def run_medha_task(
    task: str,
    max_concurrency: int = 4,
    max_repair_iterations: int = 2,
    documents: list[str] | None = None,
    source_urls: list[str] | None = None,
) -> str:
    """Run a MEDHA task synchronously and return its final summary."""
    result = await run_medha_task_impl(
        task=task,
        max_concurrency=max_concurrency,
        max_repair_iterations=max_repair_iterations,
        documents=documents,
        source_urls=source_urls,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def run_baseline(task: str, model: str = "") -> str:
    """Run a single-agent baseline without multi-agent orchestration."""
    result = await run_baseline_impl(
        task=task,
        model=model or None,
    )
    return json.dumps(result, indent=2)


def main():
    """Start the MEDHA MCP server over stdio."""
    cleanup_report = cleanup_stale_workspaces(MEDHA_STORAGE_MODE)
    if cleanup_report["failed"]:
        logger.warning("MCP startup cleanup could not remove %d expired execution workspace(s).", cleanup_report["failed"])
    logger.info("Starting MEDHA MCP Server over stdio...")
    try:
        mcp.run(transport="stdio")
    finally:
        execution_registry.clear_ephemeral_state()


if __name__ == "__main__":
    main()
