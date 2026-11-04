import asyncio
import json
import sys
from typing import Optional

from mcp.server.mcpserver import MCPServer
from mcp_server.tools import (
    get_execution_status_impl,
    get_verification_report_impl,
    run_baseline_impl,
    run_medha_task_impl,
)
from utils.logger import get_logger

logger = get_logger("mcp_server")

# Initialize MCP Server for MEDHA
mcp = MCPServer("MEDHA-MultiAgent-Server")


@mcp.tool()
async def run_medha_task(
    task: str,
    max_concurrency: int = 4,
    max_repair_iterations: int = 2,
) -> str:
    """Run an end-to-end task through the MEDHA multi-agent intelligence system.

    Decomposes the task dynamically, executes specialist agents in parallel,
    audits claims adversarially, repairs rejected claims, and synthesizes the final answer.
    """
    logger.info(f"MCP tool 'run_medha_task' called for: '{task[:60]}...'")
    result = await run_medha_task_impl(
        task=task,
        max_concurrency=max_concurrency,
        max_repair_iterations=max_repair_iterations,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_execution_status(execution_id: str) -> str:
    """Query current execution status, agent activity, and workflow progression for an execution ID."""
    logger.info(f"MCP tool 'get_execution_status' called for ID: {execution_id}")
    result = await get_execution_status_impl(execution_id=execution_id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_verification_report(execution_id: str) -> str:
    """Retrieve the full evidence audit report, accepted/rejected claims, and repair history for an execution ID."""
    logger.info(f"MCP tool 'get_verification_report' called for ID: {execution_id}")
    result = await get_verification_report_impl(execution_id=execution_id)
    return json.dumps(result, indent=2)


@mcp.tool()
async def run_baseline(task: str, model: str = "") -> str:
    """Run a single-agent baseline answer using direct LLM generation with no multi-agent orchestration."""
    logger.info(f"MCP tool 'run_baseline' called for: '{task[:60]}...'")
    result = await run_baseline_impl(task=task, model=model if model else None)
    return json.dumps(result, indent=2)


def main():
    """Main entrypoint for running the MEDHA MCP server over stdio."""
    logger.info("Starting MEDHA MCP Server over stdio...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

