import asyncio
import json
import os
import runpy
import sys
import textwrap
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


WORKSPACE = Path(__file__).resolve().parents[1]


MOCK_SERVER_SCRIPT = textwrap.dedent(
    """
    import runpy
    from core.orchestrator import MedhaResult
    import mcp_server.tools as tools

    class ProtocolTestOrchestrator:
        def __init__(self, *args, **kwargs):
            pass

        async def run_task(self, task):
            if task == "raise from mocked orchestrator":
                raise RuntimeError("mocked orchestrator failure")
            return MedhaResult(
                execution_id="exec_protocol_test",
                task=task,
                final_answer="Protocol test result",
                status="SUCCESS",
                claims=[
                    {
                        "claim_id": "claim_protocol_1",
                        "claim_text": "The protocol test claim is unreviewed.",
                        "status": "UNREVIEWED",
                    }
                ],
                evidence=[
                    {
                        "evidence_id": "evidence_protocol_1",
                        "supporting_content": "Protocol test evidence",
                    }
                ],
                review_decisions=[],
                repair_history=[],
                verification_report={
                    "total_claims": 1,
                    "accepted_claims": [],
                    "rejected_claims": [],
                    "unverifiable_claims": [],
                    "unreviewed_claims": [
                        {"claim_id": "claim_protocol_1"}
                    ],
                    "needs_revision_claims": [],
                    "support_rate": None,
                    "repair_success_rate": 0.0,
                },
            )

    tools.MedhaOrchestrator = ProtocolTestOrchestrator
    runpy.run_module("mcp_server.server", run_name="__main__")
    """
)


def _server_parameters() -> StdioServerParameters:
    environment = os.environ.copy()
    environment.pop("GROQ_API_KEY", None)
    environment.pop("GEMINI_API_KEY", None)
    return StdioServerParameters(
        command=sys.executable,
        args=["-c", MOCK_SERVER_SCRIPT],
        cwd=WORKSPACE,
        env=environment,
    )


async def _call_json(session: ClientSession, name: str, arguments: dict) -> dict:
    response = await session.call_tool(name, arguments)
    assert response.content, f"MCP tool {name} returned no content"
    payload = response.content[0].text
    return json.loads(payload)


@pytest.mark.asyncio
async def test_real_mcp_stdio_protocol_end_to_end():
    async with stdio_client(_server_parameters()) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            initialized = await session.initialize()
            assert initialized.server_info.name == "MEDHA-MultiAgent-Server"

            discovered = await session.list_tools()
            tools = {tool.name: tool for tool in discovered.tools}
            expected = {
                "start_medha_task",
                "get_execution_status",
                "get_execution_result",
                "get_verification_report",
                "run_medha_task",
                "run_baseline",
            }
            assert set(tools) == expected
            for tool in tools.values():
                assert tool.description
                assert tool.input_schema["type"] == "object"

            invalid_status = await _call_json(
                session,
                "get_execution_status",
                {"execution_id": "mcp_missing_protocol_id"},
            )
            assert "error" in invalid_status

            invalid_start = await _call_json(
                session,
                "start_medha_task",
                {"task": "", "max_concurrency": 1, "max_repair_iterations": 0},
            )
            assert invalid_start["status"] == "REJECTED"
            assert "empty" in invalid_start["error"].lower()

            malformed_start = await _call_json(
                session,
                "start_medha_task",
                {"task": "valid", "max_concurrency": 0, "max_repair_iterations": 0},
            )
            assert malformed_start["status"] == "REJECTED"
            assert "max_concurrency" in malformed_start["error"]

            malformed_schema = await session.call_tool(
                "start_medha_task",
                {"task": "valid", "max_concurrency": "not-an-int", "max_repair_iterations": 0},
            )
            assert malformed_schema.is_error or malformed_schema.content

            failed_submission = await _call_json(
                session,
                "start_medha_task",
                {"task": "raise from mocked orchestrator", "max_concurrency": 1},
            )
            failed_job_id = failed_submission["job_id"]
            failed_status = None
            for _ in range(100):
                failed_status = await _call_json(
                    session,
                    "get_execution_status",
                    {"execution_id": failed_job_id},
                )
                if failed_status["status"] == "FAILED":
                    break
                await asyncio.sleep(0.01)
            assert failed_status["status"] == "FAILED"
            assert "mocked orchestrator failure" in failed_status["error"]

            accepted = await _call_json(
                session,
                "start_medha_task",
                {
                    "task": "Run the mocked protocol task",
                    "max_concurrency": 1,
                    "max_repair_iterations": 0,
                },
            )
            assert accepted["job_id"].startswith("mcp_")
            job_id = accepted["job_id"]

            status = None
            result = None
            for _ in range(100):
                status = await _call_json(
                    session,
                    "get_execution_status",
                    {"execution_id": job_id},
                )
                if status["status"] == "COMPLETED":
                    result = await _call_json(
                        session,
                        "get_execution_result",
                        {"job_id": job_id},
                    )
                    if result.get("result"):
                        break
                await asyncio.sleep(0.01)

            assert status["status"] == "COMPLETED"
            assert result["result"]["final_answer"] == "Protocol test result"
            assert result["result"]["execution_id"] == "exec_protocol_test"

            report = await _call_json(
                session,
                "get_verification_report",
                {"execution_id": job_id},
            )
            assert report["job_id"] == job_id
            assert report["total_claims"] == 1
            assert report["unreviewed_claims"]
            assert report["all_claims"]
            assert report["evidence"]

            malformed_status = await _call_json(
                session,
                "get_execution_result",
                {"job_id": "mcp_not_found"},
            )
            assert "error" in malformed_status

            # Closing the session/stdio context is the real shutdown path; the
            # bounded context exit must not leave the server protocol hanging.
