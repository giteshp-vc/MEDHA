import json
import re
from unittest.mock import AsyncMock, MagicMock
import pytest

from core.llm_client import GeminiClient
from core.orchestrator import MedhaOrchestrator, MedhaResult
from models.schemas import ClaimKind
from models.agent_models import AgentResponse
from models.schemas import AgentStatus, ClaimStatus, TaskStatus, VerificationDecision
from models.task_models import SubTask, TaskPlan, TaskResult
from models.verification_models import Claim, VerificationReport
from utils.retry import LLMQuotaExhaustedError
from verification.evidence_ledger import EvidenceLedger


def _mock_llm_response(text: str):
    resp = MagicMock()
    resp.text = text
    resp.parsed = None
    usage = MagicMock()
    usage.prompt_token_count = 10
    usage.candidates_token_count = 20
    usage.total_token_count = 30
    usage.cached_content_token_count = 0
    resp.usage_metadata = usage
    return resp


@pytest.mark.asyncio
async def test_orchestrator_end_to_end_flow(tmp_path, monkeypatch):
    """Verify MedhaOrchestrator coordinates the complete lifecycle with SUCCESS status."""
    mock_client = MagicMock()

    plan_json = """{
        "task": "Design e-commerce system",
        "subtasks": [
            {
                "id": "T1",
                "role": "Database Architect",
                "objective": "Design relational and document stores",
                "instructions": "Evaluate read/write splits",
                "dependencies": [],
                "expected_output": "Database schema"
            },
            {
                "id": "T2",
                "role": "Security Specialist",
                "objective": "Audit authentication and payment flows",
                "instructions": "Ensure PCI-DSS compliance",
                "dependencies": [],
                "expected_output": "Security posture"
            }
        ],
        "rationale": "High-throughput database and strict security required",
        "completion_criteria": ["Data integrity guaranteed", "Zero unencrypted cardholder data"]
    }"""

    specialist_output_json = """{
        "deliverable": "Use partitioned PostgreSQL with Redis caching for sessions.",
        "findings": ["PostgreSQL supports JSONB for catalog and ACID for orders."],
        "claims": ["Read replicas scale read operations up to 40,000 QPS."],
        "evidence_references": ["PostgreSQL 16 replication benchmarks."],
        "assumptions": ["Write load does not exceed 5,000 TPS."],
        "unresolved_questions": ["Cross-region replication latency under severe partition."]
    }"""

    synthesis_text = (
        "# Architectural Synthesis for E-Commerce\n\n"
        "### Executive Summary\n"
        "A hybrid relational-cache architecture has been designed.\n\n"
        "### Verified Recommendations\n"
        "PostgreSQL read replicas scale reads up to 40k QPS."
    )

    # Route mock calls based on prompt content
    async def mock_generate_content(model, contents, config=None):
        prompt_str = str(contents)
        if "Produce a structured JSON plan" in prompt_str or "TaskPlan" in prompt_str:
            return _mock_llm_response(plan_json)
        elif "AUDIT" in prompt_str:
            m = re.search(r"\[ID:\s*([^\]]+)\]", prompt_str) or re.search(r"Claim ID:\s*([^\n]+)", prompt_str)
            cid = m.group(1).strip() if m else "c1"
            return _mock_llm_response(
                json.dumps({
                    "reviews": [
                        {
                            "claim_id": cid,
                            "decision": "ACCEPT",
                            "reasoning": "Postgres read replicas reliably scale read queries within this range.",
                            "critique": "",
                            "suggested_fix": None,
                            "is_supported_by_evidence": True,
                            "is_evidence_relevant": True,
                            "has_contradictions": False,
                            "assumptions_presented_as_facts": False,
                            "logically_valid": True,
                        }
                    ]
                })
            )
        elif "Please synthesize the final" in prompt_str or "Lead Synthesizer" in str(getattr(config, "system_instruction", "")):
            return _mock_llm_response(synthesis_text)
        else:
            return _mock_llm_response(specialist_output_json)

    mock_client.aio.models.generate_content = AsyncMock(side_effect=mock_generate_content)
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)

    orchestrator = MedhaOrchestrator(
        llm_client=gemini_client,
        max_concurrency=2,
        max_repair_iterations=1,
    )

    result = await orchestrator.run_task("Design a scalable e-commerce backend")

    assert isinstance(result, MedhaResult)
    assert result.status == "SUCCESS"
    assert result.execution_id.startswith("exec_")
    assert "Architectural Synthesis" in result.final_answer
    assert len(result.generated_agents) >= 2
    assert len(result.task_results) == 2
    assert len(result.claims) >= 1
    assert result.verification_report["total_claims"] >= 1
    assert result.duration_seconds >= 0.0


@pytest.mark.asyncio
async def test_orchestrator_failure_handling():
    """Verify orchestrator catches unhandled exceptions gracefully and returns FAILED result."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("Catastrophic API failure"))
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)

    orchestrator = MedhaOrchestrator(llm_client=gemini_client)
    result = await orchestrator.run_task("Test task failure")

    assert isinstance(result, MedhaResult)
    # When all specialist tasks fail, truthful status MUST be FAILED
    assert result.status == "FAILED"
    assert result.execution_id is not None


@pytest.mark.asyncio
async def test_orchestrator_all_specialists_fail_reports_failed():
    """Verify that when all specialist agents fail, MEDHA never reports SUCCESS and explicitly discloses 0 claims."""
    mock_client = MagicMock()

    plan_json = """{
        "task": "Build critical system",
        "subtasks": [
            {
                "id": "T1",
                "role": "Faulty Specialist 1",
                "objective": "Objective 1",
                "instructions": "Instructions 1",
                "dependencies": [],
                "expected_output": "Output 1"
            },
            {
                "id": "T2",
                "role": "Faulty Specialist 2",
                "objective": "Objective 2",
                "instructions": "Instructions 2",
                "dependencies": [],
                "expected_output": "Output 2"
            }
        ],
        "rationale": "Testing failure state",
        "completion_criteria": []
    }"""

    async def mock_generate_content(model, contents, config=None):
        prompt_str = str(contents)
        if "Produce a structured JSON plan" in prompt_str or "TaskPlan" in prompt_str:
            return _mock_llm_response(plan_json)
        else:
            raise RuntimeError("API timeout executing specialist")

    mock_client.aio.models.generate_content = AsyncMock(side_effect=mock_generate_content)
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)

    orchestrator = MedhaOrchestrator(llm_client=gemini_client)
    result = await orchestrator.run_task("Build fault system")

    assert isinstance(result, MedhaResult)
    assert result.status == "FAILED"
    assert "Execution Failure Report" in result.final_answer
    assert "### Status: FAILED" in result.final_answer
    assert result.verification_report.get("total_claims", 0) == 0


@pytest.mark.asyncio
async def test_orchestrator_partial_failure_reports_partial_failure():
    """Verify that when 1 specialist succeeds and 1 fails, status is PARTIAL_FAILURE."""
    mock_client = MagicMock()

    plan_json = """{
        "task": "Multi-agent task",
        "subtasks": [
            {
                "id": "T1",
                "role": "Success Specialist",
                "objective": "Complete objective",
                "instructions": "Do work",
                "dependencies": [],
                "expected_output": "Output"
            },
            {
                "id": "T2",
                "role": "Failing Specialist",
                "objective": "Fail objective",
                "instructions": "Fail",
                "dependencies": [],
                "expected_output": "Fail"
            }
        ],
        "rationale": "Testing partial failure",
        "completion_criteria": []
    }"""

    specialist_1_json = """{
        "deliverable": "Valid deliverable from T1.",
        "findings": ["Finding 1"],
        "claims": ["Claim 1 is verified."],
        "evidence_references": [],
        "assumptions": [],
        "unresolved_questions": []
    }"""

    synthesis_json = "# Partial Synthesis Result\n\nT1 succeeded while T2 experienced issues."

    call_count = {"specialist": 0}

    async def mock_generate_content(model, contents, config=None):
        prompt_str = str(contents)
        if "Produce a structured JSON plan" in prompt_str or "TaskPlan" in prompt_str:
            return _mock_llm_response(plan_json)
        elif "AUDIT" in prompt_str:
            m = re.search(r"\[ID:\s*([^\]]+)\]", prompt_str) or re.search(r"Claim ID:\s*([^\n]+)", prompt_str)
            cid = m.group(1).strip() if m else "c1"
            return _mock_llm_response(
                json.dumps({
                    "reviews": [
                        {
                            "claim_id": cid,
                            "decision": "ACCEPT",
                            "reasoning": "Valid",
                            "critique": "",
                            "suggested_fix": None,
                            "is_supported_by_evidence": True,
                            "is_evidence_relevant": True,
                            "has_contradictions": False,
                            "assumptions_presented_as_facts": False,
                            "logically_valid": True,
                        }
                    ]
                })
            )
        elif "Please synthesize the final" in prompt_str or "Lead Synthesizer" in str(getattr(config, "system_instruction", "")):
            return _mock_llm_response(synthesis_json)
        else:
            call_count["specialist"] += 1
            if call_count["specialist"] == 1:
                return _mock_llm_response(specialist_1_json)
            else:
                raise RuntimeError("Specialist T2 failed unexpectedly")

    mock_client.aio.models.generate_content = AsyncMock(side_effect=mock_generate_content)
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)

    orchestrator = MedhaOrchestrator(llm_client=gemini_client)
    result = await orchestrator.run_task("Multi-agent mixed outcome")

    assert isinstance(result, MedhaResult)
    assert result.status == "PARTIAL_FAILURE"
    failed_result = next(item for item in result.task_results if item["subtask_id"] == "T2")
    assert failed_result["status"] == TaskStatus.FAILED
    assert "Provider/API failure [subtask T2]" in failed_result["error"]
    assert "RuntimeError" in failed_result["error"]


@pytest.mark.asyncio
async def test_orchestrator_quota_exhaustion_during_verification_reports_partial_failure():
    """CRITICAL TEST: Daily quota exhaustion during verification must stop subsequent LLM calls,

    preserve completed specialist deliverables and claims, and report PARTIAL_FAILURE without crashing.
    """
    mock_client = MagicMock()

    plan_json = """{
        "task": "High-volume data processing system",
        "subtasks": [
            {
                "id": "T1",
                "role": "Pipeline Specialist",
                "objective": "Design Kafka streaming topology",
                "instructions": "State throughput assertions",
                "dependencies": [],
                "expected_output": "Pipeline design"
            }
        ],
        "rationale": "High throughput",
        "completion_criteria": []
    }"""

    specialist_json = """{
        "deliverable": "Kafka cluster with 12 partitions.",
        "findings": ["Kafka handles 100,000 msgs/sec."],
        "claims": ["Kafka cluster achieves 100,000 msgs/sec."],
        "evidence_references": [],
        "assumptions": ["Network bandwidth is 10Gbps."],
        "unresolved_questions": []
    }"""

    async def mock_generate_content(model, contents, config=None):
        prompt_str = str(contents)
        if "Produce a structured JSON plan" in prompt_str or "TaskPlan" in prompt_str:
            return _mock_llm_response(plan_json)
        elif "Extract discrete" in prompt_str:
            return _mock_llm_response('{"claims": ["Kafka cluster achieves 100,000 msgs/sec."], "assumptions": []}')
        elif "AUDIT" in prompt_str:
            # Simulate Groq daily token quota exhaustion on verification!
            raise LLMQuotaExhaustedError("Daily token quota limit exhausted on tokens per day (TPD): Limit 500,000 429")
        else:
            return _mock_llm_response(specialist_json)

    mock_client.aio.models.generate_content = AsyncMock(side_effect=mock_generate_content)
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)

    orchestrator = MedhaOrchestrator(llm_client=gemini_client)
    result = await orchestrator.run_task("Kafka pipeline task")

    assert isinstance(result, MedhaResult)
    # Specialist completed, so status must be PARTIAL_FAILURE (not FAILED or crash!)
    assert result.status == "PARTIAL_FAILURE"
    # Specialist deliverable must be preserved
    assert len(result.task_results) == 1
    assert "Kafka cluster with 12 partitions" in result.task_results[0]["output"]
    # Claims must be preserved
    assert len(result.claims) >= 1
    # Unfinished claim must be explicitly marked UNREVIEWED (audit interrupted), NOT falsely accepted or counted as reviewed
    for c in result.claims:
        assert c["status"] == "UNREVIEWED"
        assert "quota" in (c.get("reviewer_feedback") or "").lower()
    assert len(result.verification_report.get("unreviewed_claims", [])) >= 1
    # Final answer must be produced, disclosing quota interruption
    assert result.final_answer is not None
    assert len(result.final_answer) > 50


@pytest.mark.asyncio
async def test_orchestrator_registers_all_structured_claims_after_extraction_quota(
    monkeypatch,
):
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=LLMQuotaExhaustedError("daily provider quota exhausted during extraction")
    )
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)
    plan = TaskPlan(
        task="Attendance system analysis",
        subtasks=[
            SubTask(id=f"T{index}", role=role, objective=f"Review {role}")
            for index, role in enumerate(
                ["AI Systems Architect", "Privacy Specialist", "Cybersecurity Engineer", "Reliability Specialist"],
                1,
            )
        ],
    )
    specialist_results = [
        TaskResult(
            subtask_id="T1",
            agent_id="agent_arch",
            role="AI Systems Architect",
            status=TaskStatus.COMPLETED,
            output="Architect report",
            claims=[f"Architecture claim {index} is distinct." for index in range(10)],
        ),
        TaskResult(
            subtask_id="T2",
            agent_id="agent_privacy",
            role="Privacy Specialist",
            status=TaskStatus.COMPLETED,
            output="Privacy report without pre-identified claims",
            claims=[],
        ),
        TaskResult(
            subtask_id="T3",
            agent_id="agent_cyber",
            role="Cybersecurity Engineer",
            status=TaskStatus.COMPLETED,
            output="Cybersecurity report",
            claims=[f"Cybersecurity claim {index} is distinct." for index in range(5)],
        ),
        TaskResult(
            subtask_id="T4",
            agent_id="agent_reliability",
            role="Reliability Specialist",
            status=TaskStatus.COMPLETED,
            output="Reliability report",
            claims=[f"Reliability claim {index} is distinct." for index in range(10)],
        ),
    ]

    monkeypatch.setattr(
        "core.orchestrator.TaskPlanner.create_plan",
        AsyncMock(return_value=plan),
    )
    monkeypatch.setattr(
        "core.orchestrator.ExecutionEngine.execute_plan",
        AsyncMock(return_value=specialist_results),
    )
    monkeypatch.setattr(
        "core.orchestrator.AdversarialReviewer.review_claims",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        "core.orchestrator.RepairEngine.execute_repair_loop",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        "agents.synthesizer_agent.SynthesizerAgent.synthesize",
        AsyncMock(return_value="Mocked final report"),
    )

    result = await MedhaOrchestrator(llm_client=gemini_client).run_task("Attendance system")

    assert len(result.claims) == 25
    assert {claim["originating_role"] for claim in result.claims} == {
        "AI Systems Architect",
        "Cybersecurity Engineer",
        "Reliability Specialist",
    }
    assert {claim["subtask_id"] for claim in result.claims} == {"T1", "T3", "T4"}
    assert all(claim["status"] == ClaimStatus.UNREVIEWED for claim in result.claims)
    assert result.status == "PARTIAL_FAILURE"
    assert any("T2" in error and "quota" in error.lower() for error in result.claim_extraction_errors)
    assert mock_client.aio.models.generate_content.await_count == 1


@pytest.mark.asyncio
async def test_deterministic_synthesis_labels_and_separates_unverified_content():
    mock_client = MagicMock()
    mock_client.generate_text = AsyncMock(
        side_effect=LLMQuotaExhaustedError("synthesis provider quota exhausted")
    )
    report = VerificationReport(
        total_claims=4,
        accepted_claims=[
            Claim(
                claim_id="accepted",
                claim_text="Architecture uses a central service.",
                originating_agent="a1",
                originating_role="Architect",
                subtask_id="T1",
                source_provenance=[
                    {"agent_id": "a1", "role": "Architect", "subtask_id": "T1"},
                    {"agent_id": "a4", "role": "Reliability Specialist", "subtask_id": "T4"},
                ],
                evidence_references=["evidence_1"],
                status=ClaimStatus.SUPPORTED,
            )
        ],
        rejected_claims=[
            Claim(claim_id="rejected", claim_text="The system is flawless.", originating_agent="a2")
        ],
        unverifiable_claims=[
            Claim(
                claim_id="measurement",
                claim_text="We measured 12 ms p95 latency in a pilot.",
                originating_agent="a3",
                claim_kind=ClaimKind.EMPIRICAL_RESULT,
                status=ClaimStatus.UNVERIFIABLE,
            )
        ],
        unreviewed_claims=[
            Claim(
                claim_id="target",
                claim_text="Target uptime should be 99.9%.",
                originating_agent="a4",
                claim_kind=ClaimKind.PROPOSED_TARGET,
                status=ClaimStatus.UNREVIEWED,
            )
        ],
        summary_message="Verification interrupted due to provider quota exhaustion.",
    )
    from agents.synthesizer_agent import SynthesizerAgent

    synthesizer = SynthesizerAgent(llm_client=mock_client)
    answer = await synthesizer.synthesize(
        task_description="Design attendance system",
        specialist_results=[
            {
                "role": "Reliability Specialist",
                "subtask_id": "T4",
                "status": "COMPLETED",
                "findings": ["We recommend redundant service instances."],
                "claims": ["Target uptime should be 99.9%.", "We measured 12 ms p95 latency in a pilot."],
                "assumptions": ["Assume campus Wi-Fi remains available."],
                "output": "Partial specialist deliverable retained.",
            }
        ],
        verification_report=report,
        unresolved_questions=["Has the target been tested?"],
        partial_execution_notice="Verification interrupted; synthesis quota exhausted.",
    )

    assert "UNVERIFIED COMPILATION" in answer
    assert "Partial specialist deliverable retained." in answer
    assert "Assumptions (Unverified)" in answer
    assert "Recommendations (Unverified)" in answer
    assert "Proposed Performance Targets (Not Measurements)" in answer
    assert "Specialist-Reported Empirical Results (Unverified)" in answer
    targets_section = answer.split("### Proposed Performance Targets (Not Measurements)\n", 1)[1].split(
        "### Specialist-Reported Empirical Results (Unverified)", 1
    )[0]
    measurements_section = answer.split("### Specialist-Reported Empirical Results (Unverified)\n", 1)[1].split(
        "### Verification Audit", 1
    )[0]
    assert "Target uptime should be 99.9%" in targets_section
    assert "Target uptime should be 99.9%" not in measurements_section
    assert "We measured 12 ms p95 latency in a pilot." in measurements_section
    assert "Reviewed / supported" in answer
    assert "[accepted] status=SUPPORTED" in answer
    assert "Reliability Specialist (a4, T4)" in answer
    assert "evidence_references=[evidence_1]" in answer
    assert "Rejected" in answer
    assert "Unverifiable" in answer
    assert "Unreviewed" in answer
    assert "synthesis provider quota exhausted" in answer
    assert "Verification interrupted; synthesis quota exhausted." in answer


@pytest.mark.asyncio
async def test_claim_capacity_overflow_reports_discovered_registered_and_unregistered(monkeypatch):
    class LimitedEvidenceLedger(EvidenceLedger):
        def __init__(self, task_id=""):
            super().__init__(task_id=task_id, max_claims=2)

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock()
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)
    plan = TaskPlan(
        task="Small ledger task",
        subtasks=[SubTask(id="T1", role="Architect", objective="Provide claims")],
    )
    specialist_result = TaskResult(
        subtask_id="T1",
        agent_id="agent_arch",
        role="Architect",
        status=TaskStatus.COMPLETED,
        output="Architecture report",
        claims=[f"Distinct architecture assertion number {index}." for index in range(3)],
    )

    monkeypatch.setattr("core.orchestrator.EvidenceLedger", LimitedEvidenceLedger)
    monkeypatch.setattr("core.orchestrator.TaskPlanner.create_plan", AsyncMock(return_value=plan))
    monkeypatch.setattr(
        "core.orchestrator.ExecutionEngine.execute_plan",
        AsyncMock(return_value=[specialist_result]),
    )
    monkeypatch.setattr("core.orchestrator.AdversarialReviewer.review_claims", AsyncMock(return_value=[]))
    monkeypatch.setattr("core.orchestrator.RepairEngine.execute_repair_loop", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        "agents.synthesizer_agent.SynthesizerAgent.synthesize",
        AsyncMock(return_value="Mocked final report"),
    )

    result = await MedhaOrchestrator(llm_client=gemini_client).run_task("Small ledger task")

    assert len(result.claims) == 2
    assert result.status == "PARTIAL_FAILURE"
    diagnostic = " ".join(result.claim_extraction_errors)
    assert "discovered=3" in diagnostic
    assert "registered=2" in diagnostic
    assert "unregistered=1" in diagnostic
    assert "capacity=2" in diagnostic
