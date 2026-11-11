import asyncio
import json
import re
from unittest.mock import AsyncMock, MagicMock
import pytest

from core.llm_client import GeminiClient
from core.orchestrator import MedhaOrchestrator, MedhaResult
from benchmark.treatments import get_treatment_preset
from models.deliberation_models import (
    DeliberationExchange,
    DeliberationReport,
    DeliberationRound,
    DeliberationTerminationReason,
    DeliberationTrigger,
    DeliberationTriggerType,
)
from models.schemas import ClaimKind
from models.agent_models import AgentResponse
from models.schemas import AgentStatus, ClaimStatus, TaskStatus, VerificationDecision
from models.task_models import SubTask, TaskPlan, TaskResult
from models.frozen_plan import FrozenTaskPlan, validate_task_plan
from models.verification_models import Claim, VerificationReport
from utils.token_tracker import TokenTracker
from utils.retry import LLMQuotaExhaustedError
from verification.evidence_ledger import EvidenceLedger
from verification.claim_extractor import ClaimExtractor


def test_frozen_plan_serialization_hash_and_loaded_copies_are_stable():
    task = "Shared plan task"
    plan = TaskPlan(
        task="Synthesized plan identity",
        subtasks=[
            SubTask(id="T1", role="Source", objective="Analyze source"),
            SubTask(id="T2", role="Target", objective="Use source", dependencies=["T1"]),
        ],
    )

    frozen = FrozenTaskPlan.from_plan(task, plan)
    first_copy = frozen.load_plan(task)
    second_copy = frozen.load_plan(task)

    assert frozen.serialized_plan == FrozenTaskPlan.from_plan(task, plan).serialized_plan
    assert frozen.plan_hash == FrozenTaskPlan.from_plan(task, plan).plan_hash
    assert first_copy is not second_copy
    assert first_copy.subtasks[0] is not second_copy.subtasks[0]
    first_copy.subtasks[0].role = "Mutated treatment copy"
    assert second_copy.subtasks[0].role == "Source"
    assert frozen.load_plan(task).subtasks[0].role == "Source"


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (
            TaskPlan(
                task="Plan",
                subtasks=[
                    SubTask(id="T1", role="One", objective="First"),
                    SubTask(id="T1", role="Two", objective="Duplicate"),
                ],
            ),
            "duplicate subtask IDs",
        ),
        (
            TaskPlan(
                task="Plan",
                subtasks=[SubTask(id="T1", role="One", objective="First", dependencies=["T9"])],
            ),
            "unknown dependency IDs",
        ),
        (
            TaskPlan(
                task="Plan",
                subtasks=[
                    SubTask(id="T1", role="One", objective="First", dependencies=["T2"]),
                    SubTask(id="T2", role="Two", objective="Second", dependencies=["T1"]),
                ],
            ),
            "contains a cycle",
        ),
        (
            TaskPlan(task="Plan", subtasks=[SubTask(id="T1", role="  ", objective="First")]),
            "non-empty role",
        ),
    ],
)
def test_frozen_plan_rejects_invalid_plan_graphs(plan, message):
    with pytest.raises(ValueError, match=message):
        FrozenTaskPlan.from_plan("Source task", plan)


def test_frozen_plan_rejects_task_identity_mismatch_and_tampering():
    frozen = FrozenTaskPlan.from_plan(
        "Original source task",
        TaskPlan(
            task="Plan identity",
            subtasks=[SubTask(id="T1", role="Analyst", objective="Analyze")],
        ),
    )

    with pytest.raises(ValueError, match="different source task"):
        frozen.load_plan("Different task")
    with pytest.raises(ValueError, match="hash does not match"):
        FrozenTaskPlan(
            source_task=frozen.source_task,
            serialized_plan=frozen.serialized_plan + " ",
            plan_hash=frozen.plan_hash,
        ).load_plan(frozen.source_task)
    with pytest.raises(ValueError, match="hash does not match"):
        FrozenTaskPlan(
            source_task="Rebound source task",
            serialized_plan=frozen.serialized_plan,
            plan_hash=frozen.plan_hash,
        ).load_plan("Rebound source task")


@pytest.mark.parametrize(
    "status",
    [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED],
)
def test_frozen_plan_rejects_subtasks_with_prior_execution_state(status):
    plan = TaskPlan(
        task="Execution-state validation",
        subtasks=[SubTask(id="T1", role="Analyst", objective="Analyze", status=status)],
    )

    with pytest.raises(ValueError, match="frozen plans require every subtask to be PENDING"):
        FrozenTaskPlan.from_plan("Source task", plan)

    assert plan.subtasks[0].status == status


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


def _make_deliberation_report(execution_id, termination=DeliberationTerminationReason.RESOLVED):
    trigger = DeliberationTrigger(
        execution_id=execution_id,
        trigger_type=DeliberationTriggerType.CONFLICTING_CLAIMS,
        claim_reference="claim:exec:T1:0",
        source_agent_id="agent_T1",
        target_agent_id="agent_T2",
        source_subtask_id="T1",
        target_subtask_id="T2",
        reason="Conflicting specialist assertions.",
        target_selection_rationale="The target is the declared dependency.",
    )
    exchange = DeliberationExchange(
        exchange_id=trigger.exchange_id,
        execution_id=execution_id,
        correlation_id=trigger.correlation_id,
        claim_reference=trigger.claim_reference,
        trigger_type=trigger.trigger_type,
        initiator_agent_id=trigger.source_agent_id,
        target_agent_id=trigger.target_agent_id,
        request_message_id="msg_request",
        response_message_id="msg_response",
        response_type="DISAGREEMENT",
        response_content="The assertion remains unresolved.",
        disagreement=True,
    )
    return DeliberationReport(
        execution_id=execution_id,
        triggered=True,
        triggers=[trigger],
        exchanges=[exchange],
        rounds=[DeliberationRound(
            execution_id=execution_id,
            round_number=1,
            exchange_ids=[exchange.exchange_id],
            message_count=2,
            token_count=12,
        )],
        unresolved_disagreements=["The assertion remains unresolved."],
        diagnostics=["Mocked deliberation diagnostic."],
        termination_reason=termination,
        total_messages=2,
        total_tokens=12,
    )


def _install_orchestrator_deliberation_stubs(monkeypatch, specialist_result, deliberation_outcome, events):
    plan = TaskPlan(
        task="Integration test plan",
        subtasks=[
            SubTask(id="T1", role="Source Specialist", objective="Produce source finding"),
            SubTask(id="T2", role="Target Specialist", objective="Respond to source", dependencies=["T1"]),
        ],
    )
    captured = {}

    async def create_plan(self, task_description, max_subtasks=4):
        events.append("planning")
        return plan

    async def execute_plan(self, **kwargs):
        events.append("specialist_execution")
        captured.update(kwargs)
        captured.setdefault("execution_plan_objects", []).append(kwargs["plan"])
        captured.setdefault("execution_plan_serializations", []).append(
            json.dumps(kwargs["plan"].model_dump(mode="json"), sort_keys=True)
        )
        captured.setdefault("execution_policies", []).append(kwargs["execution_policy"])
        results = specialist_result if isinstance(specialist_result, list) else [specialist_result]
        for result in results:
            agent = MagicMock()
            agent.agent_id = result.agent_id
            agent.role = result.role
            agent.assigned_subtask_id = result.subtask_id
            agent.get_summary.return_value = {
                "agent_id": result.agent_id,
                "role": result.role,
                "subtask_id": result.subtask_id,
            }
            kwargs["agent_factory"]._registry[result.agent_id] = agent
        return results

    class StubCoordinator:
        def __init__(self, **kwargs):
            events.append("coordinator_constructed")
            captured["coordinator_kwargs"] = kwargs

        async def run(self):
            events.append("deliberation")
            if isinstance(deliberation_outcome, Exception):
                raise deliberation_outcome
            if callable(deliberation_outcome):
                return deliberation_outcome(
                    execution_id=captured["coordinator_kwargs"]["execution_id"]
                )
            return deliberation_outcome

    original_extract = ClaimExtractor.extract_claims_with_diagnostics

    async def track_extraction(self, *args, **kwargs):
        events.append("claim_extraction")
        return await original_extract(self, *args, **kwargs)

    async def mock_review(self, *args, **kwargs):
        events.append("review")
        captured["review_kwargs"] = kwargs
        captured["claims_at_review"] = await kwargs["ledger"].get_all_claims()
        captured["evidence_at_review"] = {
            claim.claim_id: await kwargs["ledger"].get_evidence_for_claim(claim.claim_id)
            for claim in captured["claims_at_review"]
        }
        return []

    async def mock_repair(self, *args, **kwargs):
        events.append("repair")
        return []

    async def mock_synthesis(self, *args, **kwargs):
        events.append("synthesis")
        captured["synthesis_kwargs"] = kwargs
        return "Mocked orchestrator integration result"

    monkeypatch.setattr("core.orchestrator.TaskPlanner.create_plan", create_plan)
    monkeypatch.setattr("core.orchestrator.ExecutionEngine.execute_plan", execute_plan)
    monkeypatch.setattr("core.orchestrator.DeliberationCoordinator", StubCoordinator)
    monkeypatch.setattr(ClaimExtractor, "extract_claims_with_diagnostics", track_extraction)
    monkeypatch.setattr("core.orchestrator.AdversarialReviewer.review_claims", mock_review)
    monkeypatch.setattr("core.orchestrator.RepairEngine.execute_repair_loop", mock_repair)
    monkeypatch.setattr("agents.synthesizer_agent.SynthesizerAgent.synthesize", mock_synthesis)
    return plan, captured


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
async def test_planned_runs_reuse_frozen_plan_and_apply_b_c_d_treatments(monkeypatch):
    events = []
    specialist_result = TaskResult(
        subtask_id="T1",
        agent_id="agent_T1",
        role="Source Specialist",
        status=TaskStatus.COMPLETED,
        output="Completed source analysis.",
        claims=["A specialist assertion remains pending review."],
    )
    _, captured = _install_orchestrator_deliberation_stubs(
        monkeypatch,
        specialist_result,
        lambda execution_id: _make_deliberation_report(execution_id),
        events,
    )
    mock_client = MagicMock()
    orchestrator = MedhaOrchestrator(llm_client=GeminiClient(api_key="mock_key", client=mock_client))
    frozen_plan = FrozenTaskPlan.from_plan(
        "Shared frozen-plan treatment test",
        TaskPlan(
            task="A stable synthesized task identity",
            subtasks=[
                SubTask(id="T1", role="Source Specialist", objective="Produce source finding"),
                SubTask(id="T2", role="Target Specialist", objective="Respond to source", dependencies=["T1"]),
            ],
        ),
    )
    arm_policies = [
        get_treatment_preset("INDEPENDENT_PARALLEL").policy,
        get_treatment_preset("DEPENDENCY_AWARE").policy,
        get_treatment_preset("FULL_MEDHA").policy,
    ]

    results = [
        await orchestrator.run_planned_task(
            "Shared frozen-plan treatment test",
            frozen_plan,
            policy,
        )
        for policy in arm_policies
    ]

    assert len(captured["execution_plan_serializations"]) == 3
    assert len(set(captured["execution_plan_serializations"])) == 1
    assert len({id(plan) for plan in captured["execution_plan_objects"]}) == 3
    assert [policy.deliberation_enabled for policy in arm_policies] == [False, False, True]
    assert all(policy.verification_enabled and policy.synthesis_enabled for policy in arm_policies)
    assert [policy.honor_dependencies for policy in captured["execution_policies"]] == [
        False,
        True,
        True,
    ]
    assert [policy.route_messages for policy in captured["execution_policies"]] == [
        False,
        True,
        True,
    ]
    assert events.count("planning") == 0
    assert events.count("deliberation") == 1
    assert events.count("review") == 3
    assert events.count("synthesis") == 3
    assert all(result.plan_artifact.plan_hash == frozen_plan.plan_hash for result in results)

    captured["execution_plan_objects"][0].subtasks[0].role = "Mutated B copy"
    assert captured["execution_plan_objects"][1].subtasks[0].role == "Source Specialist"
    assert captured["execution_plan_objects"][2].subtasks[0].role == "Source Specialist"
    assert frozen_plan.load_plan("Shared frozen-plan treatment test").subtasks[0].role == "Source Specialist"


@pytest.mark.asyncio
async def test_invalid_frozen_plan_is_rejected_without_replanning(monkeypatch):
    planner = AsyncMock()
    monkeypatch.setattr("core.orchestrator.TaskPlanner.create_plan", planner)
    orchestrator = MedhaOrchestrator(llm_client=GeminiClient(api_key="mock_key", client=MagicMock()))
    frozen_plan = FrozenTaskPlan.from_plan(
        "Different original task",
        TaskPlan(
            task="Plan identity",
            subtasks=[SubTask(id="T1", role="Analyst", objective="Analyze")],
        ),
    )

    with pytest.raises(ValueError, match="different source task"):
        await orchestrator.run_planned_task(
            "Requested task",
            frozen_plan,
            get_treatment_preset("DEPENDENCY_AWARE").policy,
        )

    planner.assert_not_awaited()


@pytest.mark.asyncio
async def test_prepare_frozen_plan_generates_one_valid_deterministic_artifact(monkeypatch):
    plan = TaskPlan(
        task="Synthesized plan identity",
        subtasks=[
            SubTask(id="T1", role="Analyst", objective="Analyze task"),
            SubTask(id="T2", role="Reviewer", objective="Review task", dependencies=["T1"]),
        ],
    )
    planner = AsyncMock(return_value=plan)
    monkeypatch.setattr("core.orchestrator.TaskPlanner.create_plan", planner)
    orchestrator = MedhaOrchestrator(llm_client=GeminiClient(api_key="mock_key", client=MagicMock()))

    first = await orchestrator.prepare_frozen_plan("Frozen plan source task")
    second = await orchestrator.prepare_frozen_plan("Frozen plan source task")

    assert planner.await_count == 2
    assert first.serialized_plan == second.serialized_plan
    assert first.plan_hash == second.plan_hash
    assert first.load_plan("Frozen plan source task").subtasks[1].dependencies == ["T1"]


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
    from verification.evidence_ledger import EvidenceLedger

    # Pin the code default so a local .env MAX_CLAIMS_PER_TASK override cannot change this test.
    monkeypatch.setattr(EvidenceLedger.__init__, "__defaults__", ("", 50))
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
    assert "## Direct Answer\n_Unverified: final synthesis failed" in answer
    import main

    assert main._answer_for_console(answer, full=False).endswith("Partial specialist deliverable retained.")
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
async def test_deterministic_synthesis_includes_deliberation_claim_status_and_disagreement():
    from agents.synthesizer_agent import SynthesizerAgent

    mock_client = MagicMock()
    mock_client.generate_text = AsyncMock(
        side_effect=LLMQuotaExhaustedError("synthesis provider quota exhausted")
    )
    report = VerificationReport(
        total_claims=1,
        unverifiable_claims=[
            Claim(
                claim_id="response_claim",
                claim_text="The response-introduced assertion is not independently supported.",
                originating_agent="agent_T2",
                status=ClaimStatus.UNVERIFIABLE,
                unverified_evidence_references=["unverified-source"],
            )
        ],
    )
    deliberation_report = _make_deliberation_report("exec_synthesis")
    exchange = deliberation_report.exchanges[0]
    exchange.introduced_claims = ["The response-introduced assertion is not independently supported."]
    exchange.introduced_claim_ids = ["response_claim"]
    exchange.evidence_references = ["unverified-source"]
    deliberation_report.unresolved_disagreements = ["Specialists disagree about this assertion."]

    answer = await SynthesizerAgent(llm_client=mock_client).synthesize(
        task_description="Synthesize the audited response",
        specialist_results=[],
        verification_report=report,
        unresolved_questions=[],
        deliberation_report=deliberation_report,
    )

    assert "Deliberation Transcript (Unverified Context; Not Evidence)" in answer
    assert "Deliberation-introduced claim [response_claim;" in answer
    assert "UNVERIFIABLE" in answer
    assert "Agent-supplied evidence references (UNVERIFIED, not ledger evidence)" in answer
    assert "Specialists disagree about this assertion." in answer
    assert "disagreement is not a factual conclusion" in answer


@pytest.mark.asyncio
async def test_synthesis_prompt_trims_long_deliverables_but_keeps_claims(monkeypatch):
    from agents.synthesizer_agent import SynthesizerAgent
    from config import settings

    monkeypatch.setattr(settings, "MAX_INPUT_TOKENS_PER_CALL", 12000)
    monkeypatch.setattr(settings, "PROVIDER_TOKENS_PER_MINUTE", 8000)
    client = MagicMock()
    client.get_token_tracker.return_value = TokenTracker(name="synthesis_trim_test", essential_token_reserves={})
    client.generate_text = AsyncMock(return_value="## Direct Answer\nOK")
    report = VerificationReport(
        total_claims=1,
        accepted_claims=[
            Claim(claim_id="kept_claim", claim_text="This accepted claim must survive trimming.", originating_agent="a1")
        ],
    )

    await SynthesizerAgent(llm_client=client).synthesize(
        task_description="Long task",
        specialist_results=[
            {"role": f"Specialist {i}", "subtask_id": f"T{i}", "status": "COMPLETED", "output": "x" * 20000}
            for i in range(4)
        ],
        verification_report=report,
        unresolved_questions=[],
    )

    kwargs = client.generate_text.await_args.kwargs
    assert TokenTracker.estimate_tokens(kwargs["prompt"] + kwargs["system_instruction"]) <= int(8000 * 0.9) - 1536
    assert "This accepted claim must survive trimming." in kwargs["prompt"]
    assert "deliverable truncated to fit the model context" in kwargs["prompt"]


@pytest.mark.asyncio
async def test_deliberation_response_claim_is_registered_and_reviewed_independently(monkeypatch):
    claim_text = "The source specialist's assertion remains pending review."
    specialist_results = [
        TaskResult(
            subtask_id="T1",
            agent_id="agent_T1",
            role="Source Specialist",
            status=TaskStatus.COMPLETED,
            output="Completed source analysis.",
            claims=[claim_text],
        ),
        TaskResult(
            subtask_id="T2",
            agent_id="agent_T2",
            role="Target Specialist",
            status=TaskStatus.COMPLETED,
        ),
    ]

    def deliberation_outcome(execution_id):
        report = _make_deliberation_report(execution_id)
        exchange = report.exchanges[0]
        exchange.claim_reference = f"claim:{execution_id}:T1:0"
        exchange.introduced_claims = [claim_text]
        exchange.clarifications = ["The response clarifies scope only."]
        exchange.evidence_references = ["unverified-source"]
        return report

    events = []
    _, captured = _install_orchestrator_deliberation_stubs(
        monkeypatch,
        specialist_results,
        deliberation_outcome,
        events,
    )
    orchestrator = MedhaOrchestrator(
        llm_client=GeminiClient(api_key="mock_key", client=MagicMock())
    )

    result = await orchestrator.run_task("Review a deliberation response claim")

    assert len(result.claims) == 1
    claim = result.claims[0]
    exchange = result.deliberation_report.exchanges[0]
    assert exchange.introduced_claim_ids == [claim["claim_id"]]
    assert claim["status"] == ClaimStatus.PENDING
    assert {source["source_kind"] for source in claim["source_provenance"] if "source_kind" in source} == {
        "DELIBERATION_RESPONSE"
    }
    response_source = next(
        source for source in claim["source_provenance"]
        if source.get("source_kind") == "DELIBERATION_RESPONSE"
    )
    assert response_source["parent_claim_id"] == claim["claim_id"]
    assert claim["unverified_evidence_references"] == ["unverified-source"]
    assert len(captured["claims_at_review"]) == 1
    assert captured["claims_at_review"][0].status == ClaimStatus.PENDING
    assert captured["evidence_at_review"][claim["claim_id"]] == []
    reviewer_context = captured["review_kwargs"]["deliberation_context"]
    assert "UNVERIFIED CONTEXT ONLY; NOT REGISTERED EVIDENCE" in reviewer_context
    assert "The response clarifies scope only." in reviewer_context
    assert "unverified-source" in reviewer_context
    assert captured["synthesis_kwargs"]["deliberation_report"] is result.deliberation_report


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


@pytest.mark.asyncio
async def test_orchestrator_runs_deliberation_between_specialists_and_claim_extraction(
    monkeypatch,
):
    events = []
    specialist_result = TaskResult(
        subtask_id="T1",
        agent_id="agent_T1",
        role="Source Specialist",
        status=TaskStatus.COMPLETED,
        output="Completed source analysis.",
        claims=["A specialist assertion remains pending review."],
    )
    _, captured = _install_orchestrator_deliberation_stubs(
        monkeypatch,
        specialist_result,
        lambda execution_id: _make_deliberation_report(execution_id),
        events,
    )
    saved_records = []

    def capture_execution_record(record, filepath=None):
        saved_records.append(record.to_dict())
        return filepath

    monkeypatch.setattr("core.orchestrator.ExecutionRecord.save", capture_execution_record)
    mock_client = MagicMock()
    orchestrator = MedhaOrchestrator(llm_client=GeminiClient(api_key="mock_key", client=mock_client))

    result = await orchestrator.run_task("Run orchestration deliberation integration")

    assert events.index("specialist_execution") < events.index("deliberation")
    assert events.index("deliberation") < events.index("claim_extraction")
    coordinator_kwargs = captured["coordinator_kwargs"]
    assert coordinator_kwargs["execution_id"] == result.execution_id
    assert coordinator_kwargs["plan"] is not None
    assert coordinator_kwargs["task_results"] == [specialist_result]
    assert coordinator_kwargs["agent_factory"] is not None
    assert coordinator_kwargs["message_bus"] is captured["message_bus"]
    assert coordinator_kwargs["shared_context"] is captured["shared_context"]
    assert coordinator_kwargs["token_tracker"] is orchestrator.tracker
    assert result.deliberation_report is not None
    assert result.deliberation_report.exchanges[0].response_message_id == "msg_response"
    assert result.claims[0]["status"] == ClaimStatus.PENDING
    assert any(stage["stage"] == "DELIBERATING" for stage in result.workflow_stages)
    assert saved_records[0]["metadata"]["deliberation_report"]["execution_id"] == result.execution_id


@pytest.mark.asyncio
async def test_no_trigger_deliberation_does_not_downgrade_success(monkeypatch):
    events = []
    specialist_result = TaskResult(
        subtask_id="T1",
        agent_id="agent_T1",
        role="Source Specialist",
        status=TaskStatus.COMPLETED,
        output="A complete specialist output with no policy trigger.",
        claims=["A structured claim available for the independent review."],
    )
    no_trigger_report = lambda execution_id: DeliberationReport(
        execution_id=execution_id,
        triggered=False,
        termination_reason=DeliberationTerminationReason.NO_RELEVANT_TRIGGER,
    )
    _install_orchestrator_deliberation_stubs(
        monkeypatch,
        specialist_result,
        no_trigger_report,
        events,
    )
    orchestrator = MedhaOrchestrator(
        llm_client=GeminiClient(api_key="mock_key", client=MagicMock())
    )

    result = await orchestrator.run_task("No trigger task")

    assert result.status == "SUCCESS"
    assert result.deliberation_report.termination_reason == DeliberationTerminationReason.NO_RELEVANT_TRIGGER


@pytest.mark.asyncio
async def test_completed_but_empty_specialists_skip_deliberation(monkeypatch):
    events = []
    specialist_result = TaskResult(
        subtask_id="T1",
        agent_id="agent_T1",
        role="Empty Specialist",
        status=TaskStatus.COMPLETED,
    )
    _install_orchestrator_deliberation_stubs(monkeypatch, specialist_result, None, events)
    orchestrator = MedhaOrchestrator(
        llm_client=GeminiClient(api_key="mock_key", client=MagicMock())
    )

    result = await orchestrator.run_task("No usable specialist output")

    assert result.status == "SUCCESS"
    assert "coordinator_constructed" not in events
    assert result.deliberation_report.triggered is False
    assert "no usable output" in result.deliberation_report.diagnostics[0]


@pytest.mark.asyncio
async def test_all_specialist_failure_skips_coordinator(monkeypatch):
    events = []
    failed_result = TaskResult(
        subtask_id="T1",
        agent_id="agent_T1",
        role="Failing Specialist",
        status=TaskStatus.FAILED,
        error="specialist failed",
    )
    _install_orchestrator_deliberation_stubs(monkeypatch, failed_result, None, events)
    orchestrator = MedhaOrchestrator(
        llm_client=GeminiClient(api_key="mock_key", client=MagicMock())
    )

    result = await orchestrator.run_task("All specialists fail")

    assert result.status == "FAILED"
    assert "coordinator_constructed" not in events
    assert result.deliberation_report.termination_reason == DeliberationTerminationReason.AGENT_FAILURE
    assert "no specialist completed" in result.deliberation_report.diagnostics[0]


@pytest.mark.asyncio
async def test_cancelled_specialist_execution_never_starts_deliberation(monkeypatch):
    events = []
    plan = TaskPlan(task="Cancelled task", subtasks=[SubTask(id="T1", role="Source", objective="Analyze")])
    monkeypatch.setattr("core.orchestrator.TaskPlanner.create_plan", AsyncMock(return_value=plan))
    monkeypatch.setattr(
        "core.orchestrator.ExecutionEngine.execute_plan",
        AsyncMock(side_effect=asyncio.CancelledError()),
    )
    coordinator_calls = []

    class CoordinatorMustNotRun:
        def __init__(self, **kwargs):
            coordinator_calls.append(kwargs)

    monkeypatch.setattr("core.orchestrator.DeliberationCoordinator", CoordinatorMustNotRun)
    orchestrator = MedhaOrchestrator(
        llm_client=GeminiClient(api_key="mock_key", client=MagicMock())
    )

    with pytest.raises(asyncio.CancelledError):
        await orchestrator.run_task("Cancelled task")

    assert coordinator_calls == []


@pytest.mark.asyncio
async def test_coordinator_cancellation_stops_before_claim_extraction(monkeypatch):
    events = []
    specialist_result = TaskResult(
        subtask_id="T1",
        agent_id="agent_T1",
        role="Source Specialist",
        status=TaskStatus.COMPLETED,
        output="Completed output before deliberation cancellation.",
        claims=["A pending specialist claim."],
    )
    _install_orchestrator_deliberation_stubs(
        monkeypatch,
        specialist_result,
        lambda execution_id: _make_deliberation_report(
            execution_id,
            DeliberationTerminationReason.TASK_CANCELLED,
        ),
        events,
    )
    saved_records = []

    def capture_execution_record(record, filepath=None):
        saved_records.append(record.to_dict())
        return filepath

    monkeypatch.setattr("core.orchestrator.ExecutionRecord.save", capture_execution_record)
    orchestrator = MedhaOrchestrator(
        llm_client=GeminiClient(api_key="mock_key", client=MagicMock())
    )

    with pytest.raises(asyncio.CancelledError):
        await orchestrator.run_task("Cancel during deliberation")

    assert "deliberation" in events
    assert "claim_extraction" not in events
    assert saved_records[0]["status"] == "CANCELLED"
    assert saved_records[0]["metadata"]["deliberation_report"]["termination_reason"] == "TASK_CANCELLED"


@pytest.mark.parametrize(
    "termination_reason",
    [
        DeliberationTerminationReason.AGENT_FAILURE,
        DeliberationTerminationReason.TOKEN_BUDGET_EXHAUSTED,
        DeliberationTerminationReason.TIMEOUT,
    ],
)
@pytest.mark.asyncio
async def test_incomplete_deliberation_preserves_specialist_results_and_report(
    monkeypatch,
    termination_reason,
):
    events = []
    specialist_result = TaskResult(
        subtask_id="T1",
        agent_id="agent_T1",
        role="Source Specialist",
        status=TaskStatus.COMPLETED,
        output="Preserved specialist output.",
        claims=["Pending claim remains available."],
    )
    _install_orchestrator_deliberation_stubs(
        monkeypatch,
        specialist_result,
        lambda execution_id: _make_deliberation_report(execution_id, termination_reason),
        events,
    )
    orchestrator = MedhaOrchestrator(
        llm_client=GeminiClient(api_key="mock_key", client=MagicMock())
    )

    result = await orchestrator.run_task("Deliberation interruption")

    assert result.status == "PARTIAL_FAILURE"
    assert result.task_results[0]["output"] == "Preserved specialist output."
    assert result.deliberation_report.termination_reason == termination_reason
    assert result.deliberation_report.exchanges[0].response_message_id == "msg_response"
    assert any(stage["stage"] == "SYNTHESIZING" for stage in result.workflow_stages)
    assert result.workflow_stages[-1]["stage"] == "PARTIAL_FAILURE"
    assert "Deliberation incomplete" in result.final_answer or events.index("deliberation") < events.index("synthesis")
