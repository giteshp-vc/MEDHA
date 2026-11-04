from unittest.mock import AsyncMock, MagicMock
import pytest

from agents.analyst_agent import AnalystAgent
from agents.research_agent import ResearchAgent
from agents.specialist_agent import (
    SpecialistAgent,
    SpecialistOutputRecoveryError,
    SpecialistOutputSchema,
)
from agents.technical_agent import TechnicalAgent
from core.agent_factory import AgentFactory
from core.llm_client import GeminiClient
from core.planner import TaskPlanner
from models.agent_models import AgentConfig
from models.schemas import AgentStatus
from models.task_models import SubTask, TaskPlan
from utils.retry import (
    LLMBudgetExceededError,
    LLMQuotaExhaustedError,
    LLMStructuredOutputError,
    LLMTokenBudgetExceededError,
)


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
async def test_planner_creates_plan():
    """Verify TaskPlanner decomposes a task into subtasks dynamically."""
    mock_client = MagicMock()
    plan_json = """{
        "task": "Design scalable quantum algorithm",
        "subtasks": [
            {
                "id": "T1",
                "role": "Quantum Physicist",
                "objective": "Formulate Hamiltonian",
                "instructions": "Focus on qubit coherence",
                "dependencies": [],
                "expected_output": "Hamiltonian formulation"
            },
            {
                "id": "T2",
                "role": "Quantum Error Correction Specialist",
                "objective": "Design surface code",
                "instructions": "Estimate threshold",
                "dependencies": ["T1"],
                "expected_output": "Error budget"
            }
        ],
        "rationale": "Specialized quantum physics and error correction expertise required",
        "completion_criteria": ["Hamiltonian verified", "Error threshold met"]
    }"""
    mock_client.aio.models.generate_content = AsyncMock(return_value=_mock_llm_response(plan_json))
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)
    planner = TaskPlanner(llm_client=gemini_client)

    plan = await planner.create_plan("Design a quantum algorithm")

    assert isinstance(plan, TaskPlan)
    assert len(plan.subtasks) == 2
    assert plan.subtasks[0].role == "Quantum Physicist"
    assert plan.subtasks[1].role == "Quantum Error Correction Specialist"
    assert plan.subtasks[1].dependencies == ["T1"]
    planner_prompt = mock_client.aio.models.generate_content.call_args.kwargs["config"].system_instruction
    assert "architecture, privacy, security, reliability, and deployment reviews" in planner_prompt


@pytest.mark.asyncio
async def test_planner_fallback_on_invalid_json():
    """Verify TaskPlanner falls back to emergency plan if LLM output fails."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=_mock_llm_response("NOT VALID JSON"))
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)
    planner = TaskPlanner(llm_client=gemini_client)

    plan = await planner.create_plan("Build an emergency shelter")

    assert isinstance(plan, TaskPlan)
    assert len(plan.subtasks) >= 2
    assert "Primary Domain Specialist" in plan.subtasks[0].role


@pytest.mark.asyncio
async def test_planner_does_not_retry_local_token_budget_exhaustion():
    mock_client = MagicMock()
    mock_client.generate_structured = AsyncMock(
        side_effect=LLMTokenBudgetExceededError("local task token budget exhausted")
    )
    mock_client.generate_text = AsyncMock()
    planner = TaskPlanner(llm_client=mock_client)

    with pytest.raises(LLMBudgetExceededError, match="local task token budget exhausted"):
        await planner.create_plan("Analyze a system")

    mock_client.generate_structured.assert_awaited_once()
    mock_client.generate_text.assert_not_awaited()


def test_agent_factory_creates_diverse_roles():
    """Verify AgentFactory routes roles to appropriate classes and caches duplicates."""
    mock_client = MagicMock()
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)
    factory = AgentFactory(llm_client=gemini_client)

    s1 = SubTask(id="T1", role="Deep Learning Research Specialist", objective="Survey transformers")
    s2 = SubTask(id="T2", role="Systems Performance Analyst", objective="Analyze latency")
    s3 = SubTask(id="T3", role="Database Architect", objective="Design schema")
    s4 = SubTask(id="T4", role="Epidemiologist", objective="Model transmission rate")

    a1 = factory.create_from_subtask(s1, task_id="task_1")
    a2 = factory.create_from_subtask(s2, task_id="task_1")
    a3 = factory.create_from_subtask(s3, task_id="task_1")
    a4 = factory.create_from_subtask(s4, task_id="task_1")

    assert isinstance(a1, ResearchAgent)
    assert isinstance(a2, AnalystAgent)
    assert isinstance(a3, TechnicalAgent)
    assert isinstance(a4, SpecialistAgent)
    assert a4.role == "Epidemiologist"

    # Verify duplicate prevention
    a1_dup = factory.create_from_subtask(s1, task_id="task_1")
    assert a1_dup.agent_id == a1.agent_id

    # Verify Reviewer and Synthesizer creation
    rev = factory.create_reviewer()
    synth = factory.create_synthesizer()
    assert rev is not None
    assert synth is not None
    assert len(factory.list_agents()) == 6


@pytest.mark.asyncio
async def test_specialist_agent_execution():
    """Verify SpecialistAgent executes and records findings, claims, and assumptions."""
    mock_client = MagicMock()
    agent_output_json = """{
        "deliverable": "The distributed ledger should use Raft consensus.",
        "findings": ["Raft provides strict leader election"],
        "claims": ["Raft achieves sub-10ms election in local networks", "Throughput reaches 5000 ops/sec"],
        "evidence_references": ["Ongaro & Ousterhout 2014 paper"],
        "assumptions": ["Network partitions are transient"],
        "unresolved_questions": ["Behavior under asymmetric Byzantine faults"]
    }"""
    mock_client.aio.models.generate_content = AsyncMock(return_value=_mock_llm_response(agent_output_json))
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)

    config = AgentConfig(
        agent_id="test_agent",
        role="Distributed Systems Architect",
        objective="Select consensus mechanism",
        instructions="Analyze Raft vs Paxos",
    )
    agent = SpecialistAgent(config=config, llm_client=gemini_client)

    response = await agent.execute(context={"task_context": "High-throughput cluster"})

    assert response.status == AgentStatus.COMPLETED
    assert len(response.findings) == 1
    assert len(response.claims) == 2
    assert "Raft achieves sub-10ms election in local networks" in response.claims
    assert len(response.assumptions) == 1
    assert len(response.unresolved_questions) == 1
    assert agent.status == AgentStatus.COMPLETED


def _create_test_specialist(client):
    config = AgentConfig(
        agent_id="recovery_test_agent",
        role="Privacy Specialist",
        objective="Assess privacy and data handling risks",
    )
    return SpecialistAgent(config=config, llm_client=client)


@pytest.mark.asyncio
async def test_specialist_preserves_nested_findings():
    mock_client = MagicMock()
    response_text = (
        '{"deliverable":"Review complete",'
        '"findings":["Data minimization",["Retention limits",["Access controls"]],'
        '{"risk":"secondary use"}]}'
    )
    mock_client.aio.models.generate_content = AsyncMock(
        return_value=_mock_llm_response(response_text)
    )
    agent = _create_test_specialist(GeminiClient(api_key="mock_key", client=mock_client))

    response = await agent.execute()

    assert response.findings == [
        "Data minimization",
        "Retention limits",
        "Access controls",
        '{"risk": "secondary use"}',
    ]


@pytest.mark.asyncio
async def test_specialist_uses_one_schema_correction_before_fallback():
    mock_client = MagicMock()
    initial_error = LLMStructuredOutputError(
        "SpecialistOutputSchema", '{"findings":[["retained finding"]]}', ValueError("bad shape")
    )
    corrected = SpecialistOutputSchema(
        deliverable="Corrected output",
        findings=["Retained finding"],
    )
    mock_client.generate_structured = AsyncMock(side_effect=[initial_error, corrected])
    mock_client.generate_text = AsyncMock()
    agent = _create_test_specialist(mock_client)

    response = await agent.execute()

    assert response.content == "Corrected output"
    assert response.findings == ["Retained finding"]
    assert mock_client.generate_structured.await_count == 2
    assert mock_client.generate_structured.await_args.kwargs["max_retries"] == 0
    mock_client.generate_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_specialist_fallback_parses_markdown_fenced_json():
    mock_client = MagicMock()
    initial_error = LLMStructuredOutputError("SpecialistOutputSchema", "not json", ValueError("invalid"))
    correction_error = LLMStructuredOutputError("SpecialistOutputSchema", "still not json", ValueError("invalid"))
    fallback = (
        'Here is the repaired response:\n```json\n'
        '{"deliverable":"Recovered", "findings":["Privacy finding"]}\n```\n'
    )
    mock_client.generate_structured = AsyncMock(side_effect=[initial_error, correction_error])
    mock_client.generate_text = AsyncMock(return_value=fallback)
    agent = _create_test_specialist(mock_client)

    response = await agent.execute()

    assert response.content == "Recovered"
    assert response.findings == ["Privacy finding"]
    mock_client.generate_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_specialist_malformed_json_fallback_fails_with_original_diagnostic():
    mock_client = MagicMock()
    initial_error = LLMStructuredOutputError(
        "SpecialistOutputSchema", "malformed original", ValueError("original schema error")
    )
    correction_error = LLMStructuredOutputError(
        "SpecialistOutputSchema", "malformed correction", ValueError("correction schema error")
    )
    mock_client.generate_structured = AsyncMock(side_effect=[initial_error, correction_error])
    mock_client.generate_text = AsyncMock(return_value="not JSON")
    agent = _create_test_specialist(mock_client)

    with pytest.raises(SpecialistOutputRecoveryError) as error:
        await agent.execute()

    message = str(error.value)
    assert "original schema error" in message
    assert "fallback parsing failed" in message
    assert "No valid JSON object found" in message
    assert agent.status == AgentStatus.FAILED


@pytest.mark.asyncio
async def test_specialist_empty_fallback_is_explicit():
    mock_client = MagicMock()
    initial_error = LLMStructuredOutputError("SpecialistOutputSchema", "", ValueError("empty original"))
    correction_error = LLMStructuredOutputError("SpecialistOutputSchema", "", ValueError("empty correction"))
    mock_client.generate_structured = AsyncMock(side_effect=[initial_error, correction_error])
    mock_client.generate_text = AsyncMock(return_value="  ")
    agent = _create_test_specialist(mock_client)

    with pytest.raises(SpecialistOutputRecoveryError, match="Empty response"):
        await agent.execute()


@pytest.mark.asyncio
async def test_specialist_does_not_recover_non_retryable_provider_error():
    mock_client = MagicMock()
    quota_error = LLMQuotaExhaustedError("Daily quota exhausted")
    mock_client.generate_structured = AsyncMock(side_effect=quota_error)
    mock_client.generate_text = AsyncMock()
    agent = _create_test_specialist(mock_client)

    with pytest.raises(LLMQuotaExhaustedError, match="Daily quota exhausted"):
        await agent.execute()

    mock_client.generate_structured.assert_awaited_once()
    mock_client.generate_text.assert_not_awaited()

