import pytest

from communication.deliberation_policy import DeliberationPolicy, temporary_claim_reference
from config import settings
from models.deliberation_models import (
    DeliberationResponsePayload,
    DeliberationResponseType,
    DeliberationTerminationReason,
    DeliberationTriggerType,
)
from models.schemas import ClaimKind, TaskStatus
from models.task_models import SubTask, TaskPlan, TaskResult
from models.verification_models import Claim


def make_plan(*subtasks: SubTask) -> TaskPlan:
    return TaskPlan(task="deliberation test", subtasks=list(subtasks))


def result(
    subtask_id: str,
    agent_id: str,
    role: str,
    *,
    output: str = "",
    claims=None,
    assumptions=None,
    questions=None,
    evidence_items=None,
) -> TaskResult:
    return TaskResult(
        task_id="exec_test",
        subtask_id=subtask_id,
        agent_id=agent_id,
        role=role,
        status=TaskStatus.COMPLETED,
        output=output,
        claims=claims or [],
        assumptions=assumptions or [],
        unresolved_questions=questions or [],
        evidence_items=evidence_items or [],
    )


def test_models_serialize_and_response_has_no_verification_status():
    payload = DeliberationResponsePayload(
        execution_id="exec_test",
        exchange_id="exchange_test",
        correlation_id="corr_test",
        sender_agent_id="agent_b",
        recipient_agent_id="agent_a",
        response_type=DeliberationResponseType.EVIDENCE_RESPONSE,
        content="The requested reference is available.",
        claim_reference="claim:exec_test:T1:0",
        evidence_references=["evidence_1"],
        unresolved_questions=["External validation remains open."],
    )

    dumped = payload.model_dump(mode="json")

    assert dumped["response_type"] == "EVIDENCE_RESPONSE"
    assert dumped["claim_reference"] == "claim:exec_test:T1:0"
    assert dumped["evidence_references"] == ["evidence_1"]
    assert "status" not in dumped
    assert DeliberationTerminationReason.MAX_ROUNDS_REACHED.value == "MAX_ROUNDS_REACHED"


def test_related_contradictory_claims_trigger_one_challenge():
    plan = make_plan(
        SubTask(id="T1", role="Performance Architect", objective="Analyze database read latency"),
        SubTask(id="T2", role="Reliability Engineer", objective="Analyze database read latency"),
    )
    results = [
        result(
            "T1", "agent_a", "Performance Architect",
            output="database read latency analysis",
            claims=["Database read latency target is 50ms."],
        ),
        result(
            "T2", "agent_b", "Reliability Engineer",
            output="database read latency analysis",
            claims=["Database read latency target is 200ms."],
        ),
    ]

    report = DeliberationPolicy().evaluate("exec_test", plan, results)

    assert report.triggered is True
    assert len(report.triggers) == 1
    assert report.triggers[0].trigger_type == DeliberationTriggerType.CONFLICTING_CLAIMS
    assert report.triggers[0].claim_reference == temporary_claim_reference("exec_test", "T1", 0)
    assert report.triggers[0].target_agent_id == "agent_b"


def test_unrelated_findings_do_not_trigger():
    plan = make_plan(
        SubTask(id="T1", role="Privacy Specialist", objective="Review data retention"),
        SubTask(id="T2", role="Network Specialist", objective="Review packet routing"),
    )
    results = [
        result("T1", "agent_a", "Privacy Specialist", claims=["Student retention policy uses deletion windows."]),
        result("T2", "agent_b", "Network Specialist", claims=["Packet routing uses redundant links."]),
    ]

    report = DeliberationPolicy().evaluate("exec_test", plan, results)

    assert report.triggers == []
    assert report.termination_reason == DeliberationTerminationReason.NO_RELEVANT_TRIGGER


def test_unsupported_empirical_claim_triggers_evidence_request():
    plan = make_plan(
        SubTask(id="T1", role="Performance Architect", objective="Analyze database read latency"),
        SubTask(id="T2", role="Reliability Engineer", objective="Analyze database read latency"),
    )
    results = [
        result(
            "T1", "agent_a", "Performance Architect",
            output="database read latency evidence review",
            claims=["We measured database read latency of 50ms."],
        ),
        result(
            "T2", "agent_b", "Reliability Engineer",
            output="database read latency evidence review",
        ),
    ]

    report = DeliberationPolicy().evaluate("exec_test", plan, results)

    assert any(
        trigger.trigger_type == DeliberationTriggerType.UNSUPPORTED_EMPIRICAL_CLAIM
        for trigger in report.triggers
    )


def test_material_assumption_triggers_clarification():
    plan = make_plan(
        SubTask(id="T1", role="Architecture Specialist", objective="Design database network latency controls"),
        SubTask(id="T2", role="Reliability Specialist", objective="Validate database network latency controls"),
    )
    results = [
        result(
            "T1", "agent_a", "Architecture Specialist",
            output="database network latency design",
            assumptions=["Database network latency is stable."],
        ),
        result("T2", "agent_b", "Reliability Specialist", output="database network latency validation"),
    ]

    report = DeliberationPolicy().evaluate("exec_test", plan, results)

    assert any(
        trigger.trigger_type == DeliberationTriggerType.UNSUPPORTED_ASSUMPTION
        for trigger in report.triggers
    )


def test_relevant_unresolved_question_targets_related_specialist():
    plan = make_plan(
        SubTask(id="T1", role="Architecture Specialist", objective="Design database replication"),
        SubTask(id="T2", role="Reliability Specialist", objective="Validate database replication"),
    )
    results = [
        result(
            "T1", "agent_a", "Architecture Specialist",
            output="database replication design",
            questions=["How does database replication behave during failover?"],
        ),
        result("T2", "agent_b", "Reliability Specialist", output="database replication failover validation"),
    ]

    report = DeliberationPolicy().evaluate("exec_test", plan, results)

    trigger = next(
        item for item in report.triggers
        if item.trigger_type == DeliberationTriggerType.CRITICAL_UNRESOLVED_QUESTION
    )
    assert trigger.source_agent_id == "agent_a"
    assert trigger.target_agent_id == "agent_b"


def test_dependency_gap_targets_only_dependency_owner():
    plan = make_plan(
        SubTask(id="T1", role="Architecture Specialist", objective="Provide architecture"),
        SubTask(id="T2", role="Security Specialist", objective="Use architecture", dependencies=["T1"]),
        SubTask(id="T3", role="Privacy Specialist", objective="Independent privacy review"),
    )
    results = [
        result("T1", "agent_a", "Architecture Specialist"),
        result("T2", "agent_b", "Security Specialist", output="security review"),
        result("T3", "agent_c", "Privacy Specialist", output="privacy review"),
    ]

    report = DeliberationPolicy().evaluate("exec_test", plan, results)

    gaps = [
        trigger for trigger in report.triggers
        if trigger.trigger_type == DeliberationTriggerType.DEPENDENCY_INFORMATION_GAP
    ]
    assert len(gaps) == 1
    assert gaps[0].source_agent_id == "agent_b"
    assert gaps[0].target_agent_id == "agent_a"


def test_duplicate_triggers_and_limits_are_suppressed():
    plan = make_plan(
        SubTask(id="T1", role="Performance Architect", objective="Analyze database read latency"),
        SubTask(id="T2", role="Reliability Engineer", objective="Analyze database read latency"),
    )
    results = [
        result(
            "T1", "agent_a", "Performance Architect",
            output="database read latency analysis",
            claims=["Database read latency target is 50ms."],
        ),
        result(
            "T2", "agent_b", "Reliability Engineer",
            output="database read latency analysis",
            claims=["Database read latency target is 200ms."],
        ),
    ]
    policy = DeliberationPolicy(max_messages_per_execution=1)
    first = policy.evaluate("exec_test", plan, results)
    second = policy.evaluate(
        "exec_test",
        plan,
        results,
        existing_triggers=first.triggers,
        current_message_count=1,
    )

    assert len(first.triggers) == 1
    assert second.triggers == []

    claim_ref = first.triggers[0].claim_reference
    exhausted = policy.evaluate(
        "exec_test",
        plan,
        results,
        exchanges_by_claim={claim_ref: 2},
    )
    assert exhausted.triggers == []


def test_claim_objects_use_existing_claim_kind_and_ledger_reference():
    plan = make_plan(
        SubTask(id="T1", role="Performance Architect", objective="Review database latency"),
        SubTask(id="T2", role="Reliability Engineer", objective="Review database latency"),
    )
    results = [
        result("T1", "agent_a", "Performance Architect", output="database latency"),
        result("T2", "agent_b", "Reliability Engineer", output="database latency"),
    ]
    claims = [
        Claim(
            claim_id="ledger_claim_1",
            claim_text="We measured database latency of 50ms.",
            originating_agent="agent_a",
            originating_role="Performance Architect",
            subtask_id="T1",
            claim_kind=ClaimKind.EMPIRICAL_RESULT,
        )
    ]

    report = DeliberationPolicy().evaluate("exec_test", plan, results, claims=claims)

    trigger = next(
        item for item in report.triggers
        if item.trigger_type == DeliberationTriggerType.UNSUPPORTED_EMPIRICAL_CLAIM
    )
    assert trigger.claim_reference == "ledger_claim_1"
