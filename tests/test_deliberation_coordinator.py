import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from communication.deliberation_coordinator import DeliberationCoordinator
from communication.deliberation_policy import DeliberationPolicy
from communication.message_bus import MessageBus
from communication.shared_context import SharedContext
from models.deliberation_models import (
    DeliberationResponsePayload,
    DeliberationResponseType,
    DeliberationTerminationReason,
    DeliberationTrigger,
    DeliberationTriggerType,
)
from models.schemas import TaskStatus
from models.task_models import SubTask, TaskPlan, TaskResult
from utils.token_tracker import TokenTracker


class StaticPolicy:
    def __init__(self, triggers):
        self.triggers = triggers

    def evaluate_triggers(self, *args, **kwargs):
        return list(self.triggers)


def make_plan():
    return TaskPlan(
        task="Coordinator test",
        subtasks=[
            SubTask(id="T1", role="Source", objective="Produce a finding"),
            SubTask(id="T2", role="Target", objective="Respond to a finding"),
        ],
    )


def make_results():
    return [
        TaskResult(
            task_id="exec_coord",
            subtask_id="T1",
            agent_id="agent_source",
            role="Source",
            status=TaskStatus.COMPLETED,
            output="Source output",
            claims=["Database read latency target is 50ms."],
        ),
        TaskResult(
            task_id="exec_coord",
            subtask_id="T2",
            agent_id="agent_target",
            role="Target",
            status=TaskStatus.COMPLETED,
            output="Target output",
        ),
    ]


def make_trigger(exchange_id="exchange_1", claim_reference="claim:exec_coord:T1:0"):
    return DeliberationTrigger(
        trigger_id=f"trigger_{exchange_id}",
        execution_id="exec_coord",
        exchange_id=exchange_id,
        correlation_id=f"corr_{exchange_id}",
        trigger_type=DeliberationTriggerType.CONFLICTING_CLAIMS,
        claim_reference=claim_reference,
        source_agent_id="agent_source",
        target_agent_id="agent_target",
        source_subtask_id="T1",
        target_subtask_id="T2",
        reason="Conflicting related claims.",
        related_claim_text="Database read latency target is 50ms.",
    )


def make_agents(client):
    factory = MagicMock()
    source = MagicMock(agent_id="agent_source", model_name="source-model", temperature=0.2)
    target = MagicMock(agent_id="agent_target", model_name="target-model", temperature=0.1)
    source.role = "Source"
    target.role = "Target"
    source.client = client
    target.client = client
    factory.get_agent.side_effect = lambda agent_id: {
        "agent_source": source,
        "agent_target": target,
    }.get(agent_id)
    return factory, source, target


def make_bus():
    bus = MessageBus(max_messages_per_execution=30, max_messages_per_agent=8)
    bus.register_agent("agent_source", execution_id="exec_coord", subtask_id="T1")
    bus.register_agent("agent_target", execution_id="exec_coord", subtask_id="T2")
    return bus


def make_response(trigger, response_type=DeliberationResponseType.CLARIFICATION, **kwargs):
    return DeliberationResponsePayload(
        execution_id=trigger.execution_id,
        exchange_id=trigger.exchange_id,
        correlation_id=trigger.correlation_id,
        sender_agent_id=trigger.target_agent_id,
        recipient_agent_id=trigger.source_agent_id,
        response_type=response_type,
        content="Target clarification",
        claim_reference=trigger.claim_reference,
        **kwargs,
    )


def make_coordinator(client, policy, **kwargs):
    factory, source, target = make_agents(client)
    bus = make_bus()
    tracker = TokenTracker(
        max_tokens_per_task=10000,
        max_input_tokens_per_call=10000,
        max_output_tokens_per_call=2048,
        essential_token_reserves={},
    )
    coordinator = DeliberationCoordinator(
        execution_id="exec_coord",
        task_results=make_results(),
        plan=make_plan(),
        agent_factory=factory,
        message_bus=bus,
        shared_context=SharedContext(task_id="exec_coord", task_description="Coordinator test"),
        token_tracker=tracker,
        policy=policy,
        **kwargs,
    )
    return coordinator, bus, tracker, source, target


@pytest.mark.asyncio
async def test_no_triggers_make_no_provider_calls():
    client = MagicMock()
    client.generate_structured = AsyncMock()
    coordinator, _, _, _, _ = make_coordinator(client, StaticPolicy([]))

    report = await coordinator.run()

    assert report.triggered is False
    assert report.termination_reason == DeliberationTerminationReason.NO_RELEVANT_TRIGGER
    client.generate_structured.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_specialist_stops_deliberation_without_provider_call():
    client = MagicMock()
    client.generate_structured = AsyncMock()
    coordinator, _, _, _, _ = make_coordinator(client, StaticPolicy([make_trigger()]))
    coordinator.task_results[0].status = TaskStatus.FAILED
    coordinator.task_results[0].error = "specialist failed"

    report = await coordinator.run()

    assert report.termination_reason == DeliberationTerminationReason.AGENT_FAILURE
    assert report.exchanges == []
    client.generate_structured.assert_not_awaited()


@pytest.mark.asyncio
async def test_challenge_targets_only_intended_agent_without_rerunning_execute():
    trigger = make_trigger()
    client = MagicMock()
    client.generate_structured = AsyncMock(return_value=make_response(trigger))
    coordinator, bus, _, source, target = make_coordinator(client, StaticPolicy([trigger]))

    report = await coordinator.run()

    assert client.generate_structured.await_count == 1
    call = client.generate_structured.await_args.kwargs
    assert call["model"] == "target-model"
    assert call["temperature"] == 0.1
    source.execute.assert_not_called()
    target.execute.assert_not_called()
    assert len(report.exchanges) == 1
    assert report.exchanges[0].response_type == DeliberationResponseType.CLARIFICATION
    messages = bus.get_history(task_id="exec_coord")
    assert len(messages) == 2
    assert messages[0].sender_id == "agent_source"
    assert messages[1].sender_id == "agent_target"
    assert messages[1].reply_to_message_id == messages[0].message_id


@pytest.mark.asyncio
async def test_prompt_includes_source_and_target_evidence_references():
    trigger = make_trigger()
    client = MagicMock()
    client.generate_structured = AsyncMock(return_value=make_response(trigger))
    coordinator, _, _, _, _ = make_coordinator(client, StaticPolicy([trigger]))
    coordinator.task_results[0].evidence_items = [{"reference": "source-ref"}]
    coordinator.task_results[1].evidence_items = [{"reference": "target-ref"}]

    await coordinator.run()

    prompt = client.generate_structured.await_args.kwargs["prompt"]
    assert "source-ref" in prompt
    assert "target-ref" in prompt


@pytest.mark.asyncio
async def test_prompt_states_expected_sender_and_recipient_ids():
    trigger = make_trigger()
    client = MagicMock()
    client.generate_structured = AsyncMock(return_value=make_response(trigger))
    coordinator, _, _, _, _ = make_coordinator(client, StaticPolicy([trigger]))

    await coordinator.run()

    prompt = client.generate_structured.await_args.kwargs["prompt"]
    assert f"use as sender_agent_id): {trigger.target_agent_id}" in prompt
    assert f"use as recipient_agent_id): {trigger.source_agent_id}" in prompt


@pytest.mark.asyncio
async def test_invalid_response_identity_is_rejected():
    trigger = make_trigger()
    client = MagicMock()
    invalid = make_response(trigger).model_copy(update={"sender_agent_id": "spoofed"})
    client.generate_structured = AsyncMock(return_value=invalid)
    coordinator, bus, _, _, _ = make_coordinator(client, StaticPolicy([trigger]))

    report = await coordinator.run()

    assert report.exchanges[0].response_message_id is None
    assert "sender identity" in report.exchanges[0].diagnostic.lower()
    assert len(bus.get_history(task_id="exec_coord")) == 1
    assert report.termination_reason == DeliberationTerminationReason.AGENT_FAILURE


@pytest.mark.asyncio
async def test_cross_execution_reply_is_rejected_by_message_bus():
    trigger = make_trigger()
    client = MagicMock()
    invalid = make_response(trigger).model_copy(update={"execution_id": "other_execution"})
    client.generate_structured = AsyncMock(return_value=invalid)
    coordinator, bus, _, _, _ = make_coordinator(client, StaticPolicy([trigger]))

    report = await coordinator.run()

    assert report.exchanges[0].response_message_id is None
    assert "execution identity" in report.exchanges[0].diagnostic.lower()


@pytest.mark.asyncio
async def test_duplicate_exchange_is_suppressed():
    trigger = make_trigger()
    client = MagicMock()
    client.generate_structured = AsyncMock(return_value=make_response(trigger))
    coordinator, bus, _, _, _ = make_coordinator(client, StaticPolicy([trigger]))
    existing = await bus.publish(
        __import__("communication.message", fromlist=["Message"]).Message(
            sender_id="agent_source",
            recipient_id="agent_target",
            execution_id="exec_coord",
            task_id="exec_coord",
            message_type=__import__("models.schemas", fromlist=["MessageType"]).MessageType.CHALLENGE,
            content="existing",
            correlation_id=trigger.correlation_id,
            metadata={"deliberation_exchange": True, "exchange_id": trigger.exchange_id},
        )
    )
    assert existing is True

    report = await coordinator.run()

    assert report.exchanges[0].response_message_id is None
    assert "duplicate" in report.exchanges[0].diagnostic.lower()
    client.generate_structured.assert_not_awaited()


@pytest.mark.asyncio
async def test_max_exchanges_per_claim_is_enforced():
    first = make_trigger("exchange_1")
    second = make_trigger("exchange_2")
    client = MagicMock()
    client.generate_structured = AsyncMock(side_effect=[make_response(first), make_response(second)])
    coordinator, _, _, _, _ = make_coordinator(client, StaticPolicy([first, second]), max_exchanges_per_claim=1)

    report = await coordinator.run()

    assert len(report.exchanges) == 1
    assert client.generate_structured.await_count == 1
    assert report.termination_reason == DeliberationTerminationReason.MAX_EXCHANGES_REACHED


@pytest.mark.asyncio
async def test_message_limit_stops_response_publication():
    trigger = make_trigger()
    client = MagicMock()
    client.generate_structured = AsyncMock(return_value=make_response(trigger))
    coordinator, _, _, _, _ = make_coordinator(client, StaticPolicy([trigger]), max_messages=1)

    report = await coordinator.run()

    assert report.exchanges[0].request_message_id is None
    assert report.exchanges[0].response_message_id is None
    assert "cannot fit a request/response pair" in report.exchanges[0].diagnostic.lower()
    client.generate_structured.assert_not_awaited()


@pytest.mark.asyncio
async def test_maximum_round_termination_is_recorded():
    trigger = make_trigger()
    client = MagicMock()
    client.generate_structured = AsyncMock(
        return_value=make_response(trigger, unresolved_questions=["A follow-up remains."])
    )
    coordinator, _, _, _, _ = make_coordinator(
        client,
        StaticPolicy([trigger]),
        max_rounds=1,
    )

    report = await coordinator.run()

    assert report.termination_reason == DeliberationTerminationReason.MAX_ROUNDS_REACHED
    assert report.rounds[0].round_number == 1


@pytest.mark.asyncio
async def test_per_agent_message_limit_skips_additional_exchange():
    first = make_trigger("exchange_1")
    second = make_trigger("exchange_2", claim_reference="claim:exec_coord:T1:1")
    client = MagicMock()
    client.generate_structured = AsyncMock(return_value=make_response(first))
    coordinator, _, _, _, _ = make_coordinator(
        client,
        StaticPolicy([first, second]),
        max_messages_per_agent=1,
    )

    report = await coordinator.run()

    assert len(report.exchanges) == 1
    assert client.generate_structured.await_count == 1


@pytest.mark.asyncio
async def test_token_budget_exhaustion_stops_provider_call():
    trigger = make_trigger()
    client = MagicMock()
    client.generate_structured = AsyncMock(return_value=make_response(trigger))
    coordinator, _, _, _, _ = make_coordinator(client, StaticPolicy([trigger]), max_tokens=1)

    report = await coordinator.run()

    assert report.exchanges[0].response_message_id is None
    assert "token budget" in report.exchanges[0].diagnostic.lower()
    client.generate_structured.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_usage_is_reconciled_and_estimates_are_preserved():
    trigger = make_trigger()
    client = MagicMock()
    tracker_holder = {}

    async def generate(**kwargs):
        await tracker_holder["tracker"].add_usage(
            prompt_tokens=3,
            candidate_tokens=2,
            total_tokens=5,
            usage_available=True,
        )
        return make_response(trigger)

    client.generate_structured = AsyncMock(side_effect=generate)
    coordinator, _, tracker, _, _ = make_coordinator(client, StaticPolicy([trigger]))
    tracker_holder["tracker"] = tracker

    report = await coordinator.run()

    assert report.total_tokens == 5
    assert "provider" in tracker.get_summary_sync()["usage_source"]


@pytest.mark.asyncio
async def test_missing_provider_usage_is_conservatively_estimated():
    trigger = make_trigger()
    client = MagicMock()
    coordinator, _, tracker, _, _ = make_coordinator(client, StaticPolicy([trigger]))

    async def generate(**kwargs):
        async with tracker.track_call(
            estimated_input_tokens=tracker.estimate_tokens(kwargs["prompt"]),
            max_output_tokens=kwargs["max_tokens"],
        ):
            await tracker.add_usage(usage_available=False)
        return make_response(trigger)

    client.generate_structured = AsyncMock(side_effect=generate)

    async with tracker.task_budget(essential_reserves={}):
        report = await coordinator.run()

    assert report.total_tokens is not None
    assert report.total_tokens > 0
    assert tracker.get_summary_sync()["usage_source"] == "estimated"


@pytest.mark.asyncio
async def test_actual_usage_over_deliberation_budget_stops_additional_calls():
    first = make_trigger("exchange_1")
    second = make_trigger("exchange_2", claim_reference="claim:exec_coord:T1:1")
    client = MagicMock()
    coordinator, _, tracker, _, _ = make_coordinator(
        client,
        StaticPolicy([first, second]),
        max_tokens=3000,
    )

    async def generate(**kwargs):
        await tracker.add_usage(prompt_tokens=3000, candidate_tokens=1000, total_tokens=4000)
        return make_response(first)

    client.generate_structured = AsyncMock(side_effect=generate)
    report = await coordinator.run()

    assert client.generate_structured.await_count == 1
    assert report.termination_reason == DeliberationTerminationReason.TOKEN_BUDGET_EXHAUSTED
    assert any("exceeded MAX_DELIBERATION_TOKENS" in item for item in report.diagnostics)


@pytest.mark.asyncio
async def test_timeout_preserves_completed_exchanges():
    first = make_trigger("exchange_1")
    second = make_trigger("exchange_2")
    client = MagicMock()
    calls = 0

    async def generate(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            await asyncio.sleep(0.05)
        return make_response(first if calls == 1 else second)

    client.generate_structured = AsyncMock(side_effect=generate)
    coordinator, _, _, _, _ = make_coordinator(
        client,
        StaticPolicy([first, second]),
        timeout_seconds=0.01,
    )

    report = await coordinator.run()

    assert len(report.exchanges) >= 1
    assert report.termination_reason == DeliberationTerminationReason.TIMEOUT


@pytest.mark.asyncio
async def test_agent_failure_preserves_previous_exchange():
    first = make_trigger("exchange_1")
    second = make_trigger("exchange_2")
    client = MagicMock()
    client.generate_structured = AsyncMock(side_effect=[make_response(first), RuntimeError("target failed")])
    coordinator, _, _, _, _ = make_coordinator(client, StaticPolicy([first, second]))

    report = await coordinator.run()

    assert len(report.exchanges) == 2
    assert report.exchanges[0].response_message_id is not None
    assert "failed" in report.exchanges[1].diagnostic.lower()
    assert report.termination_reason == DeliberationTerminationReason.AGENT_FAILURE


@pytest.mark.asyncio
async def test_cancellation_returns_cancelled_report():
    trigger = make_trigger()
    client = MagicMock()
    started = asyncio.Event()

    async def generate(**kwargs):
        started.set()
        await asyncio.Event().wait()

    client.generate_structured = AsyncMock(side_effect=generate)
    coordinator, _, _, _, _ = make_coordinator(client, StaticPolicy([trigger]))
    task = asyncio.create_task(coordinator.run())
    await started.wait()
    task.cancel()

    report = await task

    assert report.termination_reason == DeliberationTerminationReason.TASK_CANCELLED


@pytest.mark.asyncio
async def test_disagreement_is_preserved_and_agreement_does_not_change_claims():
    disagreement = make_trigger("exchange_disagreement")
    agreement = make_trigger("exchange_agreement", claim_reference="claim:exec_coord:T1:1")
    client = MagicMock()
    response_count = 0

    async def respond(**kwargs):
        nonlocal response_count
        response_count += 1
        if response_count == 1:
            return make_response(
                disagreement,
                DeliberationResponseType.DISAGREEMENT,
                disagreement=True,
                unresolved_questions=["External test remains necessary."],
            )
        if response_count == 2:
            return make_response(agreement, DeliberationResponseType.AGREEMENT)
        prompt_fields = dict(
            line.split(": ", 1) for line in kwargs["prompt"].splitlines() if ": " in line
        )
        return DeliberationResponsePayload(
            execution_id=prompt_fields["Execution ID"],
            exchange_id=prompt_fields["Exchange ID"],
            correlation_id=prompt_fields["Correlation ID"],
            sender_agent_id="agent_source",
            recipient_agent_id="agent_target",
            response_type=DeliberationResponseType.CLARIFICATION,
            content="No further evidence is available.",
            claim_reference=prompt_fields["Claim reference"],
            reply_to_message_id=prompt_fields["Reply to message ID"],
        )

    client.generate_structured = AsyncMock(side_effect=respond)
    coordinator, bus, _, _, _ = make_coordinator(
        client,
        StaticPolicy([disagreement, agreement]),
        max_tokens=20000,
    )

    report = await coordinator.run()

    assert report.unresolved_disagreements
    assert report.termination_reason == DeliberationTerminationReason.UNRESOLVED_DISAGREEMENT, report.exchanges[-1].diagnostic
    assert len(report.rounds) == 1
    assert client.generate_structured.await_count == 2
    assert report.suppressed_triggers
    assert report.triggers[0].originating_response_message_id is None
    assert all("evidence" not in str(message.message_type).lower() for message in bus.get_history(task_id="exec_coord") if message.sender_id == "agent_target" and message.metadata.get("exchange_id") == "exchange_agreement")


@pytest.mark.asyncio
async def test_invalid_reply_to_is_rejected_and_agent_evidence_stays_unverified():
    trigger = make_trigger()
    client = MagicMock()
    invalid = make_response(
        trigger,
        DeliberationResponseType.EVIDENCE_RESPONSE,
        evidence_references=["agent_asserted_ref"],
        reply_to_message_id="forged_message_id",
    )
    client.generate_structured = AsyncMock(return_value=invalid)
    coordinator, bus, _, _, _ = make_coordinator(client, StaticPolicy([trigger]))
    original_claims = list(coordinator.task_results[0].claims)

    report = await coordinator.run()

    assert report.exchanges[0].response_message_id is None
    assert "reply target" in report.exchanges[0].diagnostic.lower()
    assert coordinator.task_results[0].claims == original_claims
    assert all(
        "agent_asserted_ref" not in message.evidence_references
        for message in bus.get_history(task_id="exec_coord")
    )


def make_incremental_coordinator(client, response, **kwargs):
    coordinator, bus, tracker, source, target = make_coordinator(
        client,
        DeliberationPolicy(),
        **kwargs,
    )
    coordinator.task_results[0].unresolved_questions = [
        "Which database read latency target should reliability validate?"
    ]
    coordinator.task_results[1].output = "Database read latency target validation context."

    async def respond(**call_kwargs):
        prompt = call_kwargs["prompt"]
        fields = dict(line.split(": ", 1) for line in prompt.splitlines() if ": " in line)
        if callable(response):
            value = response(fields, call_kwargs)
            if asyncio.iscoroutine(value):
                return await value
            return value
        return response.model_copy(update={
            "execution_id": fields["Execution ID"],
            "exchange_id": fields["Exchange ID"],
            "correlation_id": fields["Correlation ID"],
            "sender_agent_id": fields.get("Target specialist context", "agent_target"),
            "reply_to_message_id": fields["Reply to message ID"],
        })

    client.generate_structured = AsyncMock(side_effect=respond)
    return coordinator, bus, tracker, source, target


@pytest.mark.asyncio
async def test_response_introduced_contradiction_targets_original_claim_owner():
    client = MagicMock()

    async def response(fields, kwargs):
        target_agent = "agent_target" if kwargs["model"] == "target-model" else "agent_source"
        recipient = "agent_source" if target_agent == "agent_target" else "agent_target"
        if kwargs["model"] == "target-model":
            return DeliberationResponsePayload(
                execution_id=fields["Execution ID"],
                exchange_id=fields["Exchange ID"],
                correlation_id=fields["Correlation ID"],
                sender_agent_id=target_agent,
                recipient_agent_id=recipient,
                response_type=DeliberationResponseType.CLARIFICATION,
                content="The measured target differs.",
                new_claims=["Database read latency target is 500ms."],
                useful_new_information=True,
                reply_to_message_id=fields["Reply to message ID"],
            )
        return DeliberationResponsePayload(
            execution_id=fields["Execution ID"],
            exchange_id=fields["Exchange ID"],
            correlation_id=fields["Correlation ID"],
            sender_agent_id=target_agent,
            recipient_agent_id=recipient,
            response_type=DeliberationResponseType.DISAGREEMENT,
            content="The two latency targets remain inconsistent.",
            disagreement=True,
            reply_to_message_id=fields["Reply to message ID"],
        )

    coordinator, bus, _, _, _ = make_incremental_coordinator(client, response, max_rounds=2)
    coordinator.task_results.append(
        TaskResult(
            task_id="exec_coord",
            subtask_id="T3",
            agent_id="agent_unrelated",
            role="Quantum Researcher",
            status=TaskStatus.COMPLETED,
            output="Quantum processors require cryogenic environments.",
            claims=["Quantum processors require cryogenic environments."],
        )
    )
    policy_calls = []
    policy = coordinator.policy
    evaluate_triggers = policy.evaluate_triggers

    def observe_policy_inputs(execution_id, plan, task_results, **kwargs):
        policy_calls.append([item.agent_id for item in task_results])
        return evaluate_triggers(execution_id, plan, task_results, **kwargs)

    policy.evaluate_triggers = observe_policy_inputs

    report = await coordinator.run()

    incremental = [
        trigger for trigger in report.triggers
        if trigger.originating_response_message_id
    ]
    assert incremental
    assert incremental[0].trigger_type == DeliberationTriggerType.CONFLICTING_CLAIMS
    assert incremental[0].target_agent_id == "agent_source"
    assert incremental[0].target_selection_rationale
    assert incremental[0].originating_response_message_id == report.exchanges[0].response_message_id
    assert incremental[0].reply_to_message_id == report.exchanges[0].response_message_id
    assert report.exchanges[0].introduced_claim_references
    assert report.exchanges[0].introduced_claim_references[0].startswith("claim:exec_coord:")
    assert any(
        suppression.originating_response_message_id == report.exchanges[0].response_message_id
        for suppression in report.suppressed_triggers
    )
    assert len(policy_calls) >= 2
    assert "agent_unrelated" not in policy_calls[1]
    assert bus.get_history(task_id="exec_coord")[-1].sender_id == "agent_source"


@pytest.mark.asyncio
async def test_response_evidence_reference_is_preserved_but_not_verified():
    client = MagicMock()

    def response(fields, kwargs):
        return DeliberationResponsePayload(
            execution_id=fields["Execution ID"],
            exchange_id=fields["Exchange ID"],
            correlation_id=fields["Correlation ID"],
            sender_agent_id="agent_target",
            recipient_agent_id="agent_source",
            response_type=DeliberationResponseType.EVIDENCE_RESPONSE,
            content="A reference is available for independent review.",
            evidence_references=["external-report-ref"],
            useful_new_information=True,
            reply_to_message_id=fields["Reply to message ID"],
        )

    coordinator, _, _, _, _ = make_incremental_coordinator(client, response)
    report = await coordinator.run()

    assert report.exchanges[0].evidence_references == ["external-report-ref"]
    assert report.exchanges[0].evidence_references_verified is False
    assert any("remain unverified" in diagnostic for diagnostic in report.diagnostics)
    assert len(report.exchanges) == 1


@pytest.mark.asyncio
async def test_unrelated_response_claim_does_not_trigger_followup():
    client = MagicMock()

    def response(fields, kwargs):
        return DeliberationResponsePayload(
            execution_id=fields["Execution ID"],
            exchange_id=fields["Exchange ID"],
            correlation_id=fields["Correlation ID"],
            sender_agent_id="agent_target",
            recipient_agent_id="agent_source",
            response_type=DeliberationResponseType.CLARIFICATION,
            content="Unrelated note.",
            new_claims=["Quantum processors require cryogenic environments."],
            useful_new_information=True,
            reply_to_message_id=fields["Reply to message ID"],
        )

    coordinator, _, _, _, _ = make_incremental_coordinator(client, response)
    report = await coordinator.run()

    assert len(report.exchanges) == 1
    assert not [trigger for trigger in report.triggers if trigger.originating_response_message_id]


@pytest.mark.asyncio
async def test_evidence_reference_resolves_information_request_without_repeated_exchange():
    client = MagicMock()

    def response(fields, kwargs):
        return DeliberationResponsePayload(
            execution_id=fields["Execution ID"],
            exchange_id=fields["Exchange ID"],
            correlation_id=fields["Correlation ID"],
            sender_agent_id="agent_target",
            recipient_agent_id="agent_source",
            response_type=DeliberationResponseType.EVIDENCE_RESPONSE,
            content="This reference may be relevant; it remains unverified.",
            evidence_references=["benchmark-report-id"],
            useful_new_information=True,
            reply_to_message_id=fields["Reply to message ID"],
        )

    coordinator, _, _, _, _ = make_incremental_coordinator(client, response)
    coordinator.task_results[0].unresolved_questions = []
    coordinator.task_results[0].claims = ["We measured database read latency of 50ms."]
    coordinator.task_results[0].output = "Database read latency measurement evidence review."
    coordinator.task_results[1].output = "Database read latency measurement evidence context."
    report = await coordinator.run()

    assert len(report.exchanges) == 1
    assert report.triggers[0].trigger_type == DeliberationTriggerType.UNSUPPORTED_EMPIRICAL_CLAIM
    assert report.exchanges[0].evidence_references_verified is False
    assert any(
        suppression.reason.startswith("Duplicate trigger")
        for suppression in report.suppressed_triggers
    )


@pytest.mark.asyncio
async def test_incremental_policy_suppresses_previously_processed_trigger_and_bounds_cycle():
    client = MagicMock()
    response_count = 0

    async def respond(fields, kwargs):
        nonlocal response_count
        response_count += 1
        sender = "agent_target" if kwargs["model"] == "target-model" else "agent_source"
        recipient = "agent_source" if sender == "agent_target" else "agent_target"
        return DeliberationResponsePayload(
            execution_id=fields["Execution ID"],
            exchange_id=fields["Exchange ID"],
            correlation_id=fields["Correlation ID"],
            sender_agent_id=sender,
            recipient_agent_id=recipient,
            response_type=DeliberationResponseType.CLARIFICATION,
            content="Question remains open.",
            clarifications=["Database read latency target remains disputed."],
            unresolved_questions=["Which database read latency target should be used?"],
            useful_new_information=True,
            reply_to_message_id=fields["Reply to message ID"],
        )

    client.generate_structured = AsyncMock(side_effect=respond)
    coordinator, _, _, _, _ = make_incremental_coordinator(
        client,
        respond,
        max_rounds=2,
    )

    report = await coordinator.run()

    assert len(report.rounds) <= 2
    assert client.generate_structured.await_count <= 2
    assert report.termination_reason in {
        DeliberationTerminationReason.MAX_ROUNDS_REACHED,
        DeliberationTerminationReason.UNRESOLVED_QUESTIONS,
        DeliberationTerminationReason.RESOLVED,
    }
    assert report.suppressed_triggers
