import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest

from communication.message import Message
from communication.message_bus import MessageBus
from communication.shared_context import SharedContext
from core.agent_factory import AgentFactory
from core.execution_engine import ExecutionEngine
from core.llm_client import GeminiClient
from core.task_manager import TaskManager
from models.agent_models import AgentConfig, AgentResponse
from models.execution_policy import ExecutionPolicy
from models.schemas import AgentStatus, MessageType, TaskStatus
from models.task_models import SubTask, TaskPlan, TaskResult


@pytest.mark.asyncio
async def test_message_bus_routing_and_isolation():
    """Verify direct and broadcast messaging across multiple agent inboxes."""
    bus = MessageBus()
    bus.register_agent("agent_1")
    bus.register_agent("agent_2")
    bus.register_agent("agent_3")

    # Direct message from agent_1 to agent_2
    msg_direct = Message(
        sender_id="agent_1",
        recipient_id="agent_2",
        message_type=MessageType.QUESTION,
        content="What is the expected QPS?",
    )
    await bus.publish(msg_direct)

    # Broadcast message from agent_3
    msg_broadcast = Message(
        sender_id="agent_3",
        recipient_id="ALL",
        message_type=MessageType.FINDING,
        content="Found cache invalidation bottleneck",
    )
    await bus.publish(msg_broadcast)

    # Check agent_2 inbox (should receive direct message and broadcast message)
    a2_msgs = await bus.drain_inbox("agent_2")
    assert len(a2_msgs) == 2
    assert a2_msgs[0].content == "What is the expected QPS?"
    assert a2_msgs[1].content == "Found cache invalidation bottleneck"

    # Check agent_1 inbox (should only receive broadcast, not direct message meant for agent_2)
    a1_msgs = await bus.drain_inbox("agent_1")
    assert len(a1_msgs) == 1
    assert a1_msgs[0].content == "Found cache invalidation bottleneck"

    # Check agent_3 inbox (should not receive its own broadcast)
    a3_msgs = await bus.drain_inbox("agent_3")
    assert len(a3_msgs) == 0


@pytest.mark.asyncio
async def test_message_protocol_metadata_reply_chain_and_evidence_refs():
    bus = MessageBus(max_response_depth=1)
    bus.register_agent("agent_1", execution_id="exec_1", subtask_id="T1")
    bus.register_agent("agent_2", execution_id="exec_1", subtask_id="T2")

    finding = Message(
        sender_id="agent_1",
        recipient_id="agent_2",
        task_id="exec_1",
        execution_id="exec_1",
        message_type=MessageType.FINDING,
        content="Finding",
        correlation_id="corr-1",
        claim_references=["claim-1"],
        evidence_references=["evidence-1"],
    )
    assert await bus.publish(finding) is True
    received = await bus.get_message("agent_2")
    assert received.correlation_id == "corr-1"
    assert received.claim_references == ["claim-1"]
    assert received.evidence_references == ["evidence-1"]

    reply = Message(
        sender_id="agent_2",
        recipient_id="agent_1",
        task_id="exec_1",
        execution_id="exec_1",
        message_type=MessageType.CLARIFICATION,
        content="Clarification",
        reply_to_message_id=finding.message_id,
    )
    assert await bus.publish(reply) is True
    assert (await bus.get_message("agent_1")).reply_to_message_id == finding.message_id


@pytest.mark.asyncio
async def test_message_bus_rejects_invalid_identity_and_execution_crossing():
    bus = MessageBus()
    bus.register_agent("agent_1", execution_id="exec_1")
    bus.register_agent("agent_2", execution_id="exec_2")

    with pytest.raises(ValueError, match="Sender agent"):
        await bus.publish(Message(sender_id="spoofed", recipient_id="agent_1", message_type=MessageType.FINDING, content="x"))

    with pytest.raises(ValueError, match="another execution"):
        await bus.publish(
            Message(
                sender_id="agent_1",
                recipient_id="agent_2",
                execution_id="exec_1",
                message_type=MessageType.FINDING,
                content="cross execution",
            )
        )

    with pytest.raises(ValueError, match="Recipient agent"):
        await bus.publish(Message(sender_id="agent_1", recipient_id="missing", message_type=MessageType.FINDING, content="x"))


@pytest.mark.asyncio
async def test_message_bus_deduplicates_and_isolates_executions():
    bus = MessageBus()
    bus.register_agent("a1", execution_id="exec_1")
    bus.register_agent("a2", execution_id="exec_1")
    bus.register_agent("b1", execution_id="exec_2")
    bus.register_agent("b2", execution_id="exec_2")

    message = Message(
        message_id="duplicate-message",
        sender_id="a1",
        recipient_id="ALL",
        execution_id="exec_1",
        message_type=MessageType.FINDING,
        content="isolated",
    )
    assert await bus.publish(message) is True
    assert await bus.publish(message) is False
    assert len(await bus.drain_inbox("a2")) == 1
    assert len(await bus.drain_inbox("b2")) == 0
    assert len(bus.get_history(task_id="exec_1")) == 1


@pytest.mark.asyncio
async def test_message_limits_and_concurrent_publication():
    bus = MessageBus(max_messages_per_execution=2, max_messages_per_agent=2)
    bus.register_agent("a1", execution_id="exec")
    bus.register_agent("a2", execution_id="exec")

    async def publish(index):
        return await bus.publish(
            Message(
                message_id=f"msg-{index}",
                sender_id="a1",
                recipient_id="a2",
                execution_id="exec",
                message_type=MessageType.FINDING,
                content=str(index),
            )
        )

    assert await asyncio.gather(publish(1), publish(2)) == [True, True]
    with pytest.raises(RuntimeError, match="Message limit"):
        await publish(3)


@pytest.mark.asyncio
async def test_shared_context_thread_safety():
    """Verify concurrent updates to SharedContext."""
    ctx = SharedContext(task_id="t1", task_description="Concurrent test")

    async def add_data(idx):
        await ctx.add_finding(f"agent_{idx}", f"Role_{idx}", f"Finding {idx}")
        await ctx.set_note(f"key_{idx}", idx)

    import asyncio
    await asyncio.gather(*[add_data(i) for i in range(10)])

    findings = await ctx.get_findings()
    notes = await ctx.get_all_notes()
    assert len(findings) == 10
    assert len(notes) == 10
    summary = await ctx.format_summary()
    assert "Finding 0" in summary


def test_task_manager_dependency_resolution():
    """Verify TaskManager identifies ready subtasks and blocks on failed dependencies."""
    plan = TaskPlan(
        task="Test Dependencies",
        subtasks=[
            SubTask(id="T1", role="Role 1", objective="Obj 1", dependencies=[]),
            SubTask(id="T2", role="Role 2", objective="Obj 2", dependencies=[]),
            SubTask(id="T3", role="Role 3", objective="Obj 3", dependencies=["T1", "T2"]),
            SubTask(id="T4", role="Role 4", objective="Obj 4", dependencies=["T3"]),
        ],
    )
    tm = TaskManager(plan=plan, task_id="task_test")

    # Initial ready tasks should be T1 and T2
    ready = tm.get_ready_subtasks()
    ready_ids = {s.id for s in ready}
    assert ready_ids == {"T1", "T2"}

    # Complete T1
    tm.record_result(
        TaskResult(subtask_id="T1", agent_id="a1", role="Role 1", status=TaskStatus.COMPLETED)
    )

    # T3 is not yet ready because T2 is still pending
    ready = tm.get_ready_subtasks()
    assert {s.id for s in ready} == {"T2"}

    # Complete T2
    tm.record_result(
        TaskResult(subtask_id="T2", agent_id="a2", role="Role 2", status=TaskStatus.COMPLETED)
    )

    # Now T3 is ready
    ready = tm.get_ready_subtasks()
    assert {s.id for s in ready} == {"T3"}

    # Fail T3
    tm.record_result(
        TaskResult(subtask_id="T3", agent_id="a3", role="Role 3", status=TaskStatus.FAILED, error="OOM")
    )

    # T4 depends on failed T3, so it should be marked BLOCKED and not be ready
    ready = tm.get_ready_subtasks()
    assert len(ready) == 0
    assert tm.subtasks["T4"].status == TaskStatus.BLOCKED
    assert tm.is_all_finished() is True
    assert tm.results["T4"].status == TaskStatus.BLOCKED
    assert "T3" in tm.results["T4"].error


@pytest.mark.asyncio
async def test_execution_engine_parallel_flow():
    """Verify ExecutionEngine executes subtasks and respects dependencies."""
    mock_client = MagicMock()
    gemini_client = GeminiClient(api_key="mock_key", client=mock_client)
    factory = AgentFactory(llm_client=gemini_client)
    bus = MessageBus()
    shared_ctx = SharedContext(task_id="exec_1", task_description="Parallel test")

    plan = TaskPlan(
        task="Test Parallel Plan",
        subtasks=[
            SubTask(id="T1", role="Systems Architect", objective="Design core", dependencies=[]),
            SubTask(id="T2", role="Security Auditor", objective="Audit security", dependencies=[]),
            SubTask(id="T3", role="Integration Specialist", objective="Integrate", dependencies=["T1", "T2"]),
        ],
    )

    # Mock agent execution
    async def mock_execute(context=None):
        return AgentResponse(
            agent_id="test",
            role="Specialist",
            task_id="exec_1",
            subtask_id="sub",
            content="Deliverable content",
            findings=["Key finding"],
            claims=["Concrete claim"],
            status=AgentStatus.COMPLETED,
        )

    # Patch factory to return mock-executing agents
    orig_create = factory.create_from_subtask

    def patched_create(subtask, task_id=""):
        agent = orig_create(subtask, task_id)
        agent.execute = AsyncMock(side_effect=mock_execute)
        return agent

    factory.create_from_subtask = patched_create

    engine = ExecutionEngine(max_concurrency=2, timeout_seconds=30.0)
    results = await engine.execute_plan(
        plan=plan,
        agent_factory=factory,
        message_bus=bus,
        shared_context=shared_ctx,
        task_id="exec_1",
    )

    assert len(results) == 3
    assert all(r.status == TaskStatus.COMPLETED for r in results)
    history = bus.get_history()
    # Ensure findings were published to message bus
    assert len(history) >= 3


@pytest.mark.asyncio
async def test_dependency_agent_consumes_prior_finding_from_message_bus():
    factory = MagicMock()
    received_contexts = {}

    def create_agent(subtask, task_id=""):
        agent = MagicMock()
        agent.agent_id = f"agent_{subtask.id}"
        agent.role = subtask.role

        async def execute(context=None):
            received_contexts[subtask.id] = context or {}
            return AgentResponse(
                agent_id=agent.agent_id,
                role=subtask.role,
                task_id=task_id,
                subtask_id=subtask.id,
                content=f"Result {subtask.id}",
                findings=[f"Finding from {subtask.id}"],
                status=AgentStatus.COMPLETED,
            )

        agent.execute = execute
        return agent

    factory.create_from_subtask.side_effect = create_agent
    plan = TaskPlan(
        task="Dependency communication",
        subtasks=[
            SubTask(id="T1", role="Architecture", objective="Produce finding"),
            SubTask(id="T2", role="Integration", objective="Consume finding", dependencies=["T1"]),
        ],
    )
    bus = MessageBus()
    context = SharedContext(task_id="exec_comm", task_description="Communication test")

    results = await ExecutionEngine(max_concurrency=2).execute_plan(
        plan,
        factory,
        bus,
        context,
        task_id="exec_comm",
        execution_policy=ExecutionPolicy(
            honor_dependencies=True,
            inject_upstream_results=True,
            route_messages=True,
        ),
    )

    assert all(result.status == TaskStatus.COMPLETED for result in results)
    assert "Finding from T1" in received_contexts["T2"]["shared_notes"]
    assert len(bus.get_history(task_id="exec_comm")) >= 2
    snapshot = await context.snapshot()
    assert snapshot["communications"]


@pytest.mark.asyncio
async def test_independent_parallel_policy_ignores_dependencies_and_message_channels():
    second_started = asyncio.Event()
    contexts = {}
    executed_ids = []

    class RecordingBus(MessageBus):
        def __init__(self):
            super().__init__()
            self.registered = []
            self.drained = []
            self.published = []

        def register_agent(self, *args, **kwargs):
            self.registered.append((args, kwargs))
            super().register_agent(*args, **kwargs)

        async def drain_inbox(self, agent_id):
            self.drained.append(agent_id)
            return await super().drain_inbox(agent_id)

        async def publish(self, message):
            self.published.append(message)
            return await super().publish(message)

    factory = MagicMock()

    def create_agent(subtask, task_id=""):
        agent = MagicMock()
        agent.agent_id = f"agent_{subtask.id}"
        agent.role = subtask.role

        async def execute(context=None):
            contexts[subtask.id] = context or {}
            executed_ids.append(subtask.id)
            if subtask.id == "T2":
                second_started.set()
            elif subtask.id == "T1":
                await asyncio.wait_for(second_started.wait(), timeout=0.5)
            return AgentResponse(
                agent_id=agent.agent_id,
                role=subtask.role,
                task_id=task_id,
                subtask_id=subtask.id,
                content=f"Result {subtask.id}",
                findings=[f"Finding {subtask.id}"],
                evidence_references=[f"reference-{subtask.id}"],
                status=AgentStatus.COMPLETED,
            )

        agent.execute = execute
        return agent

    factory.create_from_subtask.side_effect = create_agent
    plan = TaskPlan(
        task="Independent execution",
        subtasks=[
            SubTask(id="T1", role="Producer", objective="Wait for independent sibling"),
            SubTask(id="T2", role="Dependent", objective="Would normally require T1", dependencies=["T1"]),
        ],
    )
    bus = RecordingBus()
    context = SharedContext(task_id="exec_independent", task_description="Independent execution")

    results = await ExecutionEngine(max_concurrency=2, timeout_seconds=2).execute_plan(
        plan,
        factory,
        bus,
        context,
        task_id="exec_independent",
        execution_policy=ExecutionPolicy(
            honor_dependencies=False,
            inject_upstream_results=False,
            route_messages=False,
        ),
    )

    assert set(executed_ids) == {"T1", "T2"}
    assert all(result.status == TaskStatus.COMPLETED for result in results)
    assert contexts["T2"]["upstream_results"] == []
    assert contexts["T2"]["shared_notes"] == ""
    assert bus.registered == []
    assert bus.drained == []
    assert bus.published == []
    snapshot = await context.snapshot()
    assert snapshot["communications"] == []
    assert snapshot["findings"] == []


@pytest.mark.asyncio
async def test_failed_specialist_does_not_stop_independent_tasks():
    factory = MagicMock()
    executed_ids = []

    def create_agent(subtask, task_id=""):
        agent = MagicMock()
        agent.agent_id = f"agent_{subtask.id}"
        agent.role = subtask.role

        async def execute(context=None):
            executed_ids.append(subtask.id)
            if subtask.id == "T1":
                raise RuntimeError("specialist exploded")
            return AgentResponse(
                agent_id=agent.agent_id,
                role=subtask.role,
                task_id=task_id,
                subtask_id=subtask.id,
                content=f"Result for {subtask.id}",
                findings=[f"Finding from {subtask.id}"],
                status=AgentStatus.COMPLETED,
            )

        agent.execute = AsyncMock(side_effect=execute)
        return agent

    factory.create_from_subtask.side_effect = create_agent
    plan = TaskPlan(
        task="Run independent specialist work",
        subtasks=[
            SubTask(id="T1", role="Architecture", objective="May fail"),
            SubTask(id="T2", role="Privacy", objective="Independent privacy review"),
            SubTask(id="T3", role="Integration", objective="Use T1 output", dependencies=["T1"]),
            SubTask(id="T4", role="Follow-up", objective="Use T3 output", dependencies=["T3"]),
        ],
    )
    results = await ExecutionEngine(max_concurrency=3, timeout_seconds=10).execute_plan(
        plan,
        factory,
        MessageBus(),
        SharedContext(task_id="exec_test", task_description="Independent tasks"),
        task_id="exec_test",
    )

    outcomes = {result.subtask_id: result for result in results}
    assert set(executed_ids) == {"T1", "T2"}
    assert outcomes["T1"].status == TaskStatus.FAILED
    assert outcomes["T2"].status == TaskStatus.COMPLETED
    assert outcomes["T3"].status == TaskStatus.BLOCKED
    assert outcomes["T4"].status == TaskStatus.BLOCKED
    assert "Specialist execution failure [subtask T1]" in outcomes["T1"].error


@pytest.mark.asyncio
async def test_execution_engine_finishes_independent_work_when_cycle_exists():
    factory = MagicMock()

    def create_agent(subtask, task_id=""):
        agent = MagicMock()
        agent.agent_id = f"agent_{subtask.id}"
        agent.role = subtask.role
        agent.execute = AsyncMock(
            return_value=AgentResponse(
                agent_id=agent.agent_id,
                role=subtask.role,
                task_id=task_id,
                subtask_id=subtask.id,
                content="Independent deliverable",
                status=AgentStatus.COMPLETED,
            )
        )
        return agent

    factory.create_from_subtask.side_effect = create_agent
    plan = TaskPlan(
        task="Cycle plus runnable work",
        subtasks=[
            SubTask(id="T1", role="Cycle A", objective="Wait for T2", dependencies=["T2"]),
            SubTask(id="T2", role="Cycle B", objective="Wait for T1", dependencies=["T1"]),
            SubTask(id="T3", role="Independent", objective="Run independently"),
        ],
    )

    results = await ExecutionEngine(max_concurrency=2, timeout_seconds=10).execute_plan(
        plan,
        factory,
        MessageBus(),
        SharedContext(task_id="exec_cycle", task_description="Cycle test"),
        task_id="exec_cycle",
    )

    outcomes = {result.subtask_id: result for result in results}
    assert outcomes["T3"].status == TaskStatus.COMPLETED
    assert outcomes["T1"].status == TaskStatus.BLOCKED
    assert outcomes["T2"].status == TaskStatus.BLOCKED
    assert all("dependency cycle" in outcomes[subtask_id].error for subtask_id in ("T1", "T2"))


@pytest.mark.asyncio
async def test_execution_deadline_cancels_hung_worker_preserves_completed_and_blocks_dependents():
    started = asyncio.Event()
    cancelled = asyncio.Event()
    executed_ids = []
    agents = []

    class AgentStub:
        def __init__(self, subtask, task_id):
            self.agent_id = f"agent_{subtask.id}"
            self.role = subtask.role
            self.assigned_subtask_id = subtask.id
            self.status = AgentStatus.EXECUTING
            self.task_id = task_id

        def update_status(self, status):
            self.status = status

        async def execute(self, context=None):
            executed_ids.append(self.assigned_subtask_id)
            if self.assigned_subtask_id == "T1":
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    return AgentResponse(
                        agent_id=self.agent_id,
                        role=self.role,
                        task_id=self.task_id,
                        subtask_id=self.assigned_subtask_id,
                        content="Late result must be discarded.",
                        status=AgentStatus.COMPLETED,
                    )
            return AgentResponse(
                agent_id=self.agent_id,
                role=self.role,
                task_id=self.task_id,
                subtask_id=self.assigned_subtask_id,
                content=f"Completed {self.assigned_subtask_id} before the deadline.",
                findings=[f"Finding from {self.assigned_subtask_id}"],
                status=AgentStatus.COMPLETED,
            )

    factory = MagicMock()

    def create_agent(subtask, task_id=""):
        agent = AgentStub(subtask, task_id)
        agents.append(agent)
        return agent

    factory.create_from_subtask.side_effect = create_agent
    factory.list_agents.side_effect = lambda: list(agents)
    plan = TaskPlan(
        task="Execution deadline",
        subtasks=[
            SubTask(id="T1", role="Slow producer", objective="Hang until cancelled"),
            SubTask(id="T2", role="Independent", objective="Complete before deadline"),
            SubTask(id="T3", role="Dependent", objective="Requires T1", dependencies=["T1"]),
        ],
    )
    engine = ExecutionEngine(max_concurrency=2, timeout_seconds=0.05)

    results = await engine.execute_plan(
        plan,
        factory,
        MessageBus(),
        SharedContext(task_id="exec_deadline", task_description="Deadline test"),
        task_id="exec_deadline",
    )

    outcomes = {result.subtask_id: result for result in results}
    assert started.is_set()
    assert cancelled.is_set()
    assert outcomes["T1"].status == TaskStatus.TIMED_OUT
    assert "timed out" in outcomes["T1"].error.lower()
    assert "Late result" not in outcomes["T1"].output
    assert outcomes["T2"].status == TaskStatus.COMPLETED
    assert "Completed T2" in outcomes["T2"].output
    assert outcomes["T3"].status == TaskStatus.BLOCKED
    assert "T3" not in executed_ids
    assert next(agent for agent in agents if agent.assigned_subtask_id == "T1").status == AgentStatus.TIMED_OUT
    assert not [
        task for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and task.get_name().startswith("medha-subtask-")
    ]


@pytest.mark.asyncio
async def test_parent_cancellation_cancels_workers_and_propagates_without_orphans():
    started = asyncio.Event()
    cancelled = asyncio.Event()
    agents = []

    class AgentStub:
        def __init__(self, subtask, task_id):
            self.agent_id = f"agent_{subtask.id}"
            self.role = subtask.role
            self.assigned_subtask_id = subtask.id
            self.status = AgentStatus.EXECUTING
            self.task_id = task_id

        def update_status(self, status):
            self.status = status

        async def execute(self, context=None):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    factory = MagicMock()

    def create_agent(subtask, task_id=""):
        agent = AgentStub(subtask, task_id)
        agents.append(agent)
        return agent

    factory.create_from_subtask.side_effect = create_agent
    factory.list_agents.side_effect = lambda: list(agents)
    plan = TaskPlan(
        task="Parent cancellation",
        subtasks=[SubTask(id="T1", role="Slow", objective="Wait")],
    )
    engine = ExecutionEngine(timeout_seconds=10)
    execution = asyncio.create_task(
        engine.execute_plan(
            plan,
            factory,
            MessageBus(),
            SharedContext(task_id="exec_cancel", task_description="Cancellation test"),
            task_id="exec_cancel",
        )
    )
    await started.wait()
    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await execution

    assert cancelled.is_set()
    assert agents[0].status == AgentStatus.CANCELLED
    assert not [
        task for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and task.get_name().startswith("medha-subtask-")
    ]


@pytest.mark.parametrize("timeout_seconds", [0, -1, float("nan"), float("inf")])
def test_execution_engine_rejects_non_positive_or_non_finite_timeout(timeout_seconds):
    with pytest.raises(ValueError, match="finite positive"):
        ExecutionEngine(timeout_seconds=timeout_seconds)

