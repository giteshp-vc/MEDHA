import asyncio
from collections import defaultdict
from typing import Dict, List, Optional, Set

from communication.message import Message
from models.schemas import MessageType
from utils.helpers import generate_id
from utils.logger import get_logger

logger = get_logger("message_bus")


class MessageBus:
    """Asynchronous message broker for inter-agent communication and challenge exchange."""

    def __init__(
        self,
        max_messages_per_execution: int = 200,
        max_messages_per_agent: int = 50,
        max_response_depth: int = 3,
    ):
        if min(max_messages_per_execution, max_messages_per_agent) < 1:
            raise ValueError("Message limits must be positive.")
        if max_response_depth < 0:
            raise ValueError("max_response_depth cannot be negative.")
        # Per-agent async inboxes
        self._inboxes: Dict[str, asyncio.Queue[Message]] = {}
        self._agent_executions: Dict[str, str] = {}
        self._agent_subtasks: Dict[str, str] = {}
        # Global audit log of all exchanged messages
        self._history: List[Message] = []
        self._message_ids: Set[str] = set()
        self._execution_counts: Dict[str, int] = defaultdict(int)
        self._agent_counts: Dict[str, int] = defaultdict(int)
        self.max_messages_per_execution = max_messages_per_execution
        self.max_messages_per_agent = max_messages_per_agent
        self.max_response_depth = max_response_depth
        self._lock = asyncio.Lock()

    def register_agent(
        self,
        agent_id: str,
        execution_id: str = "",
        subtask_id: str = "",
    ) -> None:
        """Register an agent and create its private async message inbox."""
        if not agent_id:
            raise ValueError("agent_id cannot be empty.")
        if agent_id not in self._inboxes:
            self._inboxes[agent_id] = asyncio.Queue()
            logger.debug(f"Registered agent [{agent_id}] with message bus.")
            for message in self._history:
                if message.effective_recipient_id != "ALL":
                    continue
                if execution_id and message.effective_execution_id != execution_id:
                    continue
                target_subtasks = message.metadata.get("target_subtask_ids")
                if target_subtasks and subtask_id not in target_subtasks:
                    continue
                if message.effective_sender_id != agent_id:
                    self._inboxes[agent_id].put_nowait(message)
        if execution_id:
            self._agent_executions[agent_id] = execution_id
        if subtask_id:
            self._agent_subtasks[agent_id] = subtask_id

    def unregister_agent(self, agent_id: str) -> None:
        """Remove agent inbox from message bus."""
        self._inboxes.pop(agent_id, None)
        self._agent_executions.pop(agent_id, None)
        self._agent_subtasks.pop(agent_id, None)

    def _validate_message(self, message: Message) -> None:
        if message.sender_agent_id and message.sender_id != message.sender_agent_id:
            raise ValueError("sender_id and sender_agent_id must identify the same agent.")
        if message.recipient_agent_id and message.recipient_id != message.recipient_agent_id:
            raise ValueError("recipient_id and recipient_agent_id must identify the same agent.")
        sender_id = message.effective_sender_id
        recipient_id = message.effective_recipient_id
        if sender_id not in self._inboxes:
            raise ValueError(f"Sender agent [{sender_id}] is not registered.")
        if recipient_id != "ALL" and recipient_id not in self._inboxes:
            raise ValueError(f"Recipient agent [{recipient_id}] is not registered.")
        execution_id = message.effective_execution_id
        if execution_id and self._agent_executions.get(sender_id, execution_id) != execution_id:
            raise ValueError(f"Sender agent [{sender_id}] is not registered for execution [{execution_id}].")
        if recipient_id != "ALL" and execution_id:
            recipient_execution = self._agent_executions.get(recipient_id, execution_id)
            if recipient_execution != execution_id:
                raise ValueError(f"Recipient agent [{recipient_id}] belongs to another execution.")
        if message.reply_to_message_id:
            parent = next(
                (item for item in reversed(self._history) if item.message_id == message.reply_to_message_id),
                None,
            )
            if parent is None:
                raise ValueError(f"Reply target [{message.reply_to_message_id}] was not found.")
            depth = int(parent.metadata.get("response_depth", 0)) + 1
            if depth > self.max_response_depth:
                raise RuntimeError("Maximum inter-agent response depth reached.")
            message.metadata["response_depth"] = depth

    async def publish(self, message: Message) -> bool:
        """Publish a message to a specific agent inbox or broadcast to all registered agents."""
        async with self._lock:
            self._validate_message(message)
            if message.message_id in self._message_ids:
                logger.debug("Ignoring duplicate message [%s].", message.message_id)
                return False
            execution_id = message.effective_execution_id
            if execution_id and self._execution_counts[execution_id] >= self.max_messages_per_execution:
                raise RuntimeError(f"Message limit reached for execution [{execution_id}].")
            sender_id = message.effective_sender_id
            if self._agent_counts[sender_id] >= self.max_messages_per_agent:
                raise RuntimeError(f"Message limit reached for agent [{sender_id}].")

            self._message_ids.add(message.message_id)
            self._history.append(message)
            self._execution_counts[execution_id] += 1
            self._agent_counts[sender_id] += 1

            recipient_id = message.effective_recipient_id
            if recipient_id == "ALL":
                # Broadcast to every agent except the sender
                for agent_id, queue in self._inboxes.items():
                    target_subtasks = message.metadata.get("target_subtask_ids")
                    if agent_id != sender_id and (
                        not execution_id or self._agent_executions.get(agent_id, execution_id) == execution_id
                    ) and (not target_subtasks or self._agent_subtasks.get(agent_id) in target_subtasks):
                        await queue.put(message)
            else:
                # Direct message to specific agent inbox
                await self._inboxes[recipient_id].put(message)

        logger.debug(
            f"Message [{message.message_type}] from [{message.sender_id}] "
            f"-> [{message.recipient_id}]: {message.content[:50]}..."
        )
        return True

    async def get_message(self, agent_id: str, timeout: Optional[float] = 0.1) -> Optional[Message]:
        """Fetch the next available message for an agent with optional timeout."""
        if agent_id not in self._inboxes:
            return None
        queue = self._inboxes[agent_id]
        try:
            if timeout is not None:
                return await asyncio.wait_for(queue.get(), timeout=timeout)
            return await queue.get()
        except (asyncio.TimeoutError, TimeoutError):
            return None

    async def drain_inbox(self, agent_id: str) -> List[Message]:
        """Retrieve and drain all pending messages for an agent."""
        if agent_id not in self._inboxes:
            return []
        queue = self._inboxes[agent_id]
        messages = []
        while not queue.empty():
            try:
                messages.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    def get_history(
        self,
        task_id: Optional[str] = None,
        message_type: Optional[MessageType] = None,
    ) -> List[Message]:
        """Retrieve filtered message history from the audit log."""
        results = self._history
        if task_id:
            results = [m for m in results if m.effective_execution_id == task_id]
        if message_type:
            results = [m for m in results if m.message_type == message_type]
        return list(results)

    async def challenge_finding(
        self,
        sender_id: str,
        target_agent_id: str,
        finding_or_claim: str,
        critique: str,
        task_id: str = "",
    ) -> Message:
        """Convenience method for one agent to formally challenge another agent's assertion."""
        msg = Message(
            sender_id=sender_id,
            recipient_id=target_agent_id,
            task_id=task_id,
            execution_id=task_id,
            message_type=MessageType.CHALLENGE,
            content=f"CHALLENGE: Assertion '{finding_or_claim}' disputed. Reason: {critique}",
            metadata={"target_assertion": finding_or_claim, "critique": critique},
        )
        await self.publish(msg)
        return msg

    async def share_evidence(
        self,
        sender_id: str,
        content: str,
        evidence_references: List[str],
        recipient_id: str = "ALL",
        task_id: str = "",
    ) -> Message:
        """Convenience method to publish evidence to collaborators."""
        msg = Message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            task_id=task_id,
            execution_id=task_id,
            message_type=MessageType.EVIDENCE,
            content=content,
            evidence_references=evidence_references,
        )
        await self.publish(msg)
        return msg

    async def request_evidence(
        self,
        sender_id: str,
        recipient_id: str,
        content: str,
        claim_references: Optional[List[str]] = None,
        task_id: str = "",
        correlation_id: Optional[str] = None,
    ) -> Message:
        msg = Message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            task_id=task_id,
            execution_id=task_id,
            message_type=MessageType.EVIDENCE_REQUEST,
            content=content,
            claim_references=claim_references or [],
            correlation_id=correlation_id or generate_id("corr"),
        )
        await self.publish(msg)
        return msg

    def clear(self) -> None:
        """Reset the message bus."""
        self._inboxes.clear()
        self._history.clear()

