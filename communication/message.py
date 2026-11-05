from typing import Any, Dict, List, Optional
from pydantic import Field

from models.schemas import BaseSchema, MessageType
from utils.helpers import generate_id, utcnow_iso


class Message(BaseSchema):
    """Structured message exchanged between agents on the message bus."""

    message_id: str = Field(default_factory=lambda: generate_id("msg"))
    sender_id: str
    recipient_id: str = Field(default="ALL", description="Target agent ID or 'ALL' for broadcast")
    task_id: str = ""
    execution_id: str = ""
    sender_agent_id: Optional[str] = None
    recipient_agent_id: Optional[str] = None
    subtask_id: Optional[str] = None
    message_type: MessageType
    content: str
    correlation_id: str = Field(default_factory=lambda: generate_id("corr"))
    reply_to_message_id: Optional[str] = None
    claim_references: List[str] = Field(default_factory=list)
    evidence_references: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=utcnow_iso)

    @property
    def effective_execution_id(self) -> str:
        return self.execution_id or self.task_id

    @property
    def effective_sender_id(self) -> str:
        return self.sender_agent_id or self.sender_id

    @property
    def effective_recipient_id(self) -> str:
        return self.recipient_agent_id or self.recipient_id

