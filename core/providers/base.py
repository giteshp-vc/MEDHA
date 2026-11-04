from abc import ABC, abstractmethod
from typing import Optional, Type, TypeVar
from pydantic import BaseModel

from config.settings import MAX_RETRIES
from utils.token_tracker import TokenTracker

T = TypeVar("T", bound=BaseModel)


class BaseLLMClient(ABC):
    """Abstract base class for all MEDHA LLM provider implementations."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        token_tracker: Optional[TokenTracker] = None,
    ):
        self.api_key = api_key
        self.default_model = default_model
        self.token_tracker = token_tracker or TokenTracker(name="llm_client")

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: int = MAX_RETRIES,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate unstructured text from the LLM provider."""
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: int = MAX_RETRIES,
        max_tokens: Optional[int] = None,
    ) -> T:
        """Generate structured data adhering to a Pydantic schema."""
        pass

    def get_token_tracker(self) -> TokenTracker:
        """Return the token consumption tracker associated with this client."""
        return self.token_tracker

