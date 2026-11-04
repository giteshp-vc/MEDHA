from typing import Any, Optional

from config.settings import LLM_PROVIDER, validate_settings
from core.providers.base import BaseLLMClient
from core.providers.gemini_client import GeminiClient
from core.providers.groq_client import GroqClient
from utils.token_tracker import TokenTracker


def create_llm_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    default_model: Optional[str] = None,
    token_tracker: Optional[TokenTracker] = None,
    client: Optional[Any] = None,
) -> BaseLLMClient:
    """Factory creating the appropriate LLM client based on the requested or configured provider."""
    target_provider = (provider or LLM_PROVIDER or "groq").lower().strip()

    if target_provider == "groq":
        return GroqClient(
            api_key=api_key,
            default_model=default_model,
            token_tracker=token_tracker,
            client=client,
        )
    elif target_provider == "gemini":
        return GeminiClient(
            api_key=api_key,
            default_model=default_model,
            token_tracker=token_tracker,
            client=client,
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{target_provider}'. Supported providers are 'groq' and 'gemini'."
        )


class LLMClient:
    """Global manager and singleton provider router for MEDHA."""

    _instance: Optional[BaseLLMClient] = None

    @classmethod
    def get_instance(cls, provider: Optional[str] = None) -> BaseLLMClient:
        """Get or initialize the singleton LLM client for the active provider."""
        if cls._instance is None:
            validate_settings()
            cls._instance = create_llm_client(provider=provider)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance."""
        cls._instance = None
        GroqClient.reset_instance()
        GeminiClient.reset_instance()


def get_llm_client(provider: Optional[str] = None) -> BaseLLMClient:
    """Convenience accessor for the global LLM client."""
    return LLMClient.get_instance(provider=provider)


__all__ = [
    "BaseLLMClient",
    "GroqClient",
    "GeminiClient",
    "LLMClient",
    "create_llm_client",
    "get_llm_client",
]
