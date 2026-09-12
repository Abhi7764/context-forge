"""Abstract base class for LLM clients.

All LLM providers implement this interface so the RAG pipeline
can work with any provider without code changes.
"""

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Base interface for LLM clients.

    Every LLM provider (OpenAI, Anthropic, Gemini, Ollama) must
    implement this interface.
    """

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user/context prompt to send.
            system_prompt: Optional system-level instructions.

        Returns:
            The LLM's text response.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this LLM provider."""
        ...
