"""Anthropic (Claude) LLM client."""

import anthropic

from src.llm.base import BaseLLMClient
from src.utils.helpers import get_env, load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class AnthropicClient(BaseLLMClient):
    """LLM client using the Anthropic API.

    Args:
        model: Model name (e.g., 'claude-sonnet-4-20250514'). Defaults to config value.
        api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        config = load_config().get("llm", {})
        self.model = model or config.get("model", "claude-sonnet-4-20250514")
        self.temperature = config.get("temperature", 0.2)
        self.max_tokens = config.get("max_tokens", 1024)

        resolved_key = api_key or get_env("ANTHROPIC_API_KEY")
        if not resolved_key or resolved_key.startswith("your_"):
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )

        self._client = anthropic.Anthropic(api_key=resolved_key)
        logger.info("Anthropic client initialized (model=%s)", self.model)

    def generate_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response using Anthropic's messages API.

        Args:
            prompt: The user prompt with context.
            system_prompt: Optional system instructions.

        Returns:
            The model's text response.
        """
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        logger.info("Sending request to Anthropic (model=%s)", self.model)

        response = self._client.messages.create(**kwargs)
        answer = response.content[0].text if response.content else ""
        logger.info("Anthropic response received (%d chars)", len(answer))
        return answer

    @property
    def provider_name(self) -> str:
        return "anthropic"
