"""OpenAI LLM client."""

from openai import OpenAI

from src.llm.base import BaseLLMClient
from src.utils.helpers import get_env, load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class OpenAIClient(BaseLLMClient):
    """LLM client using the OpenAI API.

    Args:
        model: Model name (e.g., 'gpt-4o-mini'). Defaults to config value.
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        config = load_config().get("llm", {})
        self.model = model or config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 0.2)
        self.max_tokens = config.get("max_tokens", 1024)

        resolved_key = api_key or get_env("OPENAI_API_KEY")
        if not resolved_key or resolved_key.startswith("your_"):
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )

        self._client = OpenAI(api_key=resolved_key)
        logger.info("OpenAI client initialized (model=%s)", self.model)

    def generate_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response using OpenAI's chat completion API.

        Args:
            prompt: The user prompt with context.
            system_prompt: Optional system instructions.

        Returns:
            The model's text response.
        """
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        logger.info("Sending request to OpenAI (model=%s)", self.model)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = response.choices[0].message.content or ""
        logger.info("OpenAI response received (%d chars)", len(answer))
        return answer

    @property
    def provider_name(self) -> str:
        return "openai"
