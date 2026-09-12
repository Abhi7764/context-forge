"""Ollama LLM client for local models.

Calls the Ollama REST API running on localhost. No API key required.
Install Ollama and pull a model (e.g., `ollama pull llama3`) before use.
"""

import requests

from src.llm.base import BaseLLMClient
from src.utils.helpers import load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class OllamaClient(BaseLLMClient):
    """LLM client using a local Ollama instance.

    Args:
        model: Model name (e.g., 'llama3'). Defaults to config value.
        base_url: Ollama server URL. Defaults to config value.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        config = load_config()
        llm_config = config.get("llm", {})
        ollama_config = config.get("ollama", {})

        self.model = model or llm_config.get("model", "llama3")
        self.base_url = base_url or ollama_config.get(
            "base_url", "http://localhost:11434"
        )
        self.temperature = llm_config.get("temperature", 0.2)

        logger.info(
            "Ollama client initialized (model=%s, url=%s)", self.model, self.base_url
        )

    def generate_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response using the Ollama REST API.

        Args:
            prompt: The user prompt with context.
            system_prompt: Optional system instructions.

        Returns:
            The model's text response.

        Raises:
            ConnectionError: If Ollama is not running.
            RuntimeError: If the API returns an error.
        """
        url = f"{self.base_url}/api/chat"

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        logger.info("Sending request to Ollama (model=%s)", self.model)

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except requests.HTTPError as e:
            raise RuntimeError(f"Ollama API error: {e}")

        data = response.json()
        answer = data.get("message", {}).get("content", "")
        logger.info("Ollama response received (%d chars)", len(answer))
        return answer

    @property
    def provider_name(self) -> str:
        return "ollama"
