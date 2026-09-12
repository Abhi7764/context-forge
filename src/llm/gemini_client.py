"""Google Gemini LLM client.

Uses a singleton pattern to ensure the HTTP connection to Google's
API is created once and reused across all requests. Configures
httpx connection pooling for lower latency on repeated calls.
"""

import httpx
from google import genai

from src.llm.base import BaseLLMClient
from src.utils.helpers import get_env, load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class GeminiClient(BaseLLMClient):
    """LLM client using the Google Gemini API.

    Uses singleton pattern to maintain a persistent HTTP connection
    across requests, avoiding per-request connection overhead.

    Args:
        model: Model name (e.g., 'gemini-2.0-flash'). Defaults to config value.
        api_key: Google API key. Defaults to GOOGLE_API_KEY env var.
    """

    _instance: "GeminiClient | None" = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> "GeminiClient":
        """Singleton — only one client instance and HTTP connection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if GeminiClient._initialized:
            return

        config = load_config().get("llm", {})
        self.model = model or config.get("model", "gemini-2.0-flash")
        self.temperature = config.get("temperature", 0.2)
        self.max_tokens = config.get("max_tokens", 1024)

        resolved_key = api_key or get_env("GOOGLE_API_KEY")
        if not resolved_key or resolved_key.startswith("your_"):
            raise ValueError(
                "GOOGLE_API_KEY is not set. Add it to your .env file."
            )

        # Create a persistent httpx client with connection pooling.
        # This keeps TCP+TLS connections open and reuses them.
        self._httpx_client = httpx.Client(
            http2=True,
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=120,
            ),
        )

        self._client = genai.Client(
            api_key=resolved_key,
            http_options=genai.types.HttpOptions(
                httpx_client=self._httpx_client,
            ),
        )

        # Pre-build the generation config once (reused on every call)
        self._gen_config = genai.types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        GeminiClient._initialized = True
        logger.info("Gemini client initialized (model=%s) with persistent HTTP connection", self.model)

    def generate_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response using Google Gemini API.

        Args:
            prompt: The user prompt with context.
            system_prompt: Optional system instructions.

        Returns:
            The model's text response.
        """
        logger.info("Sending request to Gemini (model=%s)", self.model)

        # Clone the pre-built config and add system prompt if needed
        config = genai.types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        answer = response.text or ""
        logger.info("Gemini response received (%d chars)", len(answer))
        return answer

    @property
    def provider_name(self) -> str:
        return "gemini"

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton. Useful for testing."""
        if cls._instance and hasattr(cls._instance, '_httpx_client'):
            cls._instance._httpx_client.close()
        cls._instance = None
        cls._initialized = False

