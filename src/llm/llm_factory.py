"""LLM client factory.

Reads the configured provider from config.yaml and returns the
appropriate LLM client instance. The rest of the application
never needs to know which provider is in use.
"""

from src.llm.base import BaseLLMClient
from src.utils.helpers import load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Supported providers and their client classes (lazy imports to avoid
# importing all SDKs when only one provider is used)
_PROVIDERS = {"openai", "anthropic", "gemini", "ollama"}


def get_llm_client(provider: str | None = None) -> BaseLLMClient:
    """Create and return an LLM client for the configured provider.

    Args:
        provider: Override the provider from config. One of:
                  'openai', 'anthropic', 'gemini', 'ollama'.

    Returns:
        An initialized LLM client instance.

    Raises:
        ValueError: If the provider is unknown.
        ImportError: If the provider's SDK is not installed.
    """
    config = load_config().get("llm", {})
    resolved_provider = (provider or config.get("provider", "openai")).lower()

    if resolved_provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider: '{resolved_provider}'. "
            f"Supported: {', '.join(sorted(_PROVIDERS))}"
        )

    logger.info("Initializing LLM provider: %s", resolved_provider)

    if resolved_provider == "openai":
        from src.llm.openai_client import OpenAIClient
        return OpenAIClient()

    elif resolved_provider == "anthropic":
        from src.llm.anthropic_client import AnthropicClient
        return AnthropicClient()

    elif resolved_provider == "gemini":
        from src.llm.gemini_client import GeminiClient
        return GeminiClient()

    elif resolved_provider == "ollama":
        from src.llm.ollama_client import OllamaClient
        return OllamaClient()

    # Should never reach here due to the check above
    raise ValueError(f"Unhandled provider: {resolved_provider}")
