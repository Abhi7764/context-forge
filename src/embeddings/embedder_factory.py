"""Embedder factory.

Reads the configured provider from config.yaml and returns the
appropriate embedder instance. Currently supports Sentence Transformers;
future providers (OpenAI, Cohere) plug in here.
"""

from src.embeddings.base import BaseEmbedder
from src.utils.helpers import load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_PROVIDERS = {"sentence-transformers"}


def get_embedder(provider: str | None = None) -> BaseEmbedder:
    """Create and return an embedder for the configured provider.

    Args:
        provider: Override the provider from config. Currently:
                  'sentence-transformers'.

    Returns:
        An initialized embedder instance.

    Raises:
        ValueError: If the provider is unknown.
    """
    config = load_config().get("embedding", {})
    resolved_provider = (provider or config.get("provider", "sentence-transformers")).lower()

    if resolved_provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown embedding provider: '{resolved_provider}'. "
            f"Supported: {', '.join(sorted(_PROVIDERS))}"
        )

    logger.info("Initializing embedding provider: %s", resolved_provider)

    if resolved_provider == "sentence-transformers":
        from src.embeddings.embedder import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder()

    raise ValueError(f"Unhandled provider: {resolved_provider}")
