"""Vector store factory.

Reads the configured provider from config.yaml and returns the
appropriate vector store instance. The rest of the application
never needs to know which database is in use.
"""

from src.utils.helpers import load_config
from src.utils.logging_config import get_logger
from src.vectordb.base import BaseVectorStore

logger = get_logger(__name__)

_PROVIDERS = {"pinecone", "chroma"}


def get_vector_store(provider: str | None = None) -> BaseVectorStore:
    """Create and return a vector store for the configured provider.

    Args:
        provider: Override the provider from config. One of:
                  'pinecone', 'chroma'.

    Returns:
        An initialized vector store instance.

    Raises:
        ValueError: If the provider is unknown.
    """
    config = load_config().get("vector_db", {})
    resolved_provider = (provider or config.get("provider", "pinecone")).lower()

    if resolved_provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown vector store provider: '{resolved_provider}'. "
            f"Supported: {', '.join(sorted(_PROVIDERS))}"
        )

    logger.info("Initializing vector store provider: %s", resolved_provider)

    if resolved_provider == "pinecone":
        from src.vectordb.pinecone_store import PineconeStore
        return PineconeStore()

    elif resolved_provider == "chroma":
        from src.vectordb.vector_store import ChromaStore
        return ChromaStore()

    raise ValueError(f"Unhandled provider: {resolved_provider}")
