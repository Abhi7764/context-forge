"""Sentence Transformers embedding implementation.

Loads the embedding model once (singleton) and provides a clean
interface for generating embeddings for both documents and queries.
"""

from sentence_transformers import SentenceTransformer

from src.embeddings.base import BaseEmbedder
from src.utils.helpers import load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Generates text embeddings using a Sentence Transformers model.

    The model is loaded once on initialization and reused for all
    subsequent embedding calls. Uses singleton pattern to avoid
    loading the ~90MB model multiple times.

    Args:
        model_name: HuggingFace model identifier. Defaults to config value.
    """

    _instance: "SentenceTransformerEmbedder | None" = None
    _initialized: bool = False

    def __new__(cls, model_name: str | None = None) -> "SentenceTransformerEmbedder":
        """Singleton pattern — only one model instance in memory."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str | None = None) -> None:
        if SentenceTransformerEmbedder._initialized:
            return

        config = load_config().get("embedding", {})
        self.model_name = model_name or config.get(
            "model", "sentence-transformers/all-MiniLM-L6-v2"
        )

        logger.info("Loading embedding model: %s", self.model_name)
        try:
            self._model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(self.model_name)

        if hasattr(self._model, "get_embedding_dimension"):
            self._dimension = self._model.get_embedding_dimension()
        else:
            self._dimension = self._model.get_sentence_embedding_dimension()

        logger.info(
            "Embedding model loaded (dimension=%d)", self._dimension
        )

        SentenceTransformerEmbedder._initialized = True

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []

        embeddings = self._model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a single query string."""
        return self.embed_texts([query])[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance. Useful for testing."""
        cls._instance = None
        cls._initialized = False
