"""Abstract base class for embedding providers.

All embedding providers implement this interface so the RAG pipeline
and retriever can work with any embedding model without code changes.
"""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Interface that every embedding provider must implement."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a single query string.

        Args:
            query: The query text.

        Returns:
            A single embedding vector.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...
