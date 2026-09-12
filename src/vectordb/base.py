"""Abstract base class for vector stores.

All vector store providers (Pinecone, ChromaDB, etc.) implement this
interface so the RAG pipeline, retriever, and tests never depend on
a specific database.
"""

import hashlib
from abc import ABC, abstractmethod
from typing import Any

from src.ingestion.loader import Document


class BaseVectorStore(ABC):
    """Interface that every vector store provider must implement.

    Provides default helpers for ID generation and metadata sanitization
    that concrete implementations can reuse.
    """

    @abstractmethod
    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> list[str]:
        """Add document chunks with their embeddings to the store.

        Args:
            documents: List of Document objects (text + metadata).
            embeddings: Corresponding embedding vectors.

        Returns:
            List of generated document IDs.
        """
        ...

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
    ) -> list[Document]:
        """Find the most similar documents to a query embedding.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.

        Returns:
            List of Document objects ranked by similarity.
        """
        ...

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete all documents from the store."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of documents in the store."""
        ...

    # --- Shared helpers available to all providers ---

    @staticmethod
    def generate_id(doc: Document) -> str:
        """Generate a deterministic ID from document content and metadata.

        Uses a hash of the text + source + chunk_index to avoid duplicates
        when the same document is re-ingested.
        """
        source = doc.metadata.get("source", "")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        page = doc.metadata.get("page", 0)
        id_string = f"{source}:page{page}:chunk{chunk_idx}:{doc.text[:100]}"
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]

    @staticmethod
    def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Ensure all metadata values are store-compatible primitive types.

        Most vector databases only support str, int, float, and bool.
        """
        sanitized: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized
