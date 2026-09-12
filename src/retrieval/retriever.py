"""Retrieval module.

Accepts a user query, converts it to an embedding, and retrieves
the most relevant document chunks from the vector store.

This module does NOT call the LLM — retrieval and generation are separate.
Depends only on abstractions (BaseVectorStore, BaseEmbedder), never
on concrete implementations.
"""

import time

from src.embeddings.base import BaseEmbedder
from src.ingestion.loader import Document
from src.utils.helpers import load_config
from src.utils.logging_config import get_logger
from src.vectordb.base import BaseVectorStore

logger = get_logger(__name__)


class Retriever:
    """Retrieves relevant document chunks for a user query.

    Args:
        vector_store: Any BaseVectorStore implementation.
        embedder: Any BaseEmbedder implementation.
        top_k: Number of results to return. Defaults to config value.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        top_k: int | None = None,
    ) -> None:
        config = load_config().get("retrieval", {})
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k or config.get("top_k", 4)

        logger.info("Retriever initialized (top_k=%d)", self.top_k)

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve the most relevant document chunks for a query.

        Args:
            query: The user's question or search query.

        Returns:
            List of Document objects ranked by relevance, each containing
            text, metadata, and a similarity_score in metadata.

        Raises:
            ValueError: If the query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info("Retrieving documents for query: '%s'", query[:100])

        t0 = time.perf_counter()
        query_embedding = self.embedder.embed_query(query)
        t1 = time.perf_counter()
        logger.info("Embedding took %.3f s", t1 - t0)

        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )
        t2 = time.perf_counter()
        logger.info("Similarity search took %.3f s", t2 - t1)

        logger.info("Retrieved %d relevant chunks (total retrieval: %.3f s).", len(results), t2 - t0)
        return results

