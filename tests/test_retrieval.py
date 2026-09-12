"""Tests for the retrieval module."""

from unittest.mock import MagicMock

import pytest

from src.ingestion.loader import Document
from src.retrieval.retriever import Retriever


class TestRetriever:
    """Tests for the Retriever class."""

    def _make_retriever(self, mock_results: list[Document] | None = None) -> Retriever:
        """Create a Retriever with mocked dependencies.

        Uses MagicMock objects that satisfy BaseVectorStore and BaseEmbedder
        interfaces via duck typing.
        """
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]

        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.return_value = mock_results or []

        return Retriever(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            top_k=3,
        )

    def test_retrieve_returns_documents(self):
        """Retriever should return documents from the vector store."""
        expected_docs = [
            Document(text="Relevant text", metadata={"filename": "doc.txt", "similarity_score": 0.95}),
        ]
        retriever = self._make_retriever(expected_docs)

        results = retriever.retrieve("What is this about?")

        assert len(results) == 1
        assert results[0].text == "Relevant text"
        retriever.embedder.embed_query.assert_called_once_with("What is this about?")

    def test_retrieve_empty_query_raises(self):
        """Empty queries should raise ValueError."""
        retriever = self._make_retriever()

        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.retrieve("")

        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.retrieve("   ")

    def test_retrieve_uses_configured_top_k(self):
        """Retriever should pass top_k to the vector store."""
        retriever = self._make_retriever()
        retriever.retrieve("test query")

        retriever.vector_store.similarity_search.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=3,
        )

    def test_retrieve_no_results(self):
        """Retriever should return empty list when no matches found."""
        retriever = self._make_retriever([])
        results = retriever.retrieve("obscure query")
        assert results == []

    def test_retriever_accepts_any_vector_store(self):
        """Retriever should work with any BaseVectorStore implementation."""
        from src.vectordb.base import BaseVectorStore

        # Verify that the mock satisfies the expected interface
        mock_store = MagicMock(spec=BaseVectorStore)
        mock_store.similarity_search.return_value = []

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1]

        retriever = Retriever(vector_store=mock_store, embedder=mock_embedder, top_k=2)
        results = retriever.retrieve("test")
        assert results == []
        mock_store.similarity_search.assert_called_once()
