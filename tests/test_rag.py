"""Tests for the RAG pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.loader import Document
from src.rag.pipeline import RAGPipeline


class TestRAGPipeline:
    """Tests for the RAGPipeline class."""

    def _make_pipeline(self) -> RAGPipeline:
        """Create a RAGPipeline with all dependencies mocked.

        Mocks satisfy the ABC interfaces via duck typing — the pipeline
        never knows (or cares) which concrete provider is behind them.
        """
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "This is the answer."
        mock_llm.provider_name = "mock"

        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = [[0.1, 0.2]]
        mock_embedder.embed_query.return_value = [0.1, 0.2]

        mock_vector_store = MagicMock()
        mock_vector_store.count.return_value = 5
        mock_vector_store.add_documents.return_value = ["id1"]
        mock_vector_store.similarity_search.return_value = [
            Document(
                text="Relevant context from the document.",
                metadata={
                    "filename": "test.pdf",
                    "page": 1,
                    "similarity_score": 0.92,
                },
            )
        ]

        pipeline = RAGPipeline(
            llm_client=mock_llm,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )
        return pipeline

    def test_query_returns_answer_and_sources(self):
        """Query should return an answer with source metadata."""
        pipeline = self._make_pipeline()
        result = pipeline.query("What is in the document?")

        assert "answer" in result
        assert result["answer"] == "This is the answer."
        assert "sources" in result
        assert len(result["sources"]) >= 1
        assert result["sources"][0]["filename"] == "test.pdf"

    def test_query_empty_question_raises(self):
        """Empty questions should raise ValueError."""
        pipeline = self._make_pipeline()

        with pytest.raises(ValueError, match="Question cannot be empty"):
            pipeline.query("")

    def test_query_calls_llm(self):
        """Query should call the LLM with a prompt containing context."""
        pipeline = self._make_pipeline()
        pipeline.query("What is this?")

        pipeline._llm_client.generate_response.assert_called_once()
        call_args = pipeline._llm_client.generate_response.call_args
        assert "What is this?" in call_args.kwargs.get("prompt", call_args.args[0] if call_args.args else "")

    @patch("src.rag.pipeline.load_document")
    def test_ingest_document(self, mock_load):
        """Ingest should load, chunk, embed, and store."""
        mock_load.return_value = [
            Document(text="Document content here.", metadata={"filename": "test.txt"})
        ]

        pipeline = self._make_pipeline()
        result = pipeline.ingest("/fake/path/test.txt")

        assert result["filename"] == "test.txt"
        assert result["chunks_created"] >= 1
        mock_load.assert_called_once()
        pipeline.embedder.embed_texts.assert_called()
        pipeline.vector_store.add_documents.assert_called()

    def test_extract_sources_deduplication(self):
        """Sources should be deduplicated by filename + page."""
        docs = [
            Document(text="a", metadata={"filename": "doc.pdf", "page": 1, "similarity_score": 0.9}),
            Document(text="b", metadata={"filename": "doc.pdf", "page": 1, "similarity_score": 0.8}),
            Document(text="c", metadata={"filename": "doc.pdf", "page": 2, "similarity_score": 0.7}),
        ]

        sources = RAGPipeline._extract_sources(docs)

        assert len(sources) == 2  # page 1 and page 2
        filenames = {s["filename"] for s in sources}
        assert filenames == {"doc.pdf"}

    def test_pipeline_uses_injected_dependencies(self):
        """Pipeline should use the injected vector store, not create its own."""
        pipeline = self._make_pipeline()

        # The pipeline should use the mock we injected, not call a factory
        pipeline.vector_store.count()
        pipeline.vector_store.count.assert_called()
