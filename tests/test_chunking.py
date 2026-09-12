"""Tests for the chunking module."""

import pytest

from src.chunking.chunker import TextChunker
from src.ingestion.loader import Document


class TestTextChunker:
    """Tests for TextChunker."""

    def test_chunk_single_short_document(self):
        """A document shorter than chunk_size should produce one chunk."""
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        docs = [Document(text="Short text.", metadata={"filename": "test.txt"})]

        chunks = chunker.chunk_documents(docs)

        assert len(chunks) == 1
        assert chunks[0].text == "Short text."
        assert chunks[0].metadata["filename"] == "test.txt"
        assert chunks[0].metadata["chunk_index"] == 0

    def test_chunk_long_document(self):
        """A document longer than chunk_size should produce multiple chunks."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        long_text = "This is a sentence. " * 50  # ~1000 chars
        docs = [Document(text=long_text, metadata={"filename": "long.txt"})]

        chunks = chunker.chunk_documents(docs)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["filename"] == "long.txt"
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata

    def test_metadata_preserved(self):
        """Parent metadata should be preserved on each chunk."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        metadata = {
            "filename": "report.pdf",
            "page": 3,
            "file_type": "pdf",
            "source": "/docs/report.pdf",
        }
        docs = [Document(text="Word " * 100, metadata=metadata)]

        chunks = chunker.chunk_documents(docs)

        for chunk in chunks:
            assert chunk.metadata["filename"] == "report.pdf"
            assert chunk.metadata["page"] == 3
            assert chunk.metadata["file_type"] == "pdf"
            assert chunk.metadata["source"] == "/docs/report.pdf"

    def test_empty_document_list(self):
        """Empty input should produce empty output."""
        chunker = TextChunker()
        chunks = chunker.chunk_documents([])
        assert chunks == []

    def test_custom_chunk_size(self):
        """Custom chunk sizes should be respected."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=0)
        # Create text that is exactly 100 chars for predictable splitting
        text = "a" * 100
        docs = [Document(text=text, metadata={})]

        chunks = chunker.chunk_documents(docs)

        assert len(chunks) == 2
        for chunk in chunks:
            assert len(chunk.text) <= 50

    def test_multiple_documents(self):
        """Multiple documents should all be chunked."""
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        docs = [
            Document(text="First document content.", metadata={"filename": "a.txt"}),
            Document(text="Second document content.", metadata={"filename": "b.txt"}),
        ]

        chunks = chunker.chunk_documents(docs)

        assert len(chunks) == 2
        filenames = {c.metadata["filename"] for c in chunks}
        assert filenames == {"a.txt", "b.txt"}
