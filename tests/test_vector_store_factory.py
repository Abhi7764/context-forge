"""Tests for the vector store factory."""

from unittest.mock import patch, MagicMock

import pytest

from src.vectordb.vector_store_factory import get_vector_store


class TestVectorStoreFactory:
    """Tests for get_vector_store()."""

    def test_unknown_provider_raises(self):
        """Unknown providers should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown vector store provider"):
            get_vector_store(provider="weaviate")

    @patch("src.vectordb.vector_store_factory.load_config")
    def test_default_provider_from_config(self, mock_config):
        """Factory should read provider from config when not overridden."""
        mock_config.return_value = {"vector_db": {"provider": "chroma"}}

        with patch("src.vectordb.vector_store.ChromaStore") as mock_chroma:
            mock_chroma.return_value = MagicMock()
            store = get_vector_store()
            mock_chroma.assert_called_once()

    @patch("src.vectordb.vector_store_factory.load_config")
    def test_explicit_provider_overrides_config(self, mock_config):
        """Explicit provider argument should override config."""
        mock_config.return_value = {"vector_db": {"provider": "pinecone"}}

        with patch("src.vectordb.vector_store.ChromaStore") as mock_chroma:
            mock_chroma.return_value = MagicMock()
            # Config says pinecone, but we override to chroma
            store = get_vector_store(provider="chroma")
            mock_chroma.assert_called_once()

    @patch("src.vectordb.vector_store_factory.load_config")
    def test_pinecone_provider_loads_pinecone(self, mock_config):
        """Provider 'pinecone' should instantiate PineconeStore."""
        mock_config.return_value = {"vector_db": {"provider": "pinecone"}}

        with patch("src.vectordb.pinecone_store.PineconeStore") as mock_pinecone:
            mock_pinecone.return_value = MagicMock()
            store = get_vector_store(provider="pinecone")
            mock_pinecone.assert_called_once()

    @patch("src.vectordb.vector_store_factory.load_config")
    def test_chroma_provider_loads_chroma(self, mock_config):
        """Provider 'chroma' should instantiate ChromaStore."""
        mock_config.return_value = {"vector_db": {"provider": "chroma"}}

        with patch("src.vectordb.vector_store.ChromaStore") as mock_chroma:
            mock_chroma.return_value = MagicMock()
            store = get_vector_store(provider="chroma")
            mock_chroma.assert_called_once()

    def test_returned_store_implements_base(self):
        """Factory should return something that implements BaseVectorStore."""
        from src.vectordb.base import BaseVectorStore

        with patch("src.vectordb.vector_store.ChromaStore") as mock_chroma:
            mock_instance = MagicMock(spec=BaseVectorStore)
            mock_chroma.return_value = mock_instance
            store = get_vector_store(provider="chroma")
            assert isinstance(store, BaseVectorStore)
