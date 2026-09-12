"""Tests for the embedding module."""

from unittest.mock import MagicMock, patch

import pytest

from src.embeddings.embedder import SentenceTransformerEmbedder


class TestSentenceTransformerEmbedder:
    """Tests for the SentenceTransformerEmbedder class."""

    def setup_method(self):
        """Reset singleton before each test."""
        SentenceTransformerEmbedder.reset()

    @patch("src.embeddings.embedder.SentenceTransformer")
    def test_embed_texts(self, mock_st_class):
        """embed_texts should return a list of embedding vectors."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedder()
        result = embedder.embed_texts(["hello", "world"])

        assert len(result) == 2
        assert len(result[0]) == 3
        assert isinstance(result[0], list)
        mock_model.encode.assert_called_once()

    @patch("src.embeddings.embedder.SentenceTransformer")
    def test_embed_query(self, mock_st_class):
        """embed_query should return a single embedding vector."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedder()
        result = embedder.embed_query("test query")

        assert len(result) == 3
        assert isinstance(result, list)

    @patch("src.embeddings.embedder.SentenceTransformer")
    def test_embed_empty_list(self, mock_st_class):
        """embed_texts with empty list should return empty list."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedder()
        result = embedder.embed_texts([])

        assert result == []
        mock_model.encode.assert_not_called()

    @patch("src.embeddings.embedder.SentenceTransformer")
    def test_singleton_pattern(self, mock_st_class):
        """Embedder should be a singleton — model loaded once."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        embedder1 = SentenceTransformerEmbedder()
        embedder2 = SentenceTransformerEmbedder()

        assert embedder1 is embedder2
        assert mock_st_class.call_count == 1

    @patch("src.embeddings.embedder.SentenceTransformer")
    def test_dimension_property(self, mock_st_class):
        """Embedder should expose the embedding dimension."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedder()
        assert embedder.dimension == 384

    @patch("src.embeddings.embedder.SentenceTransformer")
    def test_implements_base_interface(self, mock_st_class):
        """SentenceTransformerEmbedder should implement BaseEmbedder."""
        from src.embeddings.base import BaseEmbedder

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        embedder = SentenceTransformerEmbedder()
        assert isinstance(embedder, BaseEmbedder)
