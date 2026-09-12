"""Tests for the embedder factory."""

from unittest.mock import patch, MagicMock

import pytest

from src.embeddings.embedder_factory import get_embedder


class TestEmbedderFactory:
    """Tests for get_embedder()."""

    def test_unknown_provider_raises(self):
        """Unknown providers should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedder(provider="openai-embeddings")

    @patch("src.embeddings.embedder_factory.load_config")
    def test_default_provider_from_config(self, mock_config):
        """Factory should read provider from config when not overridden."""
        mock_config.return_value = {"embedding": {"provider": "sentence-transformers"}}

        with patch("src.embeddings.embedder.SentenceTransformerEmbedder") as mock_st:
            mock_st.return_value = MagicMock()
            embedder = get_embedder()
            mock_st.assert_called_once()

    @patch("src.embeddings.embedder_factory.load_config")
    def test_explicit_provider_overrides_config(self, mock_config):
        """Explicit provider argument should override config."""
        mock_config.return_value = {"embedding": {"provider": "something-else"}}

        with patch("src.embeddings.embedder.SentenceTransformerEmbedder") as mock_st:
            mock_st.return_value = MagicMock()
            embedder = get_embedder(provider="sentence-transformers")
            mock_st.assert_called_once()

    @patch("src.embeddings.embedder_factory.load_config")
    def test_returned_embedder_implements_base(self, mock_config):
        """Factory should return something that implements BaseEmbedder."""
        from src.embeddings.base import BaseEmbedder

        mock_config.return_value = {"embedding": {"provider": "sentence-transformers"}}

        with patch("src.embeddings.embedder.SentenceTransformerEmbedder") as mock_st:
            mock_instance = MagicMock(spec=BaseEmbedder)
            mock_st.return_value = mock_instance
            embedder = get_embedder()
            assert isinstance(embedder, BaseEmbedder)
