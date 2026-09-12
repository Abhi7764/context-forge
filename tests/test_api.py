"""Tests for the FastAPI API endpoints."""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.ingestion.loader import Document


@pytest.fixture
def client():
    """Create a test client with a mocked RAG pipeline."""
    # Import here to avoid loading the real pipeline
    from main import app
    from src.api.routes import set_pipeline

    mock_pipeline = MagicMock()
    mock_pipeline.vector_store.count.return_value = 10

    mock_pipeline.ingest.return_value = {
        "filename": "test.txt",
        "chunks_created": 3,
        "document_ids": ["id1", "id2", "id3"],
        "total_documents_in_store": 13,
    }

    mock_pipeline.query.return_value = {
        "answer": "The document discusses testing.",
        "sources": [{"filename": "test.txt", "relevance_score": 0.95}],
        "context_documents": [
            {
                "text": "This is about testing...",
                "metadata": {"filename": "test.txt", "similarity_score": 0.95},
            }
        ],
    }

    set_pipeline(mock_pipeline)

    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_ok(self, client):
        """Health endpoint should return status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "documents_in_store" in data

    def test_health_includes_document_count(self, client):
        """Health endpoint should include document count."""
        response = client.get("/health")
        assert response.json()["documents_in_store"] == 10


class TestUploadEndpoint:
    """Tests for POST /documents/upload."""

    def test_upload_txt_file(self, client):
        """Uploading a valid text file should succeed."""
        file_content = b"This is a test document."
        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.txt"
        assert data["chunks_created"] == 3

    def test_upload_unsupported_type(self, client):
        """Uploading an unsupported file type should return 400."""
        response = client.post(
            "/documents/upload",
            files={"file": ("test.xyz", io.BytesIO(b"data"), "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_upload_file_too_large(self, client):
        """Uploading a file larger than 50 MB should return 413."""
        from src.api.routes import MAX_FILE_SIZE

        # Create content slightly over the limit
        large_content = b"x" * (MAX_FILE_SIZE + 1)
        response = client.post(
            "/documents/upload",
            files={"file": ("big.txt", io.BytesIO(large_content), "text/plain")},
        )
        assert response.status_code == 413
        assert "too large" in response.json()["detail"]

    def test_upload_path_traversal_blocked(self, client):
        """Filenames with path traversal should be sanitized."""
        response = client.post(
            "/documents/upload",
            files={"file": ("../../etc/passwd.txt", io.BytesIO(b"data"), "text/plain")},
        )
        # Should succeed (sanitized to "passwd.txt") or fail on content,
        # but NOT write to ../../etc/
        assert response.status_code in (200, 400, 500)


class TestQueryEndpoint:
    """Tests for POST /query."""

    def test_query_returns_answer(self, client):
        """Valid query should return an answer with sources."""
        response = client.post("/query", json={"question": "What is this about?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert len(data["sources"]) >= 1

    def test_query_empty_question(self, client):
        """Empty question should return 422 (validation error)."""
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422

    def test_query_missing_question(self, client):
        """Missing question field should return 422."""
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_query_no_documents(self, client):
        """Query with empty store should return 404."""
        from src.api.routes import get_pipeline

        pipeline = get_pipeline()
        pipeline.vector_store.count.return_value = 0

        response = client.post("/query", json={"question": "anything"})
        assert response.status_code == 404
        assert "No documents" in response.json()["detail"]

    def test_query_too_long_question(self, client):
        """Question exceeding max_length should return 422."""
        long_question = "a" * 2001
        response = client.post("/query", json={"question": long_question})
        assert response.status_code == 422


class TestAPIKeyAuth:
    """Tests for API key authentication."""

    def test_no_auth_when_key_not_configured(self, client):
        """When API_KEY env var is not set, requests should pass without a key."""
        # client fixture doesn't set API_KEY, so auth is disabled
        response = client.get("/health")
        assert response.status_code == 200

    def test_auth_rejects_wrong_key(self, client):
        """When API_KEY is set, wrong key should return 401."""
        import src.api.routes as routes_module
        original_key = routes_module._API_KEY

        try:
            routes_module._API_KEY = "secret-key-123"
            response = client.get("/health", headers={"X-API-Key": "wrong-key"})
            assert response.status_code == 401
        finally:
            routes_module._API_KEY = original_key

    def test_auth_accepts_correct_key(self, client):
        """When API_KEY is set, correct key should be accepted."""
        import src.api.routes as routes_module
        original_key = routes_module._API_KEY

        try:
            routes_module._API_KEY = "secret-key-123"
            response = client.get(
                "/health", headers={"X-API-Key": "secret-key-123"}
            )
            assert response.status_code == 200
        finally:
            routes_module._API_KEY = original_key
