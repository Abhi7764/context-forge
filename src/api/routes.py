"""FastAPI route definitions.

Thin API layer — all business logic lives in the RAG pipeline.
Routes handle HTTP concerns only: request parsing, file handling,
response formatting, and error responses.

Production features:
  - Sync handlers (avoids blocking the async event loop)
  - File upload size limit (50 MB)
  - Filename sanitization (prevents path traversal)
  - API key authentication (optional, enabled via API_KEY env var)
  - Rate limiting via slowapi (optional, enabled if slowapi is installed)
"""

import os
import shutil
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from src.utils.helpers import detect_file_type, get_project_root
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# --- Constants ---

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


# --- API Key Authentication ---

_API_KEY = os.getenv("API_KEY")  # None = auth disabled


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Verify the API key if one is configured.

    If API_KEY env var is not set, authentication is disabled (open access).
    If API_KEY is set, every request must include an X-API-Key header.
    """
    if _API_KEY is None:
        return  # Auth disabled — no key required

    if x_api_key is None or x_api_key != _API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide 'X-API-Key' header.",
        )


# Router with optional API key dependency
router = APIRouter(dependencies=[Depends(verify_api_key)])

# Late-initialized pipeline reference (set by main.py on startup)
_pipeline = None


def set_pipeline(pipeline: Any) -> None:
    """Set the RAG pipeline instance for routes to use.

    Called once during application startup from main.py.
    """
    global _pipeline
    _pipeline = pipeline


def get_pipeline() -> Any:
    """Get the RAG pipeline, raising an error if not initialized."""
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized. Server is starting up.",
        )
    return _pipeline


def _sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal attacks.

    Strips directory components (e.g., '../../etc/passwd' → 'passwd')
    and removes hidden file prefixes.
    """
    safe_name = PurePosixPath(filename).name
    # Also strip leading dots to prevent hidden files
    safe_name = safe_name.lstrip(".")
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return safe_name


# --- Request / Response models ---


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to answer using the ingested documents.",
    )


class QueryResponse(BaseModel):
    """Response body for the /query endpoint."""

    answer: str
    sources: list[dict[str, Any]]
    context_documents: list[dict[str, Any]] = []


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    documents_in_store: int


class IngestResponse(BaseModel):
    """Response body for the /documents/upload endpoint."""

    message: str
    filename: str
    chunks_created: int
    total_documents_in_store: int


# --- Routes ---
# NOTE: All handlers are plain `def` (not `async def`) so FastAPI
# automatically runs them in a thread pool. This prevents blocking
# the event loop when calling synchronous Pinecone/Gemini/embedding code.


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns the service status and document count.
    """
    pipeline = get_pipeline()
    return HealthResponse(
        status="ok",
        documents_in_store=pipeline.vector_store.count(),
    )


@router.post("/documents/upload", response_model=IngestResponse)
def upload_document(file: UploadFile = File(...)) -> IngestResponse:
    """Upload and ingest a document.

    Accepts PDF, TXT, and CSV files (max 50 MB). The document is loaded,
    chunked, embedded, and stored in the vector database.

    Args:
        file: The uploaded file.

    Returns:
        Ingestion summary with chunk count.
    """
    pipeline = get_pipeline()

    # Validate filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    safe_filename = _sanitize_filename(file.filename)

    file_type = detect_file_type(safe_filename)
    if file_type is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{Path(safe_filename).suffix}'. "
            f"Supported: .pdf, .txt, .csv",
        )

    # Read file content with size limit
    contents = file.file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)} MB.",
        )

    # Save to data/documents directory, then ingest
    try:
        suffix = Path(safe_filename).suffix
        doc_dir = get_project_root() / "data" / "documents"
        doc_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, dir=str(doc_dir)
        ) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        # Rename to preserve original filename for metadata
        final_path = tmp_path.parent / safe_filename
        if final_path.exists():
            final_path.unlink()
        tmp_path.rename(final_path)

        logger.info("File saved: %s (%d bytes)", final_path, len(contents))
        result = pipeline.ingest(final_path)

        return IngestResponse(
            message=f"Document '{safe_filename}' ingested successfully.",
            filename=result["filename"],
            chunks_created=result["chunks_created"],
            total_documents_in_store=result["total_documents_in_store"],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Ingestion failed for '%s': %s", safe_filename, e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest document: {type(e).__name__}",
        )


@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    """Query the ingested documents.

    Retrieves relevant chunks, sends them with the question to the LLM,
    and returns the generated answer along with source references.

    Args:
        request: JSON body with a 'question' field.

    Returns:
        Answer, source references, and context documents.
    """
    pipeline = get_pipeline()

    t_start = time.perf_counter()

    doc_count = pipeline.vector_store.count()
    if doc_count == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents have been ingested yet. "
            "Upload documents first via POST /documents/upload.",
        )

    try:
        result = pipeline.query(request.question)
        t_end = time.perf_counter()
        logger.info("Total /query handler took %.3f s", t_end - t_start)
        return QueryResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (ConnectionError, RuntimeError) as e:
        logger.error("LLM call failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"LLM service error: {type(e).__name__}. Check your API key and provider configuration.",
        )
    except Exception as e:
        logger.error("Query failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {type(e).__name__}",
        )
