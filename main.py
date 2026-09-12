"""ContextForge — RAG Application Entry Point.

Initializes logging, the RAG pipeline (with hybrid search + reranking),
and starts the FastAPI server. Uses the modern lifespan pattern for
startup/shutdown lifecycle.
"""

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, set_pipeline
from src.rag.pipeline import RAGPipeline
from src.utils.helpers import load_config
from src.utils.logging_config import get_logger, setup_logging

# Initialize logging
config = load_config()
logging_config = config.get("logging", {})
setup_logging(
    log_file=logging_config.get("log_file", "logs/app.log"),
    level=logging_config.get("level", "INFO"),
)

logger = get_logger(__name__)

# Environment detection
ENV = os.getenv("ENV", "development").lower()
IS_DEV = ENV == "development"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup and shutdown.

    Replaces the deprecated @app.on_event("startup") / ("shutdown") pattern.
    Everything before `yield` runs on startup; everything after runs on shutdown.
    """
    # --- Startup ---
    logger.info("Starting ContextForge (env=%s)...", ENV)

    pipeline = RAGPipeline()
    set_pipeline(pipeline)

    # Build BM25 keyword index from existing documents in the vector store
    pipeline.initialize_bm25()

    logger.info(
        "ContextForge ready. LLM provider: %s. Documents in store: %d.",
        config.get("llm", {}).get("provider", "openai"),
        pipeline.vector_store.count(),
    )

    yield  # Application is running and serving requests

    # --- Shutdown ---
    logger.info("Shutting down ContextForge...")
    logger.info("Shutdown complete.")


# Create FastAPI app
app = FastAPI(
    title="ContextForge",
    description="Production-grade Retrieval-Augmented Generation (RAG) API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — allows frontends to call the API.
# In production, restrict allow_origins to your actual domain(s).
allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(router)


if __name__ == "__main__":
    api_config = config.get("api", {})

    run_kwargs = {
        "app": "main:app",
        "host": api_config.get("host", "0.0.0.0"),
        "port": api_config.get("port", 8000),
    }

    if IS_DEV:
        # Development: auto-reload on code changes, single worker
        run_kwargs.update({
            "reload": True,
            "reload_dirs": ["src"],
            "reload_excludes": ["logs/*", "data/*", "*.log", "*.db", "__pycache__/*"],
        })
        logger.info("Running in DEVELOPMENT mode (reload=True)")
    else:
        # Production: no reload, multiple workers
        run_kwargs.update({
            "reload": False,
            "workers": int(os.getenv("WORKERS", "4")),
            "access_log": False,  # Use structured logging instead
        })
        logger.info("Running in PRODUCTION mode (workers=%s)", os.getenv("WORKERS", "4"))

    uvicorn.run(**run_kwargs)
