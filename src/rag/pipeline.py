"""Central RAG pipeline.

Orchestrates the full retrieval-augmented generation flow:
  Query → Hybrid Retrieve → Rerank → Build Context → Prompt → LLM → Answer

Also handles document ingestion:
  File → Load → Chunk → Embed → Store → Update BM25 Index

This module has NO dependency on FastAPI. It can be used
independently from the REST API (e.g., in scripts, notebooks).

Depends only on abstractions (BaseVectorStore, BaseEmbedder,
BaseLLMClient). Concrete providers are resolved via factories.
"""

import time
from pathlib import Path
from typing import Any

from src.chunking.chunker import TextChunker
from src.embeddings.base import BaseEmbedder
from src.embeddings.embedder_factory import get_embedder
from src.ingestion.document_loader import load_document
from src.ingestion.loader import Document
from src.llm.base import BaseLLMClient
from src.llm.llm_factory import get_llm_client
from src.prompts.prompt_templates import RAG_SYSTEM_PROMPT, build_rag_prompt
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.helpers import load_config
from src.utils.logging_config import get_logger
from src.vectordb.base import BaseVectorStore
from src.vectordb.vector_store_factory import get_vector_store

logger = get_logger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline for document ingestion and query answering.

    This is the main entry point for the application's business logic.
    API routes and scripts should interact with this class rather than
    calling individual modules directly.

    Uses a hybrid retrieval pipeline:
      1. Semantic search (vector similarity)
      2. BM25 keyword search
      3. Reciprocal Rank Fusion (score merging)
      4. Cross-encoder reranking (optional)

    All dependencies are injected via constructor or resolved through
    factories. The pipeline never imports concrete provider classes.

    Args:
        llm_client: An LLM client instance. If None, uses the factory.
        vector_store: A vector store instance. If None, uses the factory.
        embedder: An embedder instance. If None, uses the factory.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        vector_store: BaseVectorStore | None = None,
        embedder: BaseEmbedder | None = None,
    ) -> None:
        self.embedder = embedder or get_embedder()
        self.vector_store = vector_store or get_vector_store()
        self.chunker = TextChunker()

        # Check retrieval strategy from config
        config = load_config()
        strategy = config.get("retrieval", {}).get("strategy", "hybrid")

        # Use hybrid retriever (semantic + BM25 + reranking)
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
        )
        self._strategy = strategy

        # Eagerly initialize the LLM client so its HTTP connection
        # is created once at startup, not on every query.
        self._llm_client = llm_client or get_llm_client()

        # Track all ingested chunks for BM25 index
        self._all_chunks: list[Document] = []

        logger.info("RAG pipeline initialized (strategy=%s).", self._strategy)

    @property
    def llm_client(self) -> BaseLLMClient:
        """Return the LLM client (initialized at startup)."""
        return self._llm_client

    def _build_bm25_index_from_store(self) -> None:
        """Build BM25 index from documents already in the vector store.

        Fetches all documents via a broad similarity search and builds
        the BM25 keyword index from them. Called at startup.
        """
        doc_count = self.vector_store.count()
        if doc_count == 0:
            logger.info("No documents in store — BM25 index is empty.")
            return

        # Use a zero-vector query to fetch all documents from the store.
        # This gets us the text content needed for the BM25 index.
        dim = self.embedder.dimension
        zero_vector = [0.0] * dim
        all_docs = self.vector_store.similarity_search(
            query_embedding=zero_vector,
            top_k=min(doc_count, 10000),  # Cap at 10k for memory
        )

        self._all_chunks = all_docs
        self.retriever.build_bm25_index(all_docs)
        logger.info("BM25 index loaded from store (%d documents).", len(all_docs))

    def initialize_bm25(self) -> None:
        """Public method to initialize BM25 index from existing documents.

        Called during application startup after the pipeline is created.
        """
        self._build_bm25_index_from_store()

    def ingest(self, filepath: str | Path) -> dict[str, Any]:
        """Ingest a document: load → chunk → embed → store → update BM25.

        Args:
            filepath: Path to the document file.

        Returns:
            Summary dict with filename, chunk count, and document IDs.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is unsupported or empty.
        """
        filepath = Path(filepath)
        logger.info("Ingesting document: %s", filepath.name)

        # 1. Load document
        documents = load_document(filepath)

        # 2. Chunk
        chunks = self.chunker.chunk_documents(documents)

        # 3. Embed
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)

        # 4. Store
        ids = self.vector_store.add_documents(chunks, embeddings)

        # 5. Update BM25 index with new chunks
        self._all_chunks.extend(chunks)
        self.retriever.build_bm25_index(self._all_chunks)

        result = {
            "filename": filepath.name,
            "chunks_created": len(chunks),
            "document_ids": ids,
            "total_documents_in_store": self.vector_store.count(),
        }

        logger.info(
            "Ingestion complete: %s → %d chunks stored.",
            filepath.name,
            len(chunks),
        )
        return result

    def query(self, question: str) -> dict[str, Any]:
        """Answer a question using hybrid retrieval + LLM generation.

        Pipeline:
          1. Hybrid retrieve (semantic + BM25 + RRF + rerank)
          2. Build prompt with retrieved context
          3. Call LLM for answer generation

        Args:
            question: The user's question.

        Returns:
            Dict with 'answer', 'sources', and 'context_documents'.

        Raises:
            ValueError: If the question is empty.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        logger.info("Processing query: '%s'", question[:100])
        t_query_start = time.perf_counter()

        # 1. Hybrid retrieve (semantic + BM25 + rerank)
        relevant_docs = self.retriever.retrieve(question)
        t_after_retrieve = time.perf_counter()

        # 2. Build prompt
        prompt = build_rag_prompt(relevant_docs, question)
        logger.info("Prompt length: %d chars", len(prompt))

        # 3. Call LLM
        t_llm_start = time.perf_counter()
        answer = self.llm_client.generate_response(
            prompt=prompt,
            system_prompt=RAG_SYSTEM_PROMPT,
        )
        t_llm_end = time.perf_counter()
        logger.info("LLM call took %.3f s", t_llm_end - t_llm_start)

        # 4. Extract source metadata
        sources = self._extract_sources(relevant_docs)

        result = {
            "answer": answer,
            "sources": sources,
            "context_documents": [
                {
                    "text": doc.text[:200] + "..." if len(doc.text) > 200 else doc.text,
                    "metadata": doc.metadata,
                }
                for doc in relevant_docs
            ],
        }

        t_query_end = time.perf_counter()
        logger.info(
            "Query complete in %.3f s (retrieve=%.3f s, llm=%.3f s). Sources: %d",
            t_query_end - t_query_start,
            t_after_retrieve - t_query_start,
            t_llm_end - t_llm_start,
            len(sources),
        )
        return result

    @staticmethod
    def _extract_sources(documents: list) -> list[dict[str, Any]]:
        """Extract unique source information from retrieved documents.

        Args:
            documents: List of retrieved Document objects.

        Returns:
            List of dicts with source filename, page, and similarity score.
        """
        seen: set[str] = set()
        sources: list[dict[str, Any]] = []

        for doc in documents:
            filename = doc.metadata.get("filename", "Unknown")
            page = doc.metadata.get("page")
            key = f"{filename}:{page}"

            if key not in seen:
                seen.add(key)
                source: dict[str, Any] = {"filename": filename}
                if page:
                    source["page"] = page
                score = doc.metadata.get("rerank_score") or doc.metadata.get("similarity_score")
                if score:
                    source["relevance_score"] = score
                sources.append(source)

        return sources
