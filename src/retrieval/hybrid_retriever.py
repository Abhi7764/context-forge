"""Hybrid retriever with Reciprocal Rank Fusion (RRF).

Combines semantic search (vector similarity) and keyword search (BM25)
using RRF score fusion, then optionally reranks the merged results
with a cross-encoder model.

This is the industry-standard retrieval pipeline used by Google,
OpenAI, Cohere, and most production RAG systems:

    Query
      ├── Semantic Search (vector DB)  → top N candidates
      ├── Keyword Search  (BM25)       → top N candidates
      ▼
    Merge via RRF (deduplicate + fuse scores)
      ▼
    Cross-Encoder Reranker (optional)
      ▼
    Final top-K results → sent to LLM
"""

import time
from typing import Any

from src.embeddings.base import BaseEmbedder
from src.ingestion.loader import Document
from src.retrieval.bm25_search import BM25Search
from src.retrieval.reranker import Reranker
from src.utils.helpers import load_config
from src.utils.logging_config import get_logger
from src.vectordb.base import BaseVectorStore

logger = get_logger(__name__)


def _reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF assigns each document a score of 1/(k + rank) for each list
    it appears in, then sums the scores across all lists. This gives
    a fair, balanced merge that doesn't depend on raw score magnitudes.

    Formula: RRF_score(doc) = Σ 1 / (k + rank_i)

    Args:
        ranked_lists: List of ranked document lists (from different retrievers).
        k: RRF constant (default 60, standard in literature).

    Returns:
        Merged and deduplicated list of Documents, sorted by RRF score.
    """
    # Track RRF scores by document text (as unique key)
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_docs in ranked_lists:
        for rank, doc in enumerate(ranked_docs):
            # Use text content as the deduplication key
            key = doc.text[:500]  # First 500 chars as key
            rrf_score = 1.0 / (k + rank + 1)  # rank is 0-indexed

            rrf_scores[key] = rrf_scores.get(key, 0.0) + rrf_score

            # Keep the version with the most metadata
            if key not in doc_map or len(doc.metadata) > len(doc_map[key].metadata):
                doc_map[key] = doc

    # Sort by RRF score descending
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    results: list[Document] = []
    for key in sorted_keys:
        doc = doc_map[key]
        merged_doc = Document(
            text=doc.text,
            metadata={
                **doc.metadata,
                "rrf_score": round(rrf_scores[key], 6),
            },
        )
        results.append(merged_doc)

    return results


class HybridRetriever:
    """Hybrid retriever combining semantic + keyword search with reranking.

    Retrieval pipeline:
      1. Semantic search via vector store (finds meaning-similar docs)
      2. BM25 keyword search (finds exact keyword matches)
      3. Merge via Reciprocal Rank Fusion
      4. Rerank via cross-encoder (optional)

    Args:
        vector_store: Any BaseVectorStore implementation.
        embedder: Any BaseEmbedder implementation.
        top_k: Final number of results to return.
        semantic_top_k: Candidates from semantic search.
        bm25_top_k: Candidates from BM25 search.
        rrf_k: RRF fusion constant.
        reranking_enabled: Whether to use cross-encoder reranking.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        top_k: int | None = None,
        semantic_top_k: int | None = None,
        bm25_top_k: int | None = None,
        rrf_k: int | None = None,
        reranking_enabled: bool | None = None,
    ) -> None:
        config = load_config()
        retrieval_config = config.get("retrieval", {})
        reranking_config = config.get("reranking", {})

        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k or retrieval_config.get("top_k", 3)
        self.semantic_top_k = semantic_top_k or retrieval_config.get("semantic_top_k", 10)
        self.bm25_top_k = bm25_top_k or retrieval_config.get("bm25_top_k", 10)
        self.rrf_k = rrf_k or retrieval_config.get("rrf_k", 60)

        # BM25 keyword search
        self.bm25 = BM25Search()

        # Reranking (optional)
        if reranking_enabled is None:
            reranking_enabled = reranking_config.get("enabled", True)
        self.reranking_enabled = reranking_enabled
        self._reranker: Reranker | None = None

        if self.reranking_enabled:
            self._reranker = Reranker()

        logger.info(
            "HybridRetriever initialized (top_k=%d, semantic=%d, bm25=%d, rerank=%s)",
            self.top_k, self.semantic_top_k, self.bm25_top_k, self.reranking_enabled,
        )

    def build_bm25_index(self, documents: list[Document]) -> None:
        """Build or rebuild the BM25 index from documents.

        Called during pipeline initialization and after document ingestion.

        Args:
            documents: All document chunks stored in the vector database.
        """
        self.bm25.build_index(documents)

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve relevant documents using hybrid search + reranking.

        Args:
            query: The user's question or search query.

        Returns:
            List of Document objects ranked by relevance.

        Raises:
            ValueError: If the query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info("Hybrid retrieval for query: '%s'", query[:100])
        t_start = time.perf_counter()

        # --- Step 1: Semantic search ---
        t0 = time.perf_counter()
        query_embedding = self.embedder.embed_query(query)
        semantic_results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=self.semantic_top_k,
        )
        t1 = time.perf_counter()
        logger.info(
            "Semantic search: %d results in %.3f s",
            len(semantic_results), t1 - t0,
        )

        # --- Step 2: BM25 keyword search ---
        t2 = time.perf_counter()
        bm25_results = self.bm25.search(query, top_k=self.bm25_top_k)

        t3 = time.perf_counter()
        logger.info(
            "BM25 search: %d results in %.3f s",
            len(bm25_results), t3 - t2,
        )

        # --- Step 3: Merge via Reciprocal Rank Fusion ---
        merged = _reciprocal_rank_fusion(
            [semantic_results, bm25_results],
            k=self.rrf_k,
        )
        logger.info("RRF merged: %d unique candidates.", len(merged))

        # --- Step 4: Rerank (optional) ---
        if self.reranking_enabled and self._reranker and len(merged) > 1:
            t4 = time.perf_counter()
            final_results = self._reranker.rerank(
                query=query,
                documents=merged,
                top_k=self.top_k,
            )
            t5 = time.perf_counter()
            logger.info("Reranking: %d → %d in %.3f s", len(merged), len(final_results), t5 - t4)
        else:
            final_results = merged[:self.top_k]


        t_end = time.perf_counter()
        logger.info(
            "Hybrid retrieval complete: %d results in %.3f s "
            "(semantic=%.3fs, bm25=%.3fs, total=%.3fs)",
            len(final_results),
            t_end - t_start,
            t1 - t0,
            t3 - t2,
            t_end - t_start,
        )
        return final_results
