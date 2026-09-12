"""BM25 keyword search engine.

Builds an in-memory BM25 index from document texts for keyword-based
retrieval. Used alongside vector similarity search in the hybrid
retriever to catch exact keyword matches that semantic search misses.

BM25 (Best Matching 25) is a bag-of-words ranking function that scores
documents by how many query terms they contain, weighted by term rarity.
"""

import re

from rank_bm25 import BM25Okapi

from src.ingestion.loader import Document
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer.

    Lowercases text, strips punctuation, and splits on whitespace.
    Good enough for BM25 — no heavy NLP dependencies needed.
    """
    text = text.lower()
    # Remove punctuation, keep alphanumeric and spaces
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


class BM25Search:
    """In-memory BM25 keyword search over document chunks.

    Builds a BM25 index from a list of documents. Supports searching
    by exact word matches and returning ranked results.

    Usage:
        bm25 = BM25Search()
        bm25.build_index(documents)
        results = bm25.search("what is machine learning", top_k=10)
    """

    def __init__(self) -> None:
        self._index: BM25Okapi | None = None
        self._documents: list[Document] = []
        logger.info("BM25Search initialized.")

    @property
    def is_ready(self) -> bool:
        """Check if the index has been built."""
        return self._index is not None and len(self._documents) > 0

    def build_index(self, documents: list[Document]) -> None:
        """Build the BM25 index from a list of documents.

        Args:
            documents: List of Document objects to index.
        """
        if not documents:
            logger.warning("No documents to index for BM25.")
            return

        self._documents = documents
        tokenized_corpus = [_tokenize(doc.text) for doc in documents]
        self._index = BM25Okapi(tokenized_corpus)

        logger.info("BM25 index built with %d documents.", len(documents))

    def search(self, query: str, top_k: int = 10) -> list[Document]:
        """Search the BM25 index for documents matching the query.

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            List of Document objects ranked by BM25 score, with
            'bm25_score' added to each document's metadata.
        """
        if not self.is_ready:
            logger.warning("BM25 index not built. Returning empty results.")
            return []

        tokenized_query = _tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Get top_k indices sorted by score (descending)
        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results: list[Document] = []
        for idx, score in scored_indices:
            if score <= 0:
                continue  # Skip zero-score documents

            doc = self._documents[idx]
            # Create a copy with BM25 score in metadata
            result_doc = Document(
                text=doc.text,
                metadata={**doc.metadata, "bm25_score": round(float(score), 4)},
            )
            results.append(result_doc)

        logger.info(
            "BM25 search returned %d results (query: '%s').",
            len(results),
            query[:50],
        )
        return results
