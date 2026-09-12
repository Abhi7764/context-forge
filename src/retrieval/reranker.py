"""Cross-encoder reranker.

Uses a cross-encoder model to re-score retrieved documents against
the user's query. Unlike bi-encoders (used for embedding), cross-encoders
read the query AND document together, giving much more accurate
relevance scores — but they're slower, so we only rerank a small
candidate set (typically 10-20 documents).

Uses singleton pattern to load the ~80MB model only once.
"""

from sentence_transformers import CrossEncoder

from src.ingestion.loader import Document
from src.utils.helpers import load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Default model — small, fast, trained on MS MARCO passage ranking
_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Reranks documents using a cross-encoder model.

    The cross-encoder reads each (query, document) pair together and
    outputs a relevance score. This is much more accurate than cosine
    similarity but too slow to run on the entire corpus — so we use
    it only on the top candidates from the initial retrieval.

    Uses singleton pattern: the model is loaded once and reused.

    Args:
        model_name: HuggingFace cross-encoder model name.
    """

    _instance: "Reranker | None" = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> "Reranker":
        """Singleton — only one model loaded in memory."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str | None = None) -> None:
        if Reranker._initialized:
            return

        config = load_config().get("reranking", {})
        self.model_name = model_name or config.get("model", _DEFAULT_MODEL)

        logger.info("Loading reranker model: %s", self.model_name)
        try:
            self._model = CrossEncoder(self.model_name, local_files_only=True)
        except Exception:
            logger.info("Model not cached locally, downloading: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)

        Reranker._initialized = True
        logger.info("Reranker ready (model=%s).", self.model_name)

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 3,
    ) -> list[Document]:
        """Rerank documents by relevance to the query.

        Args:
            query: The user's query.
            documents: Candidate documents to rerank.
            top_k: Number of top results to return after reranking.

        Returns:
            List of Document objects reranked by cross-encoder score,
            with 'rerank_score' added to metadata.
        """
        if not documents:
            return []

        if len(documents) <= 1:
            return documents[:top_k]

        # Build (query, document_text) pairs for the cross-encoder
        pairs = [(query, doc.text) for doc in documents]

        # Score all pairs at once (batched inference)
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Pair documents with scores, sort by score descending
        scored_docs = sorted(
            zip(documents, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        # Build result with rerank_score in metadata
        results: list[Document] = []
        for doc, score in scored_docs[:top_k]:
            reranked_doc = Document(
                text=doc.text,
                metadata={
                    **doc.metadata,
                    "rerank_score": round(float(score), 4),
                },
            )
            results.append(reranked_doc)

        logger.info(
            "Reranked %d candidates → top %d results.",
            len(documents),
            len(results),
        )
        return results

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton. Useful for testing."""
        cls._instance = None
        cls._initialized = False
