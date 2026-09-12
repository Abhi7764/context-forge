"""Pinecone vector store implementation.

Wraps Pinecone behind the BaseVectorStore interface. Used when
config.yaml has `vector_db.provider: pinecone`.

Uses the gRPC transport for significantly lower query latency
compared to the default REST client (persistent connection,
binary protocol, no per-request TLS handshake).

Requires PINECONE_API_KEY in .env.
"""

import time
from typing import Any

from pinecone.grpc import PineconeGRPC, GRPCIndex
from pinecone import ServerlessSpec

from src.ingestion.loader import Document
from src.utils.helpers import get_env, load_config
from src.utils.logging_config import get_logger
from src.vectordb.base import BaseVectorStore

logger = get_logger(__name__)


class PineconeStore(BaseVectorStore):
    """Pinecone serverless vector store.

    Creates the index on first use if it doesn't exist.
    Uses gRPC transport for lower latency.

    Args:
        index_name: Pinecone index name. Defaults to config value.
        api_key: Pinecone API key. Defaults to PINECONE_API_KEY env var.
        dimension: Embedding dimension. Auto-detected if not provided.
    """

    def __init__(
        self,
        index_name: str | None = None,
        api_key: str | None = None,
        dimension: int = 384,
    ) -> None:
        config = load_config().get("vector_db", {})

        self.index_name = index_name or config.get("index_name", "context-forge")
        self.namespace = config.get("namespace", "default")

        resolved_key = api_key or get_env("PINECONE_API_KEY")
        if not resolved_key or resolved_key.startswith("your_"):
            raise ValueError(
                "PINECONE_API_KEY is not set. Add it to your .env file."
            )

        logger.info("Initializing Pinecone gRPC (index=%s)", self.index_name)

        self._pc = PineconeGRPC(api_key=resolved_key)

        # Create index if it doesn't exist
        existing_indexes = [idx.name for idx in self._pc.list_indexes()]
        if self.index_name not in existing_indexes:
            cloud = config.get("cloud", "aws")
            region = config.get("region", "us-east-1")
            logger.info(
                "Creating Pinecone index '%s' (dimension=%d, cloud=%s, region=%s)",
                self.index_name, dimension, cloud, region,
            )
            self._pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

        self._index: GRPCIndex = self._pc.Index(self.index_name)

        # Cache vector count to avoid repeated describe_index_stats RPCs
        self._cached_count: int = 0
        self._count_fetched_at: float = 0.0
        self._COUNT_CACHE_TTL: float = 60.0  # seconds

        self._refresh_count()
        logger.info("Pinecone ready. Index '%s' has %d vectors.", self.index_name, self._cached_count)

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> list[str]:
        """Add document chunks with their embeddings to Pinecone."""
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(documents)} documents vs {len(embeddings)} embeddings"
            )

        if not documents:
            logger.warning("No documents to add.")
            return []

        vectors: list[dict[str, Any]] = []
        ids: list[str] = []

        for doc, embedding in zip(documents, embeddings):
            doc_id = self.generate_id(doc)
            ids.append(doc_id)

            metadata = self.sanitize_metadata(doc.metadata)
            # Pinecone stores text in metadata for retrieval
            metadata["text"] = doc.text[:40000]  # Pinecone metadata limit ~40KB

            vectors.append({
                "id": doc_id,
                "values": embedding,
                "metadata": metadata,
            })

        # Upsert in batches of 100 (Pinecone recommended batch size)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(vectors=batch, namespace=self.namespace)

        # Invalidate cached count after adding documents
        self._cached_count += len(ids)
        self._count_fetched_at = time.monotonic()

        logger.info("Added %d documents to Pinecone index '%s'.", len(ids), self.index_name)
        return ids

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
    ) -> list[Document]:
        """Find the most similar documents in Pinecone."""
        results = self._index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
        )

        documents: list[Document] = []
        for match in results.get("matches", []):
            metadata = dict(match.get("metadata", {}))
            text = metadata.pop("text", "")
            metadata["similarity_score"] = round(match.get("score", 0), 4)
            documents.append(Document(text=text, metadata=metadata))

        logger.info("Similarity search returned %d results.", len(documents))
        return documents

    def delete_collection(self) -> None:
        """Delete all vectors in the namespace."""
        self._index.delete(delete_all=True, namespace=self.namespace)
        # Reset cached count after deletion
        self._cached_count = 0
        self._count_fetched_at = time.monotonic()
        logger.info(
            "Deleted all vectors in namespace '%s' of index '%s'.",
            self.namespace, self.index_name,
        )

    def _refresh_count(self) -> None:
        """Fetch the latest count from Pinecone and update the cache."""
        stats = self._index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        ns_stats = namespaces.get(self.namespace, {})
        self._cached_count = ns_stats.get("vector_count", 0)
        self._count_fetched_at = time.monotonic()

    def count(self) -> int:
        """Return the number of vectors in the index (cached).

        Uses a cached value to avoid a network round-trip on every call.
        The cache is refreshed when documents are added/deleted or
        when the TTL expires.
        """
        age = time.monotonic() - self._count_fetched_at
        if age > self._COUNT_CACHE_TTL:
            self._refresh_count()
        return self._cached_count
