"""ChromaDB vector store implementation.

Wraps ChromaDB behind the BaseVectorStore interface. Used when
config.yaml has `vector_db.provider: chroma`.
"""

from typing import Any

import chromadb

from src.ingestion.loader import Document
from src.utils.helpers import get_project_root, load_config
from src.utils.logging_config import get_logger
from src.vectordb.base import BaseVectorStore

logger = get_logger(__name__)


class ChromaStore(BaseVectorStore):
    """Persistent ChromaDB vector store.

    Args:
        persist_directory: Path for ChromaDB persistent storage.
        collection_name: Name of the ChromaDB collection.
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        config = load_config().get("vector_db", {})
        project_root = get_project_root()

        self.persist_directory = str(
            project_root / (persist_directory or config.get("chroma_persist_directory", "data/chroma"))
        )
        self.collection_name = collection_name or config.get(
            "collection_name", "rag_documents"
        )

        logger.info(
            "Initializing ChromaDB (dir=%s, collection=%s)",
            self.persist_directory,
            self.collection_name,
        )

        self._client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "ChromaDB ready. Collection '%s' has %d documents.",
            self.collection_name,
            self._collection.count(),
        )

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> list[str]:
        """Add document chunks with their embeddings to ChromaDB."""
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(documents)} documents vs {len(embeddings)} embeddings"
            )

        if not documents:
            logger.warning("No documents to add.")
            return []

        ids = [self.generate_id(doc) for doc in documents]
        texts = [doc.text for doc in documents]
        metadatas = [self.sanitize_metadata(doc.metadata) for doc in documents]

        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info("Added %d documents to collection '%s'.", len(ids), self.collection_name)
        return ids

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
    ) -> list[Document]:
        """Find the most similar documents in ChromaDB."""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count() or top_k),
            include=["documents", "metadatas", "distances"],
        )

        documents: list[Document] = []
        if results["documents"] and results["documents"][0]:
            for text, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                meta = dict(metadata) if metadata else {}
                meta["similarity_score"] = round(1 - distance, 4)
                documents.append(Document(text=text, metadata=meta))

        logger.info("Similarity search returned %d results.", len(documents))
        return documents

    def delete_collection(self) -> None:
        """Delete and recreate the ChromaDB collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Deleted and recreated collection '%s'.", self.collection_name)

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()
