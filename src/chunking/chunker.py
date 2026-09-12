"""Text chunking module.

Splits documents into smaller, overlapping chunks suitable for
embedding and retrieval. Each chunk preserves metadata from
its parent document.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loader import Document
from src.utils.helpers import load_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class TextChunker:
    """Splits documents into overlapping text chunks.

    Uses LangChain's RecursiveCharacterTextSplitter, which tries to split
    on natural boundaries (paragraphs, sentences, words) before falling
    back to character-level splits.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between adjacent chunks.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        config = load_config().get("chunking", {})
        self.chunk_size = chunk_size or config.get("chunk_size", 500)
        self.chunk_overlap = chunk_overlap or config.get("chunk_overlap", 50)

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.info(
            "TextChunker initialized (chunk_size=%d, overlap=%d)",
            self.chunk_size,
            self.chunk_overlap,
        )

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split a list of documents into smaller chunks.

        Each chunk inherits metadata from its parent document, with an
        additional `chunk_index` field.

        Args:
            documents: List of Document objects to split.

        Returns:
            List of chunked Document objects.
        """
        chunks: list[Document] = []

        for doc in documents:
            text_chunks = self._splitter.split_text(doc.text)

            for idx, chunk_text in enumerate(text_chunks):
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_index": idx,
                    "total_chunks": len(text_chunks),
                }
                chunks.append(Document(text=chunk_text, metadata=chunk_metadata))

        logger.info(
            "Chunked %d document(s) into %d chunks", len(documents), len(chunks)
        )
        return chunks
