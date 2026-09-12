"""Document loaders for different file types.

Each loader converts a raw file into a list of Document objects
with extracted text and metadata. New file types can be supported
by adding a new loader class here.
"""

import csv
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    """A loaded document chunk with text and metadata.

    Attributes:
        text: The extracted text content.
        metadata: Key-value metadata (source, filename, file_type, page, etc.).
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, filepath: Path) -> list[Document]:
        """Load a file and return a list of Document objects.

        Args:
            filepath: Absolute path to the file.

        Returns:
            List of Document objects with text and metadata.
        """
        ...


class PDFLoader(BaseLoader):
    """Loader for PDF files using pypdf."""

    def load(self, filepath: Path) -> list[Document]:
        """Load a PDF file, extracting text page by page.

        Args:
            filepath: Path to the PDF file.

        Returns:
            One Document per page with page number in metadata.

        Raises:
            ImportError: If pypdf is not installed.
            ValueError: If no text could be extracted.
        """
        from pypdf import PdfReader

        logger.info("Loading PDF: %s", filepath.name)
        reader = PdfReader(str(filepath))
        documents: list[Document] = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                documents.append(
                    Document(
                        text=text.strip(),
                        metadata={
                            "source": str(filepath),
                            "filename": filepath.name,
                            "file_type": "pdf",
                            "page": page_num,
                            "total_pages": len(reader.pages),
                        },
                    )
                )

        if not documents:
            logger.warning("No text could be extracted from PDF: %s", filepath.name)
            raise ValueError(f"No text could be extracted from PDF: {filepath.name}")

        logger.info("Extracted %d pages from %s", len(documents), filepath.name)
        return documents


class TextLoader(BaseLoader):
    """Loader for plain text files."""

    def load(self, filepath: Path) -> list[Document]:
        """Load a text file as a single Document.

        Args:
            filepath: Path to the text file.

        Returns:
            A single-element list with the file content.

        Raises:
            ValueError: If the file is empty.
        """
        logger.info("Loading TXT: %s", filepath.name)
        text = filepath.read_text(encoding="utf-8").strip()

        if not text:
            logger.warning("Text file is empty: %s", filepath.name)
            raise ValueError(f"Text file is empty: {filepath.name}")

        return [
            Document(
                text=text,
                metadata={
                    "source": str(filepath),
                    "filename": filepath.name,
                    "file_type": "txt",
                },
            )
        ]


class CSVLoader(BaseLoader):
    """Loader for CSV files.

    Each row is converted to a text representation preserving column names.
    """

    def load(self, filepath: Path) -> list[Document]:
        """Load a CSV file, converting each row into a Document.

        Args:
            filepath: Path to the CSV file.

        Returns:
            One Document per row with column-value pairs as text.

        Raises:
            ValueError: If the CSV has no data rows.
        """
        logger.info("Loading CSV: %s", filepath.name)
        content = filepath.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(content))
        documents: list[Document] = []

        for row_num, row in enumerate(reader, start=1):
            # Convert row to readable text: "Column: Value, Column: Value"
            text_parts = [f"{col}: {val}" for col, val in row.items() if val]
            text = "; ".join(text_parts)

            if text.strip():
                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "source": str(filepath),
                            "filename": filepath.name,
                            "file_type": "csv",
                            "row": row_num,
                        },
                    )
                )

        if not documents:
            logger.warning("CSV file has no data rows: %s", filepath.name)
            raise ValueError(f"CSV file has no data rows: {filepath.name}")

        logger.info("Extracted %d rows from %s", len(documents), filepath.name)
        return documents
