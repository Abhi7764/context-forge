"""Document loading entry point and factory.

Routes files to the correct loader based on file extension.
This is the single entry point the rest of the application uses
to load documents — callers never touch individual loaders directly.
"""

from pathlib import Path

from src.ingestion.loader import BaseLoader, CSVLoader, Document, PDFLoader, TextLoader
from src.utils.helpers import detect_file_type
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Registry mapping file types to their loader classes
_LOADER_REGISTRY: dict[str, type[BaseLoader]] = {
    "pdf": PDFLoader,
    "txt": TextLoader,
    "csv": CSVLoader,
}


def get_loader(file_type: str) -> BaseLoader:
    """Get the appropriate loader for a file type.

    Args:
        file_type: Lowercase file extension without dot (e.g., 'pdf').

    Returns:
        An instance of the matching loader.

    Raises:
        ValueError: If the file type is not supported.
    """
    loader_class = _LOADER_REGISTRY.get(file_type)
    if loader_class is None:
        supported = ", ".join(sorted(_LOADER_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported file type: '{file_type}'. Supported: {supported}"
        )
    return loader_class()


def load_document(filepath: str | Path) -> list[Document]:
    """Load a document from the given file path.

    Detects file type, selects the appropriate loader, and returns
    a list of Document objects with text and metadata.

    Args:
        filepath: Path to the document file.

    Returns:
        List of Document objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is unsupported or the file is empty.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    file_type = detect_file_type(filepath.name)
    if file_type is None:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")

    logger.info("Loading document: %s (type: %s)", filepath.name, file_type)
    loader = get_loader(file_type)
    documents = loader.load(filepath)
    logger.info("Loaded %d document(s) from %s", len(documents), filepath.name)

    return documents
