"""Centralized logging configuration for ContextForge.

Sets up a single logger that writes to both console and a rotating log file.
All modules should import `get_logger` instead of using `logging` directly.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_file: str = "logs/app.log", level: str = "INFO") -> None:
    """Configure the root logger with console and file handlers.

    Args:
        log_file: Path to the log file (relative to project root).
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler with rotation (5 MB per file, keep 3 backups)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Suppress verbose third-party logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    # Avoid duplicate handlers on repeated calls
    if not root_logger.handlers:
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger.

    Args:
        name: Logger name, typically `__name__` from the calling module.

    Returns:
        A configured Logger instance.
    """
    return logging.getLogger(name)
