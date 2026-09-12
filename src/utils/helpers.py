"""Utility helpers for ContextForge.

Provides configuration loading, environment variable access,
and common helper functions used across the application.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env once at import time
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load and cache application configuration from config.yaml.

    Args:
        config_path: Optional override path. Defaults to project root config.yaml.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if config_path is None:
        path = _PROJECT_ROOT / "config.yaml"
    else:
        path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def get_env(key: str, default: str | None = None) -> str | None:
    """Get an environment variable.

    Args:
        key: Environment variable name.
        default: Fallback value if the variable is not set.

    Returns:
        The environment variable value, or the default.
    """
    return os.getenv(key, default)


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT


def detect_file_type(filename: str) -> str | None:
    """Detect document type from filename extension.

    Args:
        filename: Name or path of the file.

    Returns:
        Lowercase file extension without dot (e.g., 'pdf', 'txt', 'csv'),
        or None if the extension is not recognized.
    """
    supported = {"pdf", "txt", "csv"}
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext if ext in supported else None
