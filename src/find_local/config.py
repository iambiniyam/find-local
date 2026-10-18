"""Defaults and paths."""

from __future__ import annotations

import os
from pathlib import Path


def qwen_embedding_dir() -> Path:
    return Path(
        os.getenv(
            "FIND_QWEN_EMBEDDING_DIR",
            str(Path.home() / ".cache" / "find-local" / "models" / "qwen3-embedding"),
        )
    )


def siglip2_dir() -> Path:
    return Path(
        os.getenv(
            "FIND_SIGLIP2_DIR",
            str(Path.home() / ".cache" / "find-local" / "models" / "siglip2-base"),
        )
    )


def embeddinggemma_dir() -> Path:
    return Path(
        os.getenv(
            "FIND_EMBEDDINGGEMMA_DIR",
            str(Path.home() / ".cache" / "find-local" / "models" / "embeddinggemma-2"),
        )
    )


def db_path() -> Path:
    return Path(os.getenv("FIND_DB", str(Path.home() / ".find-local" / "index.db")))


TEXT_MODEL = os.getenv(
    "FIND_TEXT_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
IMAGE_MODEL = os.getenv("FIND_IMAGE_MODEL", "sentence-transformers/clip-ViT-B-32")
EMBEDDINGGEMMA_MODEL = os.getenv("FIND_EMBEDDINGGEMMA_MODEL", "google/embeddinggemma-2")
QWEN_EMBEDDING_MODEL = os.getenv("FIND_QWEN_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
SIGLIP2_MODEL = os.getenv("FIND_SIGLIP2_MODEL", "google/siglip2-base-patch16-224")

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".org", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".html", ".css",
    ".sh", ".bash", ".zsh", ".go", ".rs", ".java", ".kt", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".sql", ".rb", ".php", ".swift", ".tex", ".log",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
PDF_EXTENSIONS = {".pdf"}

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".cache", "dist", "build",
    ".next", ".nuxt", "target", "vendor", "Library", ".Trash",
}

MAX_TEXT_BYTES = 5 * 1024 * 1024
MAX_IMAGE_BYTES = 25 * 1024 * 1024
