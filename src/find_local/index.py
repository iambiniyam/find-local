"""SQLite-backed file index with text chunks and image embeddings."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pymupdf as fitz
from PIL import Image

from .config import (
    IMAGE_EXTENSIONS,
    MAX_IMAGE_BYTES,
    MAX_TEXT_BYTES,
    PDF_EXTENSIONS,
    SKIP_DIRS,
    TEXT_EXTENSIONS,
)
from .embed import Embedder


def classify(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if suffix in PDF_EXTENSIONS:
        return "pdf"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    return None


def chunk_text(text: str, size: int = 700, overlap: int = 100) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def read_text_file(path: Path) -> str | None:
    data = path.read_bytes()
    if b"\x00" in data:
        return None
    return data.decode("utf-8", errors="ignore")


def read_pdf(path: Path, max_pages: int = 200) -> str:
    document = fitz.open(path)
    parts: list[str] = []
    for index, page in enumerate(document):
        if index >= max_pages:
            break
        parts.append(page.get_text())
    document.close()
    return "\n".join(parts)


@dataclass(slots=True)
class IndexStats:
    files: int = 0
    chunks: int = 0
    images: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


class Store:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._schema()

    def _schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                mtime REAL NOT NULL,
                size INTEGER NOT NULL,
                kind TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
            CREATE INDEX IF NOT EXISTS idx_images_file ON images(file_id);
            """
        )
        self.conn.commit()

    def stats(self) -> dict[str, int]:
        def count(table: str) -> int:
            return int(self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])

        return {"files": count("files"), "chunks": count("chunks"), "images": count("images")}

    def close(self) -> None:
        self.conn.close()


class Indexer:
    def __init__(self, store: Store, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def index_paths(self, paths: Iterable[Path]) -> IndexStats:
        stats = IndexStats()
        for path in paths:
            for file in self._iter_files(Path(path)):
                try:
                    self._index_file(file, stats)
                except Exception as exc:  # noqa: BLE001 - keep going, report at the end
                    stats.errors.append(f"{file}: {exc}")
        self.store.conn.commit()
        return stats

    def _iter_files(self, root: Path) -> Iterator[Path]:
        if root.is_file():
            yield root.resolve()
            return
        for directory, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS and not name.startswith(".")]
            for filename in filenames:
                if filename.startswith("."):
                    continue
                path = Path(directory) / filename
                if classify(path):
                    yield path.resolve()

    def _index_file(self, path: Path, stats: IndexStats) -> None:
        kind = classify(path)
        if kind is None:
            return
        stat = path.stat()
        if kind in {"text", "pdf"} and stat.st_size > MAX_TEXT_BYTES:
            stats.skipped += 1
            return
        if kind == "image" and stat.st_size > MAX_IMAGE_BYTES:
            stats.skipped += 1
            return

        existing = self.store.conn.execute(
            "SELECT id, mtime, size FROM files WHERE path = ?", (str(path),)
        ).fetchone()
        if existing and abs(existing["mtime"] - stat.st_mtime) < 1 and existing["size"] == stat.st_size:
            stats.skipped += 1
            return
        if existing:
            self.store.conn.execute("DELETE FROM files WHERE id = ?", (existing["id"],))

        cursor = self.store.conn.execute(
            "INSERT INTO files (path, mtime, size, kind) VALUES (?, ?, ?, ?)",
            (str(path), stat.st_mtime, stat.st_size, kind),
        )
        file_id = int(cursor.lastrowid)
        stats.files += 1

        if kind == "text":
            text = read_text_file(path) or ""
            self._add_chunks(file_id, chunk_text(text), stats)
        elif kind == "pdf":
            text = read_pdf(path)
            self._add_chunks(file_id, chunk_text(text), stats)
        elif kind == "image":
            with Image.open(path) as image:
                vectors = self.embedder.encode_images([image.convert("RGB")])
            if len(vectors):
                self.store.conn.execute(
                    "INSERT INTO images (file_id, embedding, dim) VALUES (?, ?, ?)",
                    (file_id, vectors[0].astype(np.float32).tobytes(), vectors.shape[1]),
                )
                stats.images += 1
        self.store.conn.commit()

    def _add_chunks(self, file_id: int, chunks: list[str], stats: IndexStats) -> None:
        if not chunks:
            return
        vectors = self.embedder.encode_texts(chunks)
        for ordinal, (chunk, vector) in enumerate(zip(chunks, vectors)):
            self.store.conn.execute(
                "INSERT INTO chunks (file_id, ordinal, text, embedding, dim) VALUES (?, ?, ?, ?, ?)",
                (file_id, ordinal, chunk, vector.astype(np.float32).tobytes(), vector.shape[0]),
            )
            stats.chunks += 1
