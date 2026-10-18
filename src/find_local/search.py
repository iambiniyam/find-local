"""Semantic search over indexed text chunks and images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embed import Embedder
from .index import Store


@dataclass(slots=True)
class Result:
    path: str
    score: float
    kind: str
    modality: str
    snippet: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "score": round(self.score, 4),
            "kind": self.kind,
            "modality": self.modality,
            "snippet": self.snippet,
        }


def _best_per_file(rows, query: np.ndarray) -> dict[str, tuple[float, str, str]]:
    best: dict[str, tuple[float, str, str]] = {}
    for row in rows:
        vector = np.frombuffer(row["embedding"], dtype=np.float32)
        if vector.shape[0] != query.shape[0]:
            continue
        score = float(vector @ query)
        path = row["path"]
        if path not in best or score > best[path][0]:
            best[path] = (score, row["text"], row["kind"])
    return best


def search(
    store: Store,
    embedder: Embedder,
    query: str,
    limit: int = 10,
    kind: str | None = None,
) -> list[Result]:
    query = query.strip()
    if not query:
        return []

    results: dict[str, Result] = {}
    text_query = embedder.encode_texts([query])[0]
    rows = store.conn.execute(
        """
        SELECT f.path, f.kind, c.text, c.embedding
        FROM chunks c JOIN files f ON f.id = c.file_id
        """
    ).fetchall()
    text_best = _best_per_file(rows, text_query)
    for path, (score, snippet, file_kind) in text_best.items():
        if kind and file_kind != kind:
            continue
        results[path] = Result(path, score, file_kind, "text", snippet[:220])

    if embedder.can_images and kind in {None, "image"}:
        try:
            image_query = embedder.encode_image_query(query)
            image_rows = store.conn.execute(
                """
                SELECT f.path, f.kind, i.embedding
                FROM images i JOIN files f ON f.id = i.file_id
                """
            ).fetchall()
            for row in image_rows:
                vector = np.frombuffer(row["embedding"], dtype=np.float32)
                if vector.shape[0] != image_query.shape[0]:
                    continue
                score = float(vector @ image_query)
                path = row["path"]
                if path not in results or score > results[path].score:
                    results[path] = Result(path, score, "image", "image", "[image]")
        except RuntimeError:
            pass

    ordered = sorted(results.values(), key=lambda item: item.score, reverse=True)
    ordered = [result for result in ordered if result.score > 0]
    return ordered[:limit]


def stats(store: Store) -> dict[str, int]:
    return store.stats()
