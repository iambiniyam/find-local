"""Semantic search over indexed text chunks and images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embed import Embedder
from .index import IndexMismatch, Store


@dataclass(slots=True)
class Result:
    path: str
    score: float
    kind: str
    modality: str
    snippet: str = ""
    size: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "score": round(self.score, 4),
            "kind": self.kind,
            "modality": self.modality,
            "snippet": self.snippet,
            "size": self.size,
        }


def _best_per_file(rows, query: np.ndarray) -> dict[str, tuple[float, str, str, int]]:
    best: dict[str, tuple[float, str, str, int]] = {}
    for row in rows:
        vector = np.frombuffer(row["embedding"], dtype=np.float32)
        if vector.shape[0] != query.shape[0]:
            continue
        score = float(vector @ query)
        path = row["path"]
        if path not in best or score > best[path][0]:
            best[path] = (score, row["text"], row["kind"], int(row["size"]))
    return best


VISUAL_WORDS = (
    "photo",
    "image",
    "picture",
    "screenshot",
    "portrait",
    "face",
    "leaf",
    "plant",
    "dog",
    "cat",
    "beach",
    "car",
    "diagram",
    "chart",
    "logo",
    "scan",
    "person",
)


def _is_visual(query: str) -> bool:
    lowered = query.lower()
    return any(word in lowered for word in VISUAL_WORDS)


def _weights(query: str, kind: str | None) -> tuple[float, float]:
    if kind == "image":
        return 0.0, 1.0
    if kind in {"text", "pdf"}:
        return 1.0, 0.0
    if _is_visual(query):
        return 0.35, 0.65
    return 0.85, 0.15


def _max_score(results: list[Result]) -> float:
    return max((result.score for result in results), default=0.0) or 1.0


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

    expected = store.get_meta("embedder")
    if expected and expected != embedder.name:
        raise IndexMismatch(
            f"This index was built with {expected}. Delete it and index again with {embedder.name}."
        )

    text_results: list[Result] = []
    image_results: list[Result] = []
    text_query = embedder.encode_query(query)
    rows = store.conn.execute(
        """
        SELECT f.path, f.kind, f.size, c.text, c.embedding
        FROM chunks c JOIN files f ON f.id = c.file_id
        """
    ).fetchall()
    text_best = _best_per_file(rows, text_query)
    for path, (score, snippet, file_kind, size) in text_best.items():
        if kind and file_kind != kind:
            continue
        text_results.append(Result(path, score, file_kind, "text", snippet[:240], size))

    visual = _is_visual(query)
    search_images = embedder.can_images and kind in {None, "image"} and (
        visual or kind == "image" or embedder.image_ready
    )
    if search_images:
        try:
            image_query = embedder.encode_image_query(query)
            image_rows = store.conn.execute(
                """
                SELECT f.path, f.kind, f.size, i.embedding
                FROM images i JOIN files f ON f.id = i.file_id
                """
            ).fetchall()
            for row in image_rows:
                vector = np.frombuffer(row["embedding"], dtype=np.float32)
                if vector.shape[0] != image_query.shape[0]:
                    continue
                score = float(vector @ image_query)
                path = row["path"]
                image_results.append(Result(path, score, "image", "image", "", int(row["size"])))
        except RuntimeError:
            pass

    text_weight, image_weight = _weights(query, kind)
    text_scale = _max_score(text_results)
    image_scale = _max_score(image_results)

    merged: dict[str, Result] = {}
    for result in text_results:
        result.score = (result.score / text_scale) * text_weight
        merged[result.path] = result
    for result in image_results:
        result.score = (result.score / image_scale) * image_weight
        current = merged.get(result.path)
        if current is None or result.score > current.score:
            merged[result.path] = result

    ordered = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    ordered = [result for result in ordered if result.score > 0]
    return ordered[:limit]


def stats(store: Store) -> dict[str, int]:
    return store.stats()
