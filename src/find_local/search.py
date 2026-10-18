"""Semantic search over indexed text chunks and images."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path

import numpy as np

from .embed import Embedder
from .index import IndexMismatch, Store

FILENAME_BOOST = 0.15

_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "kb": 1024,
    "k": 1024,
    "kib": 1024,
    "mb": 1024**2,
    "m": 1024**2,
    "mib": 1024**2,
    "gb": 1024**3,
    "g": 1024**3,
    "gib": 1024**3,
    "tb": 1024**4,
    "t": 1024**4,
    "tib": 1024**4,
}


def parse_size(value: float | str | None) -> int | None:
    """Parse a byte size such as ``10MB``, ``1.5 GB``, or ``512``.

    Plain numbers are treated as bytes. An empty value means no filter.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"Invalid size: {value!r}")
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"Size cannot be negative: {value!r}")
        return int(value)
    text = re.sub(r"\s+", "", str(value).lower())
    if not text:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([a-z]*)", text)
    if match is None:
        raise ValueError(f"Invalid size {value!r}; use a number with an optional unit, e.g. 10MB.")
    number, unit = match.group(1), match.group(2)
    if unit not in _SIZE_UNITS:
        raise ValueError(f"Unknown size unit {unit!r}; use B, KB, MB, GB, or TB.")
    return int(float(number) * _SIZE_UNITS[unit])


def parse_date(
    value: date | datetime | str | None,
    *,
    end_of_day: bool = False,
) -> float | None:
    """Convert a date or ISO-8601 string to a local epoch timestamp.

    A plain date uses local midnight for ``since`` and the last moment of the
    day for ``until``, so both boundaries are inclusive. A full timestamp is
    used exactly as given.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value if value.tzinfo is not None else value.astimezone()
    elif isinstance(value, date):
        boundary = time.max if end_of_day else time.min
        moment = datetime.combine(value, boundary).astimezone()
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            day = date.fromisoformat(text)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid date {value!r}; use YYYY-MM-DD or an ISO timestamp."
                ) from exc
            moment = parsed if parsed.tzinfo is not None else parsed.astimezone()
        else:
            boundary = time.max if end_of_day else time.min
            moment = datetime.combine(day, boundary).astimezone()
    else:
        raise TypeError("date filters must be a date, datetime, ISO string, or None")
    return moment.timestamp()


def _folder_prefix(folder: str | Path | None) -> str | None:
    if folder is None:
        return None
    text = str(folder).strip()
    if not text:
        return None
    return str(Path(text).expanduser().resolve())


def _in_folder(path: str, folder: str | None) -> bool:
    if folder is None:
        return True
    base = folder.rstrip(os.sep)
    if not base:
        return path.startswith(os.sep)
    return path == base or path.startswith(base + os.sep)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.lower())


def _keyword_score(text: str, query: str) -> float:
    """Cheap keyword overlap used to complement the embedding score."""
    wanted = list(dict.fromkeys(_tokens(query)))
    if not wanted:
        return 0.0
    have = set(_tokens(text))
    matched = sum(1 for term in wanted if term in have)
    coverage = matched / len(wanted)
    phrase = " ".join(wanted) in " ".join(_tokens(text))
    return min(1.0, coverage * 0.8 + (0.2 if phrase else 0.0))


def _filename_match(path: str, query: str) -> float:
    """Return 0..1 query-token coverage in the file name, then the parent path.

    File name hits count fully and directory-only hits count half, so an exact
    file name or stem match scores 1.0.
    """
    query_tokens = list(dict.fromkeys(_tokens(query)))
    if not query_tokens:
        return 0.0
    name = Path(path).name
    stem = Path(path).stem
    normalized = " ".join(query_tokens)
    if normalized in {" ".join(_tokens(name)), " ".join(_tokens(stem))}:
        return 1.0
    name_tokens = set(_tokens(name))
    path_tokens = set(_tokens(path))
    score = 0.0
    for token in query_tokens:
        if token in name_tokens:
            score += 1.0
        elif token in path_tokens:
            score += 0.5
    return score / len(query_tokens)


def _metadata_where(
    since_ts: float | None,
    until_ts: float | None,
    min_bytes: int | None,
    max_bytes: int | None,
) -> tuple[str, list[float | int]]:
    clauses: list[str] = []
    params: list[float | int] = []
    if since_ts is not None:
        clauses.append("f.mtime >= ?")
        params.append(since_ts)
    if until_ts is not None:
        clauses.append("f.mtime <= ?")
        params.append(until_ts)
    if min_bytes is not None:
        clauses.append("f.size >= ?")
        params.append(min_bytes)
    if max_bytes is not None:
        clauses.append("f.size <= ?")
        params.append(max_bytes)
    if not clauses:
        return "", []
    return " WHERE " + " AND ".join(clauses), params


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


def _best_per_file(
    rows,
    query: np.ndarray,
    query_text: str,
    keyword_weight: float,
) -> dict[str, tuple[float, str, str, int]]:
    best: dict[str, tuple[float, str, str, int]] = {}
    for row in rows:
        vector = np.frombuffer(row["embedding"], dtype=np.float32)
        if vector.shape[0] != query.shape[0]:
            continue
        cosine = max(-1.0, min(1.0, float(vector @ query)))
        vector_score = max(0.0, cosine)
        keyword = _keyword_score(row["text"], query_text)
        score = (1.0 - keyword_weight) * vector_score + keyword_weight * keyword
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
    since: date | datetime | str | None = None,
    until: date | datetime | str | None = None,
    min_size: float | str | None = None,
    max_size: float | str | None = None,
    folder: str | Path | None = None,
    filename_boost: float = FILENAME_BOOST,
    keyword_weight: float = 0.25,
) -> list[Result]:
    """Return the best files for ``query``.

    Filters are optional and inclusive. Dates accept ``date`` objects or
    ISO-8601 strings; a date-only ``until`` covers the whole local day. Sizes
    accept byte numbers or strings such as ``10MB``. ``folder`` is a path
    prefix. ``filename_boost`` adds a small bonus for query words in the file
    name or path; set it to zero to disable.
    """
    query = query.strip()
    if not query:
        return []

    since_ts = parse_date(since)
    until_ts = parse_date(until, end_of_day=True)
    if since_ts is not None and until_ts is not None and since_ts > until_ts:
        raise ValueError("since must be on or before until")
    min_bytes = parse_size(min_size)
    max_bytes = parse_size(max_size)
    if min_bytes is not None and max_bytes is not None and min_bytes > max_bytes:
        raise ValueError("min_size must be less than or equal to max_size")
    folder_prefix = _folder_prefix(folder)
    keyword_weight = max(0.0, min(1.0, float(keyword_weight)))

    expected = store.get_meta("embedder")
    if expected and expected != embedder.name:
        raise IndexMismatch(
            f"This index was built with {expected}. Delete it and index again with {embedder.name}."
        )

    where, params = _metadata_where(since_ts, until_ts, min_bytes, max_bytes)
    text_results: list[Result] = []
    image_results: list[Result] = []
    text_query = embedder.encode_query(query)
    rows = store.conn.execute(
        f"""
        SELECT f.path, f.kind, f.size, c.text, c.embedding
        FROM chunks c JOIN files f ON f.id = c.file_id
        {where}
        """,
        params,
    ).fetchall()
    rows = [row for row in rows if _in_folder(row["path"], folder_prefix)]
    text_best = _best_per_file(rows, text_query, query, keyword_weight)
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
                f"""
                SELECT f.path, f.kind, f.size, i.embedding
                FROM images i JOIN files f ON f.id = i.file_id
                {where}
                """,
                params,
            ).fetchall()
            image_rows = [row for row in image_rows if _in_folder(row["path"], folder_prefix)]
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

    if filename_boost > 0:
        for result in merged.values():
            result.score += filename_boost * _filename_match(result.path, query)

    ordered = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    ordered = [
        result
        for result in ordered
        if result.score > 0
        and _in_folder(result.path, folder_prefix)
        and (min_bytes is None or result.size >= min_bytes)
        and (max_bytes is None or result.size <= max_bytes)
    ]
    return ordered[:limit]


def stats(store: Store) -> dict[str, int]:
    return store.stats()
