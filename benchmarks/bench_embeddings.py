"""Smoke benchmark for find-local embedding stacks.

Usage:
    python benchmarks/bench_embeddings.py --embedder auto
    python benchmarks/bench_embeddings.py --embedder gemma
    python benchmarks/bench_embeddings.py --embedder legacy
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from find_local.embed import build_embedder
from find_local.index import Indexer, Store
from find_local.search import search

QUERIES = {
    "invoice from March": "expenses-march.md",
    "offline privacy notes": "offline-privacy.md",
    "a leaf with orange spots": "leaf.jpg",
    "portrait of a person": "face.jpg",
    "chocolate cake recipe": "cake-recipe.md",
    "semantic search without a cloud": "handbook.pdf",
    "march expenses": "expenses-march.md",
    "local embeddings": "offline-privacy.md",
    "someone's face": "face.jpg",
    "baking instructions": "cake-recipe.md",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", default="auto", choices=["auto", "gemma", "legacy"])
    parser.add_argument("--examples", default="examples")
    args = parser.parse_args()

    examples = Path(args.examples).resolve()
    if not examples.exists():
        raise SystemExit(f"examples folder not found: {examples}")

    with tempfile.TemporaryDirectory() as temp:
        db = Path(temp) / "index.db"
        embedder = build_embedder(args.embedder)
        started = time.perf_counter()
        stats = Indexer(Store(db), embedder).index_paths([examples])
        index_seconds = round(time.perf_counter() - started, 2)

        rows = []
        for query, expected in QUERIES.items():
            started = time.perf_counter()
            results = search(Store(db), embedder, query, limit=5)
            elapsed = round((time.perf_counter() - started) * 1000, 1)
            paths = [item.path for item in results]
            rank = next(
                (index + 1 for index, path in enumerate(paths) if path.endswith(expected)),
                None,
            )
            rows.append(
                {
                    "query": query,
                    "expected": expected,
                    "rank": rank,
                    "ms": elapsed,
                    "top": [Path(path).name for path in paths],
                }
            )

    top1 = sum(1 for row in rows if row["rank"] == 1) / len(rows)
    top3 = sum(1 for row in rows if row["rank"] and row["rank"] <= 3) / len(rows)
    mrr = sum(1 / row["rank"] for row in rows if row["rank"]) / len(rows)
    payload = {
        "embedder": embedder.name,
        "index_seconds": index_seconds,
        "files": stats.files,
        "chunks": stats.chunks,
        "images": stats.images,
        "top1": top1,
        "top3": top3,
        "mrr": round(mrr, 3),
        "avg_ms": round(sum(row["ms"] for row in rows) / len(rows), 1),
        "rows": rows,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
