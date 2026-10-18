"""Polling file watcher for incremental local indexing."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

from .config import SKIP_DIRS
from .embed import build_embedder
from .index import Indexer, Store, classify


def discover(paths: Iterable[Path]) -> Iterator[Path]:
    for root in paths:
        root = Path(root).expanduser().resolve()
        if root.is_file():
            if classify(root):
                yield root
            continue
        for directory, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                name for name in dirnames if name not in SKIP_DIRS and not name.startswith(".")
            ]
            for filename in filenames:
                if filename.startswith("."):
                    continue
                path = Path(directory) / filename
                if classify(path):
                    yield path.resolve()


def snapshot(paths: Iterable[Path]) -> dict[str, tuple[float, int]]:
    state: dict[str, tuple[float, int]] = {}
    for path in discover(paths):
        try:
            stat = path.stat()
        except OSError:
            continue
        state[str(path)] = (stat.st_mtime, stat.st_size)
    return state


def changed(current: dict[str, tuple[float, int]], previous: dict[str, tuple[float, int]]) -> list[Path]:
    return [Path(path) for path, stamp in current.items() if previous.get(path) != stamp]


def watch(paths: list[Path], db: Path, embedder_name: str, interval: float = 5.0) -> None:
    store = Store(db)
    indexer = Indexer(store, build_embedder(embedder_name))
    stats = indexer.index_paths(paths)
    print(f"indexed {stats.files} files · watching {len(paths)} path(s)")
    previous = snapshot(paths)
    try:
        while True:
            time.sleep(max(0.5, interval))
            current = snapshot(paths)
            updates = changed(current, previous)
            previous = current
            if not updates:
                continue
            result = indexer.index_paths(updates)
            print(f"updated {result.files} files · {result.chunks} chunks · {result.images} images")
    except KeyboardInterrupt:
        print("stopped")
