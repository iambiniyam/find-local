import time
from pathlib import Path

import pymupdf as fitz
from PIL import Image

from find_local.embed import HashEmbedder
from find_local.index import Indexer, Store
from find_local.search import search


def _store(tmp_path: Path) -> Store:
    return Store(tmp_path / "index.db")


def test_text_search(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "fox.txt").write_text("the quick brown fox jumps over the lazy dog")
    (folder / "cake.md").write_text("chocolate cake recipe with cocoa and butter")
    store = _store(tmp_path)
    stats = Indexer(store, HashEmbedder()).index_paths([folder])
    assert stats.files == 2
    assert stats.chunks >= 2

    results = search(store, HashEmbedder(), "fox", limit=5)
    assert results
    assert results[0].path.endswith("fox.txt")

    results = search(store, HashEmbedder(), "cake", limit=5)
    assert results[0].path.endswith("cake.md")


def test_pdf_search(tmp_path):
    path = tmp_path / "notes.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "semantic search notes about offline privacy and local embeddings")
    doc.save(path)
    doc.close()

    store = _store(tmp_path)
    Indexer(store, HashEmbedder()).index_paths([path])
    results = search(store, HashEmbedder(), "privacy", limit=5)
    assert results
    assert results[0].path.endswith("notes.pdf")
    assert results[0].kind == "pdf"


def test_image_indexed(tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (64, 64), "red").save(path)
    store = _store(tmp_path)
    stats = Indexer(store, HashEmbedder()).index_paths([path])
    assert stats.images == 1
    assert store.stats()["images"] == 1


def test_incremental(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    fox = folder / "fox.txt"
    fox.write_text("quick brown fox")
    (folder / "cake.md").write_text("chocolate cake")
    store = _store(tmp_path)

    first = Indexer(store, HashEmbedder()).index_paths([folder])
    assert first.files == 2
    second = Indexer(store, HashEmbedder()).index_paths([folder])
    assert second.files == 0
    assert second.skipped == 2

    time.sleep(0.02)
    fox.write_text("quick brown fox and a new sentence")
    third = Indexer(store, HashEmbedder()).index_paths([folder])
    assert third.files == 1
    assert third.skipped == 1


def test_empty_query(tmp_path):
    store = _store(tmp_path)
    assert search(store, HashEmbedder(), "   ") == []
