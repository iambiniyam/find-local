from pathlib import Path

from PIL import Image

from find_local.embed import HashEmbedder
from find_local.index import Indexer, Store
from find_local.search import search


def test_image_ocr_adds_searchable_chunks(tmp_path, monkeypatch):
    image = tmp_path / "receipt.png"
    Image.new("RGB", (80, 60), "white").save(image)
    monkeypatch.setattr("find_local.index.ocr_image", lambda path: "Blue Bottle receipt total 14.58")
    store = Store(tmp_path / "index.db")
    Indexer(store, HashEmbedder(), ocr=True).index_paths([tmp_path])
    results = search(store, HashEmbedder(), "receipt", limit=5)
    assert any(Path(result.path).name == "receipt.png" for result in results)
    assert any("14.58" in result.snippet for result in results)


def test_image_ocr_is_opt_in(tmp_path, monkeypatch):
    image = tmp_path / "receipt.png"
    Image.new("RGB", (80, 60), "white").save(image)
    monkeypatch.setattr("find_local.index.ocr_image", lambda path: "should not be called")
    store = Store(tmp_path / "index.db")
    Indexer(store, HashEmbedder(), ocr=False).index_paths([tmp_path])
    results = search(store, HashEmbedder(), "receipt", limit=5)
    assert all(result.kind != "text" for result in results)


def test_keyword_search_handles_concatenated_ocr(tmp_path):
    from find_local.search import _keyword_score
    assert _keyword_score("BLUEBOTTLECOFFEE TOTAL14.58", "blue bottle") > 0.5
