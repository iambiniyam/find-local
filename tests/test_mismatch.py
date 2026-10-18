import pytest

from find_local.embed import HashEmbedder
from find_local.index import Indexer, IndexMismatch, Store
from find_local.search import search


class OtherEmbedder(HashEmbedder):
    name = "other"
    dim = 128
    image_dim = 128


def test_model_mismatch_is_reported(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "note.txt").write_text("local embeddings and privacy")

    store = Store(tmp_path / "index.db")
    Indexer(store, HashEmbedder()).index_paths([folder])

    with pytest.raises(IndexMismatch):
        Indexer(store, OtherEmbedder()).index_paths([folder])

    with pytest.raises(IndexMismatch):
        search(store, OtherEmbedder(), "privacy")
