from __future__ import annotations

import os
from datetime import date, datetime, time, timezone
from pathlib import Path

import pytest

from find_local.embed import HashEmbedder
from find_local.index import Indexer, Store
from find_local.search import parse_date, parse_size, search


def _epoch(year: int, month: int, day: int, hour: int = 12) -> float:
    return datetime(year, month, day, hour).timestamp()


def _seed(tmp_path: Path) -> tuple[Store, Path, Path]:
    docs = tmp_path / "docs"
    docs.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()

    old = docs / "old.md"
    old.write_text("project notes from the early meeting")
    os.utime(old, (_epoch(2024, 1, 10), _epoch(2024, 1, 10)))

    large = docs / "new.md"
    large.write_text("project " + "x" * 4096)
    os.utime(large, (_epoch(2024, 6, 1), _epoch(2024, 6, 1)))

    archived = archive / "archived.md"
    archived.write_text("project notes in the archive")
    os.utime(archived, (_epoch(2024, 3, 15), _epoch(2024, 3, 15)))

    store = Store(tmp_path / "index.db")
    Indexer(store, HashEmbedder()).index_paths([docs, archive])
    return store, docs, archive


def _names(results) -> set[str]:
    return {Path(result.path).name for result in results}


def test_parse_size_values():
    assert parse_size(None) is None
    assert parse_size("") is None
    assert parse_size(512) == 512
    assert parse_size(1.5) == 1
    assert parse_size("512") == 512
    assert parse_size("10KB") == 10 * 1024
    assert parse_size("10 MB") == 10 * 1024**2
    assert parse_size("1.5 GB") == int(1.5 * 1024**3)
    assert parse_size("2TiB") == 2 * 1024**4

    with pytest.raises(ValueError):
        parse_size("10XB")
    with pytest.raises(ValueError):
        parse_size("-10MB")
    with pytest.raises(ValueError):
        parse_size(-5)
    with pytest.raises(TypeError):
        parse_size(True)


def test_parse_date_inclusive_day_bounds():
    day = date(2024, 3, 15)
    start = parse_date(day)
    end = parse_date(day, end_of_day=True)

    assert start == datetime(2024, 3, 15).timestamp()
    assert end == datetime.combine(day, time.max).timestamp()
    assert end > start
    assert parse_date("2024-03-15") == start
    assert parse_date("2024-03-15", end_of_day=True) == end
    assert parse_date("2024-03-15T08:30:00") == datetime(2024, 3, 15, 8, 30).timestamp()


def test_parse_date_accepts_aware_timestamps():
    aware = datetime(2024, 3, 15, 8, 30, tzinfo=timezone.utc)
    assert parse_date(aware) == aware.timestamp()
    assert parse_date("2024-03-15T08:30:00Z") == aware.timestamp()


def test_parse_date_rejects_garbage():
    with pytest.raises(ValueError):
        parse_date("next tuesday")


def test_filter_by_since_and_until(tmp_path):
    store, _, _ = _seed(tmp_path)

    assert _names(search(store, HashEmbedder(), "project", since="2024-01-10")) == {
        "old.md",
        "new.md",
        "archived.md",
    }
    assert _names(search(store, HashEmbedder(), "project", since="2024-05-01")) == {"new.md"}
    assert _names(search(store, HashEmbedder(), "project", since="2024-04-01")) == {"new.md"}

    assert _names(search(store, HashEmbedder(), "project", until="2024-01-10")) == {"old.md"}
    assert _names(search(store, HashEmbedder(), "project", until="2024-02-01")) == {"old.md"}


def test_filter_by_size(tmp_path):
    store, _, _ = _seed(tmp_path)

    assert _names(search(store, HashEmbedder(), "project", min_size="1KB")) == {"new.md"}
    assert _names(search(store, HashEmbedder(), "project", min_size=10)) == {
        "old.md",
        "new.md",
        "archived.md",
    }
    assert _names(search(store, HashEmbedder(), "project", max_size="1KB")) == {
        "old.md",
        "archived.md",
    }
    assert _names(
        search(store, HashEmbedder(), "project", min_size=10, max_size=100)
    ) == {"old.md", "archived.md"}


def test_filter_by_folder(tmp_path):
    store, docs, archive = _seed(tmp_path)

    assert _names(search(store, HashEmbedder(), "project", folder=str(docs))) == {
        "old.md",
        "new.md",
    }
    assert _names(search(store, HashEmbedder(), "project", folder=f"{docs}/")) == {
        "old.md",
        "new.md",
    }
    assert _names(search(store, HashEmbedder(), "project", folder=str(archive))) == {
        "archived.md"
    }
    assert _names(search(store, HashEmbedder(), "project", folder=str(docs / "missing"))) == set()


def test_filters_combine(tmp_path):
    store, docs, _ = _seed(tmp_path)
    results = search(
        store,
        HashEmbedder(),
        "project",
        since="2024-01-01",
        until="2024-12-31",
        min_size="1KB",
        max_size="10KB",
        folder=str(docs),
    )
    assert _names(results) == {"new.md"}


def test_invalid_filter_ranges(tmp_path):
    store, _, _ = _seed(tmp_path)

    with pytest.raises(ValueError, match="since"):
        search(store, HashEmbedder(), "project", since="2024-05-01", until="2024-04-01")
    with pytest.raises(ValueError, match="min_size"):
        search(store, HashEmbedder(), "project", min_size="10MB", max_size="1MB")
    with pytest.raises(ValueError):
        search(store, HashEmbedder(), "project", since="not-a-date")
    with pytest.raises(ValueError):
        search(store, HashEmbedder(), "project", min_size="a few megabytes")


def test_filename_boost_ranks_exact_name_first(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    body = "quarterly budget totals for the finance team"
    (folder / "notes.md").write_text(body)
    (folder / "budget-report.md").write_text(body)

    store = Store(tmp_path / "index.db")
    Indexer(store, HashEmbedder()).index_paths([folder])

    results = search(store, HashEmbedder(), "budget report", limit=5)
    assert len(results) == 2
    assert Path(results[0].path).name == "budget-report.md"


def test_exact_filename_surfaces_without_semantic_overlap(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "zebra-report.md").write_text("completely unrelated words")
    (folder / "alpha.md").write_text("zebra facts from the savanna")

    store = Store(tmp_path / "index.db")
    Indexer(store, HashEmbedder()).index_paths([folder])

    results = search(store, HashEmbedder(), "zebra-report", limit=5)
    assert [Path(result.path).name for result in results] == ["alpha.md", "zebra-report.md"]

    without_boost = search(store, HashEmbedder(), "zebra-report", limit=5, filename_boost=0)
    assert _names(without_boost) == {"alpha.md"}


def test_path_tokens_contribute_to_filename_boost(tmp_path):
    (tmp_path / "budget").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "budget" / "notes.md").write_text("monthly totals for the team")
    (tmp_path / "other" / "notes.md").write_text("monthly totals for the team")

    store = Store(tmp_path / "index.db")
    Indexer(store, HashEmbedder()).index_paths([tmp_path / "budget", tmp_path / "other"])

    results = search(store, HashEmbedder(), "budget", limit=5)
    assert _names(results) == {"notes.md"}
    assert "budget" in Path(results[0].path).parts
