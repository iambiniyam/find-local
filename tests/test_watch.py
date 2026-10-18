import time

from find_local.watch import changed, discover, snapshot


def test_discover_skips_unsupported_and_hidden(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.bin").write_text("b")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "c.md").write_text("c")
    names = {path.name for path in discover([tmp_path])}
    assert names == {"a.txt"}


def test_snapshot_and_changed(tmp_path):
    file = tmp_path / "a.txt"
    file.write_text("one")
    first = snapshot([tmp_path])
    assert str(file.resolve()) in first
    time.sleep(0.01)
    file.write_text("two different")
    second = snapshot([tmp_path])
    assert changed(second, first) == [file.resolve()]
    assert changed(second, second) == []
