"""Static accessibility invariants for the local UI shells."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Audit(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lang = ""
        self.title = ""
        self.has_viewport = False
        self.ids: list[str] = []
        self.labels: set[str] = set()
        self.controls: list[dict[str, str]] = []
        self.images: list[dict[str, str]] = []
        self.buttons: list[dict[str, str]] = []
        self.links: list[dict[str, str]] = []
        self._in_title = False
        self._text = ""

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if tag == "html":
            self.lang = data.get("lang", "")
        if tag == "meta" and data.get("name") == "viewport":
            self.has_viewport = True
        if tag == "title":
            self._in_title = True
        if "id" in data:
            self.ids.append(data["id"])
        if tag == "label" and "for" in data:
            self.labels.add(data["for"])
        if tag in {"input", "select", "textarea"} and data.get("type") != "hidden":
            self.controls.append(data)
        if tag == "img":
            self.images.append(data)
        if tag == "button":
            self.buttons.append(data)
        if tag == "a":
            self.links.append(data)

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        self._text += data


def audit(path: Path) -> tuple[Audit, str]:
    parser = Audit()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser, parser._text


def test_static_shells():
    shells = sorted((ROOT / "src").glob("*/static/index.html"))
    assert shells, "no static index.html found"
    for shell in shells:
        parsed, text = audit(shell)
        assert parsed.lang, f"{shell}: missing html lang"
        assert parsed.title.strip(), f"{shell}: missing title"
        assert parsed.has_viewport, f"{shell}: missing viewport meta"
        assert "#main" in {link.get("href") for link in parsed.links}, f"{shell}: missing skip link"
        assert "main" in parsed.ids, f"{shell}: missing main id"
        assert len(parsed.ids) == len(set(parsed.ids)), f"{shell}: duplicate ids"
        for control in parsed.controls:
            name = control.get("id", "")
            described = name in parsed.labels or control.get("aria-label") or control.get("aria-labelledby")
            assert described, f"{shell}: unlabeled control {control}"
        for image in parsed.images:
            assert "alt" in image, f"{shell}: image without alt {image}"
        for button in parsed.buttons:
            assert button.get("type") in {"button", "submit", "reset"}, f"{shell}: button without type {button}"
        assert text.strip(), f"{shell}: empty document"
