"""Browser-level accessibility checks for find-local."""

from __future__ import annotations

import os
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest
import uvicorn

os.environ["FIND_DB"] = str(Path(tempfile.mkdtemp()) / "index.db")
os.environ["FIND_EMBEDDER"] = "hash"

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright

from find_local.server import app

AXE = Path(__file__).with_name("axe.min.js")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def live_server():
    port = free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def axe_violations(page) -> list[dict]:
    page.add_script_tag(path=str(AXE))
    report = page.evaluate(
        """async () => await axe.run(document, {
          runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21aa"] }
        })"""
    )
    return [item for item in report["violations"] if item["impact"] in {"serious", "critical"}]


def test_initial_and_empty_search_states(live_server):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        tab = browser.new_page(viewport={"width": 1182, "height": 900})
        tab.goto(live_server, wait_until="networkidle")
        assert axe_violations(tab) == []

        tab.fill("#q", "offline privacy notes")
        tab.press("#q", "Enter")
        tab.wait_for_timeout(1000)
        assert axe_violations(tab) == []
        browser.close()
