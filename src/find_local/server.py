"""Local web search UI."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import db_path
from .embed import build_embedder
from .index import IndexMismatch, Store
from .search import search as run_search

DB = Path(os.environ.get("FIND_DB", str(db_path())))
STATIC = Path(__file__).resolve().parent / "static"

store = Store(DB)
embedder = build_embedder(os.environ.get("FIND_EMBEDDER", "auto"))
WARM = {"done": False}


def _warm() -> None:
    try:
        embedder.warmup()
    except Exception:  # noqa: BLE001, S110 - a failed warmup should not kill the server
        pass
    finally:
        WARM["done"] = True


threading.Thread(target=_warm, daemon=True).start()
app = FastAPI(title="find-local", version="0.2.6")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", media_type="text/html")


@app.get("/healthz")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "index": str(DB),
        "stats": store.stats(),
        "warming": not WARM["done"],
        "embedder": {"name": embedder.name, "dim": embedder.dim},
    }


@app.get("/api/status")
def status() -> JSONResponse:
    return JSONResponse(
        {
            "index": str(DB),
            "stats": store.stats(),
            "warming": not WARM["done"],
            "embedder": {"name": embedder.name, "dim": embedder.dim},
        }
    )


@app.get("/api/search")
def api_search(
    q: str = Query(..., min_length=1, max_length=500),
    limit: int = Query(12, ge=1, le=50),
    kind: str | None = Query(None),
    since: str | None = Query(None, description="Modified on or after this date (YYYY-MM-DD)."),
    until: str | None = Query(None, description="Modified on or before this date (YYYY-MM-DD)."),
    min_size: str | None = Query(None, description="Minimum file size, e.g. 10MB."),
    max_size: str | None = Query(None, description="Maximum file size, e.g. 10MB."),
    folder: str | None = Query(None, description="Only files inside this folder prefix."),
) -> JSONResponse:
    if kind not in {None, "text", "pdf", "image"}:
        raise HTTPException(status_code=400, detail="kind must be text, pdf, or image")
    started = time.perf_counter()
    try:
        results = run_search(
            store,
            embedder,
            q,
            limit=limit,
            kind=kind,
            since=since,
            until=until,
            min_size=min_size,
            max_size=max_size,
            folder=folder,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IndexMismatch as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Search is unavailable: {exc}") from exc
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    return JSONResponse(
        {
            "query": q,
            "count": len(results),
            "took_ms": elapsed,
            "stats": store.stats(),
            "results": [result.to_dict() for result in results],
        }
    )


@app.get("/thumb")
def thumb(path: str) -> FileResponse:
    row = store.conn.execute("SELECT path, kind FROM files WHERE path = ?", (path,)).fetchone()
    if row is None or row["kind"] != "image":
        raise HTTPException(status_code=404, detail="not an indexed image")
    file = Path(row["path"])
    if not file.exists():
        raise HTTPException(status_code=404, detail="file missing")
    return FileResponse(file)
