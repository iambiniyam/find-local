"""Local web search UI."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .config import db_path
from .embed import build_embedder
from .index import Store
from .search import search as run_search

DB = Path(os.environ.get("FIND_DB", str(db_path())))
store = Store(DB)
embedder = build_embedder(os.environ.get("FIND_EMBEDDER", "auto"))
app = FastAPI(title="find-local", version="0.1.0")

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>find-local: search by describing</title>
<style>
  :root { color-scheme: light dark; --ink:#131a16; --paper:#f6f4ea; --green:#1f7a4d; --line:#ddd7c6; }
  * { box-sizing:border-box; }
  body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; background:var(--paper); color:var(--ink); }
  header { max-width:820px; margin:0 auto; padding:34px 20px 8px; }
  h1 { font-size:2.1rem; margin:0 0 6px; letter-spacing:-.03em; }
  .sub { color:#5d6a61; margin:0 0 18px; }
  main { max-width:820px; margin:0 auto; padding:0 20px 48px; }
  .bar { display:flex; gap:8px; background:#fff; border:1px solid var(--line); border-radius:14px; padding:10px; box-shadow:0 6px 20px rgba(20,30,24,.05); }
  input[type=search] { flex:1; border:0; outline:0; font:inherit; padding:10px 12px; background:transparent; color:inherit; }
  button { background:var(--green); color:#fff; border:0; border-radius:9px; padding:10px 16px; font-weight:700; cursor:pointer; }
  .result { background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px 16px; margin-top:12px; display:flex; gap:14px; align-items:flex-start; }
  .result img { width:96px; height:96px; object-fit:cover; border-radius:8px; border:1px solid #eee7d8; }
  .path { font-weight:700; font-size:.95rem; word-break:break-all; }
  .snippet { color:#4f5c53; font-size:.9rem; margin-top:5px; line-height:1.4; }
  .meta { color:#7a857d; font-size:.78rem; margin-top:6px; }
  .empty { color:#6c766f; margin-top:18px; }
</style>
</head>
<body>
<header>
  <h1>find-local</h1>
  <p class="sub">Search your computer by describing it. Everything is local.</p>
</header>
<main>
  <div class="bar">
    <input id="q" type="search" placeholder="invoice from March, dog on a beach, error screenshot..." autofocus>
    <button id="go">Search</button>
  </div>
  <div id="out"></div>
</main>
<script>
const q = document.getElementById('q'), go = document.getElementById('go'), out = document.getElementById('out');
function esc(s) { const d = document.createElement('div'); d.textContent = s == null ? '' : String(s); return d.innerHTML; }
async function run() {
  const query = q.value.trim();
  if (!query) return;
  out.innerHTML = '<p class="empty">Searching locally...</p>';
  const res = await fetch('/api/search?q=' + encodeURIComponent(query) + '&limit=12');
  const data = await res.json();
  if (!data.results.length) { out.innerHTML = '<p class="empty">Nothing yet. Index a folder first: find-local index ~/Documents</p>'; return; }
  out.innerHTML = data.results.map(r => {
    const thumb = r.kind === 'image' ? `<img src="/thumb?path=${encodeURIComponent(r.path)}" alt="">` : '';
    return `<div class="result">${thumb}<div><div class="path">${esc(r.path)}</div>
      <div class="snippet">${esc(r.snippet)}</div>
      <div class="meta">${esc(r.kind)} · score ${Number(r.score).toFixed(3)}</div></div></div>`;
  }).join('');
}
go.addEventListener('click', run);
q.addEventListener('keydown', e => { if (e.key === 'Enter') run(); });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/healthz")
def health() -> dict[str, object]:
    return {"ok": True, "index": str(DB), "stats": store.stats()}


@app.get("/api/search")
def api_search(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    kind: str | None = Query(None),
) -> JSONResponse:
    results = run_search(store, embedder, q, limit=limit, kind=kind)
    return JSONResponse({"results": [result.to_dict() for result in results]})


@app.get("/thumb")
def thumb(path: str) -> FileResponse:
    row = store.conn.execute("SELECT path, kind FROM files WHERE path = ?", (path,)).fetchone()
    if row is None or row["kind"] != "image":
        raise HTTPException(status_code=404, detail="not an indexed image")
    file = Path(row["path"])
    if not file.exists():
        raise HTTPException(status_code=404, detail="file missing")
    return FileResponse(file)
