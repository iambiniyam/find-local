# find-local

**Search your computer by describing it.**

Point `find-local` at your folders. Then ask for *“the invoice from March”*, *“a dog on a beach”*, or *“that error screenshot”* — and get ranked files with snippets and thumbnails. Everything runs locally; nothing is uploaded.

> Status: early MVP. Text and PDF search use multilingual embeddings; images use CLIP. No OCR inside images yet.

![find-local web UI](docs/demo.png)

## Why

Spotlight and Windows Search match filenames. They do not understand *what is inside* your photos, PDFs, or documents. Cloud tools that do understand require uploading your life to someone else's server.

`find-local` is:
- **Local** — embeddings and the index stay on disk.
- **Cross-format** — text files, PDFs, and images in one query.
- **Incremental** — re-index only what changed.
- **Agent-friendly** — ships an MCP server so coding agents can search your files.

## Quickstart

```bash
git clone https://github.com/iambiniyam/find-local.git
cd find-local
uv venv && source .venv/bin/activate
uv pip install -e ".[models]"

# optional in mainland China: download models from the mirror
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1   # if the hub download stalls

find-local index ~/Documents ~/Pictures
find-local search "invoice from March"
find-local serve        # http://127.0.0.1:8790
```

The first index downloads two models (~1.1 GB total) and embeds every file. Later runs only touch changed files.

## Try the bundled demo

The repo ships a tiny corpus in `examples/` (three notes, a PDF, a leaf photo, a face photo):

```bash
find-local index examples
find-local search "invoice from March"
find-local search "a leaf with orange spots"
find-local serve
```

## Models

| Purpose | Model | Dim | Size |
|---|---|---|---|
| Text & PDF | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | ~470 MB |
| Images | `sentence-transformers/clip-ViT-B-32` | 512 | ~600 MB |

Override with `FIND_TEXT_MODEL` / `FIND_IMAGE_MODEL`. Use `--no-images` for a text-only index.

## CLI

```bash
# index one or more folders
find-local index ~/Documents ~/Downloads ~/Pictures

# search
find-local search "offline privacy notes"
find-local search "beach" --kind image
find-local search "invoice" --limit 5 --json

# diagnostics
find-local doctor
```

## Web UI

`find-local serve` opens a single search box. Results show the file path, a snippet, the score, and an image thumbnail when relevant.

## MCP server

Give any MCP-compatible agent access to your local index:

```json
{
  "mcpServers": {
    "find-local": {
      "command": "find-local",
      "args": ["mcp"]
    }
  }
}
```

Tools exposed: `search_files(query, limit)`, `index_folder(path)`, `index_stats()`.

## How it works

```
folder ─► walk & classify ─► text/PDF chunks ─► multilingual embeddings ─┐
                        └──► images ──────────► CLIP embeddings ─────────┤
                                                                          ▼
                                                                 SQLite (WAL)
                                                                          │
query ─► text embedding ──────────────────────────────────────────────────┤
      └► CLIP text embedding ─────────────────────────────────────────────┘
                                                                          ▼
                                                    cosine search → merged results
```

- **Chunking** keeps long PDFs searchable; results show the best-matching chunk.
- **Two vector spaces** (text and image) are searched separately and merged by score.
- **SQLite** stores vectors as float32 blobs; no external vector database.

## Limits

- **No OCR inside images** — CLIP matches visual semantics, not text in a screenshot. OCR is on the roadmap.
- **Semantic, not exact** — for precise symbol search (function names, IDs), pair it with ripgrep.
- **First index is slow** on large folders; embeddings run on CPU/MPS.
- **Languages** — the text model covers 50+ languages; quality varies.
- **Not a file watcher yet** — run `index` again to pick up changes (incremental).

## Privacy

- No network calls after the models are downloaded.
- The index is a local SQLite file at `~/.find-local/index.db`.
- The web UI binds to `127.0.0.1` by default.
- Thumbnails are served only for paths already in your index.

## Roadmap

- [ ] OCR text inside images and screenshots
- [ ] File watcher for continuous indexing
- [ ] Hybrid BM25 + vector search
- [ ] Metadata filters (date, type, size, folder)
- [ ] On-device reranker for top-50 results
- [ ] Encrypted index option
- [ ] Benchmark: recall@10 on a personal-file corpus

## License

MIT.
