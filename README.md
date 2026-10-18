# find-local

[![ci](https://github.com/iambiniyam/find-local/actions/workflows/ci.yml/badge.svg)](https://github.com/iambiniyam/find-local/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Search your files by describing what is in them.

Point find-local at one or more folders. Ask for "the invoice from March", "a dog on a beach", or "that error screenshot". It returns ranked files with snippets and image thumbnails. The index stays on your machine.

![find-local web UI](docs/demo.png)

Status: early MVP. Text and PDF search use multilingual embeddings. Images use CLIP. It does not read text inside images yet.

## Why

Spotlight and Windows Search mostly match file names. They do not know what is inside a PDF or a photo. Cloud search tools can look inside files, but they upload everything first.

find-local builds a local index instead. It handles text files, PDFs, and photos in one query, and it ships an MCP server so coding agents can search the same index.

## Quickstart

```bash
git clone https://github.com/iambiniyam/find-local.git
cd find-local
uv venv && source .venv/bin/activate
uv pip install -e ".[models]"

# optional in mainland China
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

find-local index ~/Documents ~/Pictures
find-local search "invoice from March"
find-local serve
```

The first index downloads two models, about 1.1 GB in total, and embeds every file. Later runs only process files that changed.

## Bundled demo

The repo includes a small corpus in `examples/`: three notes, a PDF, a leaf photo, and a face photo.

```bash
find-local index examples
find-local search "invoice from March"
find-local search "a leaf with orange spots"
find-local serve
```

## Models

| Purpose | Model | Dimensions | Size |
|---|---|---|---|
| Text and PDF | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 384 | about 470 MB |
| Images | sentence-transformers/clip-ViT-B-32 | 512 | about 600 MB |

Override them with `FIND_TEXT_MODEL` and `FIND_IMAGE_MODEL`. Use `--no-images` for a text-only index.

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

`find-local serve` opens a single search box. Results show the file path, the best matching snippet, the score, and a thumbnail for images.

## MCP server

Any MCP client can use the index:

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

The server exposes `search_files`, `index_folder`, and `index_stats`.

## How it works

```
folder
  |
  +--> text and PDF chunks --> multilingual embeddings --+
  |                                                       |
  +--> images -------------> CLIP embeddings ------------+
                                                          |
                                                          v
                                                    SQLite (WAL)
                                                          |
query --> text embedding ---------------------------------+
      --> CLIP text embedding ----------------------------+
                                                          |
                                                          v
                                              cosine search and merge
```

Text and image vectors live in different spaces. They are searched separately and merged by score. Chunks keep long PDFs searchable, and results show the best matching chunk.

## Limits

- It does not read text inside images. CLIP matches visual meaning, not words in a screenshot. OCR is on the roadmap.
- Search is semantic, not exact. For function names and IDs, use ripgrep alongside it.
- The first index is slow on large folders. Embeddings run on CPU or MPS.
- The text model covers 50 or more languages, with varying quality.
- There is no file watcher yet. Run `index` again to pick up changes.

## Privacy

- No network calls after the models are downloaded.
- The index is a local SQLite file at `~/.find-local/index.db`.
- The web UI binds to 127.0.0.1 by default.
- Thumbnails are served only for paths that are already in the index.

## Tests

```bash
pytest -q
```

The suite has 5 tests and uses a deterministic hash embedder, so it does not download models.

## Roadmap

- OCR for text inside images and screenshots
- a file watcher for continuous indexing
- hybrid BM25 and vector search
- metadata filters for date, type, size, and folder
- an on-device reranker for the top 50 results
- an encrypted index option
- a recall@10 benchmark on a personal file corpus

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Related tools

- [shamba](https://github.com/iambiniyam/shamba): offline plant health scanner for gardeners and farmers.
- [redact](https://github.com/iambiniyam/redact): offline PII redaction for PDFs and photos.
- [snap2sheet](https://github.com/iambiniyam/snap2sheet): receipt and invoice photos to CSV or Excel.

## License

MIT. See [THIRD_PARTY.md](THIRD_PARTY.md) for the model licenses.
