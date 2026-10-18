# Changelog

## 0.2.1 - 2026-10-18

- Cap search queries at 500 characters.
- Clean lint findings and make the lazy image load explicit.
- No behavior change to ranking.

## 0.2.0 - 2026-10-18

- Qwen3-Embedding 0.6B replaces MiniLM for text and PDF retrieval.
- SigLIP 2 base replaces CLIP for image retrieval.
- EmbeddingGemma 2 is available with `FIND_EMBEDDER=gemma`.
- Rebuilt the web interface with filters, keyboard access, loading states, empty states, and dark mode.
- Added intent-aware score fusion for text and image results.
- Added background model warmup and index model metadata.
- Added an embedding benchmark and evaluation notes.

## 0.1.0 - 2026-10-18

First public release.

- Initial working version.
- CLI and local web interface.
- Test suite and GitHub Actions workflow.
