# Embedding evaluation

This is a smoke benchmark for the bundled example corpus. It is small, hand-labelled, and intended to catch regressions, not to rank models in general.

## Corpus

`examples/` contains:

- three Markdown notes: expenses, privacy, cake recipe
- one PDF: handbook
- two images: a leaf photo and a face photo

## Queries

| Query | Expected file |
|---|---|
| invoice from March | expenses-march.md |
| offline privacy notes | offline-privacy.md |
| a leaf with orange spots | leaf.jpg |
| portrait of a person | face.jpg |
| chocolate cake recipe | cake-recipe.md |
| semantic search without a cloud | handbook.pdf |
| march expenses | expenses-march.md |
| local embeddings | offline-privacy.md |
| someone's face | face.jpg |
| baking instructions | cake-recipe.md |

## Method

1. Build a fresh index with each embedding stack.
2. Run each query with `limit=5`.
3. Record the rank of the expected file.
4. Report top-1 accuracy, top-3 accuracy, mean reciprocal rank, and average query time.

Run it with:

```bash
python benchmarks/bench_embeddings.py --embedder auto
python benchmarks/bench_embeddings.py --embedder gemma
python benchmarks/bench_embeddings.py --embedder legacy
```

## Results

| Stack | Top-1 | Top-3 | MRR | Average query |
|---|---|---|---|---|
| Qwen3-Embedding 0.6B plus SigLIP 2 | 0.90 | 1.00 | 0.95 | 343 ms |
| EmbeddingGemma 2 | 0.80 | 1.00 | 0.90 | 1559 ms |
| MiniLM plus CLIP | 0.90 | 1.00 | 0.95 | 1229 ms |

Average includes the first query, which loads the model. Warm queries are much faster: the Qwen text model measured about 0.06 s per query on an M-series Mac, and SigLIP about 0.3 to 0.5 s per query.

## Notes

- The corpus is tiny. One query changes top-1 by 10 points.
- The queries are English and visual-heavy.
- The legacy stack ties on top-1 and MRR here, but it runs two older models and is slower to warm.
- Qwen3-Embedding and SigLIP 2 use newer, stronger models and leave room for a reranker.
- EmbeddingGemma 2 is a single unified model. It is simpler to deploy and multilingual, but it lost two queries in this set.

## Failure cases

Qwen plus SigLIP placed "local embeddings" at rank 2, behind the handbook. EmbeddingGemma placed the same query at rank 2 and "someone's face" at rank 3, behind a text note.

These are the cases a reranker or a hybrid lexical score should fix. They are listed here so the next change can be measured against them.
