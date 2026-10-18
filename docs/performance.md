# OCR indexing performance

Measured on an Apple Silicon Mac with 8 GB RAM.

Workload: three generated 700x240 receipt images with two lines of text, indexed with `--embedder hash --ocr --no-images`.

| Run | Wall time |
|---|---:|
| RapidOCR engine created per image | 2.68 s |
| One cached engine reused | 1.49 s |

The engine is now created once per process and reused. The measured wall time reduction is about 44% for this small batch. Larger batches should benefit more because model initialization is paid once.

This is a synthetic receipt workload, not a general OCR benchmark. Image size, text density, and hardware change the result.
