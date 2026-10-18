"""Optional OCR for screenshots and photos."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "OCR is not installed. Install with: uv pip install 'find-local[ocr]'"
        ) from exc
    return RapidOCR()


def ocr_image(path: Path) -> str:
    result, _elapsed = _engine()(str(path))
    if not result:
        return ""
    return "\n".join(str(item[1]) for item in result if str(item[1]).strip())
