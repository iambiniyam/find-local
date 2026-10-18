"""Optional OCR for screenshots and photos."""

from __future__ import annotations

from pathlib import Path


def ocr_image(path: Path) -> str:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "OCR is not installed. Install with: uv pip install 'find-local[ocr]'"
        ) from exc
    engine = RapidOCR()
    result, _elapsed = engine(str(path))
    if not result:
        return ""
    return "\n".join(str(item[1]) for item in result if str(item[1]).strip())
