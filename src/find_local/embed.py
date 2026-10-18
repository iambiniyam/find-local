"""Embedding backends: deterministic hash (tests) and sentence-transformers (real)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from .config import IMAGE_MODEL, TEXT_MODEL


class Embedder(Protocol):
    dim: int
    image_dim: int
    can_images: bool
    name: str

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...
    def encode_image_query(self, query: str) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashEmbedder:
    """Deterministic bag-of-tokens embedding for tests and key-less demos."""

    dim = 256
    image_dim = 256
    can_images = True
    name = "hash"

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    def _text_vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            vector[int.from_bytes(digest, "big") % self.dim] += 1.0
        return vector

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._text_vector(text) for text in texts]))

    def _image_vector(self, image: Image.Image) -> np.ndarray:
        array = np.asarray(image.convert("RGB").resize((32, 32)), dtype=np.float32) / 255.0
        hist, _ = np.histogramdd(
            array.reshape(-1, 3), bins=(4, 4, 4), range=((0, 1), (0, 1), (0, 1))
        )
        flat = hist.reshape(-1)
        vector = np.zeros(self.image_dim, dtype=np.float32)
        vector[: min(len(flat), self.image_dim)] = flat[: self.image_dim]
        return vector

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.image_dim), dtype=np.float32)
        return _normalize(np.stack([self._image_vector(image) for image in images]))

    def encode_image_query(self, query: str) -> np.ndarray:
        vector = self._text_vector(query)
        return vector / (np.linalg.norm(vector) or 1.0)


class SentenceEmbedder:
    """sentence-transformers text model + CLIP image model, loaded lazily."""

    def __init__(
        self,
        text_model: str = TEXT_MODEL,
        image_model: str = IMAGE_MODEL,
        with_images: bool = True,
    ):
        self.text_model_name = text_model
        self.image_model_name = image_model
        self.with_images = with_images
        self.name = f"{text_model} + {image_model if with_images else 'text-only'}"
        self._text = None
        self._image = None
        self.dim = 384
        self.image_dim = 512
        self.can_images = with_images

    @staticmethod
    def _load(model_name: str):
        import os

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        try:
            import transformers

            transformers.utils.logging.set_verbosity_error()
            transformers.utils.logging.disable_progress_bar()
        except Exception:  # noqa: BLE001 - logging setup is best-effort
            pass
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install the extra: "
                "uv pip install 'find-local[models]'"
            ) from exc
        return SentenceTransformer(model_name)

    @property
    def text(self):
        if self._text is None:
            self._text = self._load(self.text_model_name)
            dimension = getattr(self._text, "get_embedding_dimension", None)
            self.dim = int(dimension() if dimension else self._text.get_sentence_embedding_dimension())
        return self._text

    @property
    def image(self):
        if not self.with_images:
            raise RuntimeError("image embeddings are disabled")
        if self._image is None:
            self._image = self._load(self.image_model_name)
            dimension = getattr(self._image, "get_embedding_dimension", None)
            self.image_dim = int(dimension() if dimension else self._image.get_sentence_embedding_dimension())
        return self._image

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(
            self.text.encode(list(texts), normalize_embeddings=True, batch_size=32),
            dtype=np.float32,
        )

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.image_dim), dtype=np.float32)
        return np.asarray(
            self.image.encode(list(images), normalize_embeddings=True, batch_size=8),
            dtype=np.float32,
        )

    def encode_image_query(self, query: str) -> np.ndarray:
        return np.asarray(
            self.image.encode([query], normalize_embeddings=True)[0], dtype=np.float32
        )


def build_embedder(kind: str = "auto", with_images: bool = True) -> Embedder:
    if kind == "hash":
        return HashEmbedder()
    return SentenceEmbedder(with_images=with_images)
