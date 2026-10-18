"""Embedding backends: EmbeddingGemma 2 by default, legacy models, and a test hash."""

from __future__ import annotations

import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from .config import (
    EMBEDDINGGEMMA_MODEL,
    IMAGE_MODEL,
    QWEN_EMBEDDING_MODEL,
    SIGLIP2_MODEL,
    TEXT_MODEL,
    embeddinggemma_dir,
    qwen_embedding_dir,
    siglip2_dir,
)


class Embedder(Protocol):
    dim: int
    image_dim: int
    can_images: bool
    image_ready: bool
    name: str

    def encode_query(self, query: str) -> np.ndarray: ...
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...
    def encode_image_query(self, query: str) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _configure_logging() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        import transformers

        transformers.utils.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
    except Exception:  # noqa: BLE001 - logging setup is best-effort
        pass


class HashEmbedder:
    """Deterministic bag-of-tokens embedding for tests and key-less demos."""

    dim = 256
    image_dim = 256
    can_images = True
    image_ready = True
    name = "hash"

    def warmup(self) -> None:
        return None

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    def _text_vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            vector[int.from_bytes(digest, "big") % self.dim] += 1.0
        return vector

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._text_vector(text) for text in texts]))

    def encode_query(self, query: str) -> np.ndarray:
        vector = self._text_vector(query)
        return vector / (np.linalg.norm(vector) or 1.0)

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
        return self.encode_query(query)


class EmbeddingGemmaEmbedder:
    """Google EmbeddingGemma 2: text and images in one 768-dimensional space.

    The model is multimodal, multilingual, and small enough for a laptop. It
    uses task prompts for text and no prompt for images.
    """

    dim = 768
    image_dim = 768
    can_images = True
    name = "embeddinggemma-2"

    def __init__(
        self,
        model_name: str = EMBEDDINGGEMMA_MODEL,
        model_dir: Path | str | None = None,
        with_images: bool = True,
    ):
        self.model_name = model_name
        self.model_dir = Path(model_dir) if model_dir else embeddinggemma_dir()
        self.with_images = with_images
        self.can_images = with_images
        self.dim = 768
        self.image_dim = 768
        self._model = None
        self._lock = threading.Lock()

    def warmup(self) -> None:
        self.model.encode("warmup", prompt_name="SearchQuery")

    @property
    def image_ready(self) -> bool:
        return self._model is not None

    def _load(self):
        _configure_logging()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install the extra: "
                "uv pip install 'find-local[models]'"
            ) from exc

        local = self.model_dir.exists()
        target = str(self.model_dir) if local else self.model_name
        config_kwargs = {"audio_config": None}
        if not self.with_images:
            config_kwargs["vision_config"] = None
        return SentenceTransformer(
            target,
            config_kwargs=config_kwargs,
            local_files_only=local,
        )

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load()
                    dimension = getattr(self._model, "get_embedding_dimension", None)
                    self.dim = int(dimension() if dimension else 768)
                    self.image_dim = self.dim
        return self._model

    def encode_query(self, query: str) -> np.ndarray:
        vector = self.model.encode(
            query,
            prompt_name="SearchQuery",
            normalize_embeddings=True,
        )
        return np.asarray(vector, dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            prompt_name="Document",
            normalize_embeddings=True,
            batch_size=8,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.image_dim), dtype=np.float32)
        vectors = [
            self.model.encode({"image": [image.convert("RGB")]}, normalize_embeddings=True)
            for image in images
        ]
        return np.asarray(np.stack(vectors), dtype=np.float32)

    def encode_image_query(self, query: str) -> np.ndarray:
        return self.encode_query(query)


class QwenSiglipEmbedder:
    """Qwen3-Embedding for text and SigLIP 2 for images.

    This pair won the local benchmark: stronger text retrieval than the legacy
    MiniLM model and stronger image retrieval than CLIP.
    """

    dim = 1024
    image_dim = 768
    can_images = True
    name = "qwen3-embedding-0.6b + siglip2-base"

    def __init__(
        self,
        text_model: str = QWEN_EMBEDDING_MODEL,
        image_model: str = SIGLIP2_MODEL,
        text_dir: Path | str | None = None,
        image_dir: Path | str | None = None,
        with_images: bool = True,
    ):
        self.text_model_name = text_model
        self.image_model_name = image_model
        self.text_dir = Path(text_dir) if text_dir else qwen_embedding_dir()
        self.image_dir = Path(image_dir) if image_dir else siglip2_dir()
        self.with_images = with_images
        self.can_images = with_images
        self.dim = 1024
        self.image_dim = 768
        self._text = None
        self._image = None
        self._processor = None
        self._text_lock = threading.Lock()
        self._image_lock = threading.Lock()

    def warmup(self) -> None:
        self.text.encode(["warmup"], prompt_name="query")
        if self.with_images:
            self.encode_images([Image.new("RGB", (64, 64), "white")])

    @property
    def image_ready(self) -> bool:
        return self._image is not None

    @staticmethod
    def _target(local: Path, model_name: str) -> tuple[str, bool]:
        return (str(local), True) if local.exists() else (model_name, False)

    def _load_text(self):
        _configure_logging()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install the extra: "
                "uv pip install 'find-local[models]'"
            ) from exc
        target, local_only = self._target(self.text_dir, self.text_model_name)
        return SentenceTransformer(target, local_files_only=local_only)

    def _load_image(self):
        _configure_logging()
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed. Install the extra: "
                "uv pip install 'find-local[models]'"
            ) from exc
        target, local_only = self._target(self.image_dir, self.image_model_name)
        model = AutoModel.from_pretrained(target, local_files_only=local_only).eval()
        processor = AutoProcessor.from_pretrained(target, local_files_only=local_only)
        return model, processor

    @property
    def text(self):
        if self._text is None:
            with self._text_lock:
                if self._text is None:
                    self._text = self._load_text()
                    dimension = getattr(self._text, "get_embedding_dimension", None)
                    self.dim = int(dimension() if dimension else 1024)
        return self._text

    @property
    def image(self):
        if not self.with_images:
            raise RuntimeError("image embeddings are disabled")
        if self._image is None:
            with self._image_lock:
                if self._image is None:
                    self._image, self._processor = self._load_image()
                    self.image_dim = int(self._image.config.vision_config.hidden_size)
        return self._image

    @staticmethod
    def _pool(output):
        return output.pooler_output if hasattr(output, "pooler_output") else output

    def encode_query(self, query: str) -> np.ndarray:
        vector = self.text.encode(
            [query], prompt_name="query", normalize_embeddings=True
        )[0]
        return np.asarray(vector, dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(
            self.text.encode(list(texts), normalize_embeddings=True, batch_size=8),
            dtype=np.float32,
        )

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.image_dim), dtype=np.float32)
        import torch
        import torch.nn.functional as F

        self.image
        inputs = self._processor(
            images=[image.convert("RGB") for image in images], return_tensors="pt"
        )
        with torch.no_grad():
            vectors = self._pool(self.image.get_image_features(**inputs))
        return F.normalize(vectors, dim=-1).cpu().numpy().astype(np.float32)

    def encode_image_query(self, query: str) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self.image
        inputs = self._processor(
            text=[query], padding="max_length", truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            vector = self._pool(self.image.get_text_features(**inputs))
        return F.normalize(vector, dim=-1)[0].cpu().numpy().astype(np.float32)


class SentenceEmbedder:
    """Legacy text model plus CLIP image model, loaded lazily."""

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
        self._text_lock = threading.Lock()
        self._image_lock = threading.Lock()

    def warmup(self) -> None:
        self.text.encode(["warmup"])

    @property
    def image_ready(self) -> bool:
        return self._image is not None

    @staticmethod
    def _load(model_name: str):
        _configure_logging()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install the extra: "
                "uv pip install 'find-local[models]'"
            ) from exc
        kwargs = {}
        if Path(model_name).exists():
            kwargs["local_files_only"] = True
        return SentenceTransformer(model_name, **kwargs)

    @property
    def text(self):
        if self._text is None:
            with self._text_lock:
                if self._text is None:
                    self._text = self._load(self.text_model_name)
                    dimension = getattr(self._text, "get_embedding_dimension", None)
                    self.dim = int(dimension() if dimension else 384)
        return self._text

    @property
    def image(self):
        if not self.with_images:
            raise RuntimeError("image embeddings are disabled")
        if self._image is None:
            with self._image_lock:
                if self._image is None:
                    self._image = self._load(self.image_model_name)
                    dimension = getattr(self._image, "get_embedding_dimension", None)
                    self.image_dim = int(dimension() if dimension else 512)
        return self._image

    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray(self.text.encode([query], normalize_embeddings=True)[0], dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
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
    kind = (kind or "auto").lower()
    if kind == "hash":
        return HashEmbedder()
    if kind == "legacy":
        return SentenceEmbedder(with_images=with_images)
    if kind in {"gemma", "embeddinggemma"}:
        return EmbeddingGemmaEmbedder(with_images=with_images)
    return QwenSiglipEmbedder(with_images=with_images)
