
"""
SentenceTransformers embedding backend (ONLY).

This module provides a single backend for embedding both:
- corpus chunks (for indexing)
- user queries (for retrieval)

Design goals:
- production-ready (clear errors, batching, type hints)
- consistent output (np.float32, optional L2 normalization)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

import numpy as np


class EmbeddingBackend(Protocol):
    """Minimal interface used by embedding + search code."""
    name: str
    dim: int

    def embed_texts(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        """Return embeddings with shape (n, dim)."""

    def embed_query(self, text: str) -> np.ndarray:
        """Return a single embedding with shape (dim,)."""


def _clean_texts(texts: Sequence[str]) -> List[str]:
    """
    Ensure no empty/None strings are passed to the model.
    Empty strings can produce unstable outputs in some models.
    """
    out: List[str] = []
    for t in texts:
        t = (t or "").strip()
        out.append(t if t else " ")
    return out


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization (recommended for cosine similarity)."""
    if x.ndim == 1:
        denom = float(np.linalg.norm(x) + eps)
        return x / denom
    denom = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / denom


@dataclass
class SentenceTransformersBackend:
    """
    Local embeddings via sentence-transformers.

    Default model is a strong baseline for academic paper search:
    sentence-transformers/all-MiniLM-L6-v2
    """
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True
    device: Optional[str] = None  # "cpu" or "cuda"

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Add it to requirements.txt and run: pip install -r requirements.txt"
            ) from exc

        kwargs = {}
        if self.device:
            kwargs["device"] = self.device

        self._model = SentenceTransformer(self.model_name, **kwargs)
        self.name = f"st::{self.model_name}"

        # Determine embedding dimension robustly
        try:
            self.dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            test = self._model.encode(["test"], convert_to_numpy=True)
            self.dim = int(test.shape[1])

    def embed_texts(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        texts_clean = _clean_texts(texts)

        vecs = self._model.encode(
            texts_clean,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalize ourselves to be explicit
        ).astype(np.float32)

        if self.normalize:
            vecs = _l2_normalize(vecs)

        return vecs

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text], batch_size=1)[0]


def get_backend(
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
    device: Optional[str] = None,
) -> SentenceTransformersBackend:
    """
    Factory for the single supported backend.

    Returns
    -------
    SentenceTransformersBackend
    """
    return SentenceTransformersBackend(model_name=model_name, normalize=normalize, device=device)

