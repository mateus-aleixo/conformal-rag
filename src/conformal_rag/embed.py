"""Embedders behind one protocol.

- HashEmbedder: deterministic, dependency-free. Used by tests and CI so the whole
  suite runs with no model downloads. Not semantically meaningful — but stable, so
  retrieval mechanics (fusion, ranking, storage round-trips) are fully testable.
- BgeEmbedder: sentence-transformers bge-small-en-v1.5, the real thing. Optional
  extra: `pip install -e ".[embed]"`.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashEmbedder:
    """Bag-of-hashed-tokens, L2-normalised. Deterministic across runs/platforms."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in _TOKEN.findall(text.lower()):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=4).digest(), "big")
                out[row, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.divide(out, norms, out=out, where=norms > 0)
        return out


class BgeEmbedder:
    """Real semantic embeddings. Import cost paid lazily and only when chosen."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer  # optional extra

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(list(texts), normalize_embeddings=True), dtype=np.float32
        )


def get_embedder(name: str = "hash") -> Embedder:
    if name == "hash":
        return HashEmbedder()
    if name == "bge":
        return BgeEmbedder()
    raise ValueError(f"unknown embedder: {name!r}")
