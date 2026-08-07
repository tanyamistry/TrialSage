"""Local sentence-transformer embeddings (bge-small-en-v1.5, 384-dim).

Runs on the Apple GPU via Metal (MPS) when available, otherwise CPU. Nothing
here calls a paid API.

The one non-obvious detail is the **query prefix**. BGE models are trained
asymmetrically: documents are embedded as-is, but a search query must be
prefixed with "Represent this sentence for searching relevant passages: ".
Getting this backwards -- or applying the prefix to both sides -- measurably
degrades recall, and it fails silently, so the two paths are separate functions
here rather than one function with a flag that is easy to forget.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence

import numpy as np

from ..config import settings


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def get_model():
    """Load the embedding model once per process (it is ~130MB)."""
    from sentence_transformers import SentenceTransformer

    cfg = settings()["embedding"]
    model = SentenceTransformer(cfg["model"], device=_device())
    return model


def embed_documents(texts: Sequence[str], *, batch_size: int | None = None,
                    show_progress: bool = False) -> np.ndarray:
    """Embed passages for storage. No instruction prefix -- this is the doc side."""
    cfg = settings()["embedding"]
    model = get_model()
    return model.encode(
        list(texts),
        batch_size=batch_size or cfg["batch_size"],
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # lets cosine distance work as a dot product
        convert_to_numpy=True,
    )


def embed_query(text: str) -> np.ndarray:
    """Embed a search query. Applies the BGE instruction prefix."""
    cfg = settings()["embedding"]
    model = get_model()
    return model.encode(
        cfg["query_prefix"] + text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def embed_queries(texts: Sequence[str]) -> np.ndarray:
    cfg = settings()["embedding"]
    model = get_model()
    return model.encode(
        [cfg["query_prefix"] + t for t in texts],
        batch_size=cfg["batch_size"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def describe() -> str:
    cfg = settings()["embedding"]
    return f"{cfg['model']} ({cfg['dim']}-dim) on {_device()}"
