"""Cross-encoder reranking with bge-reranker-base.

The bi-encoder used for retrieval embeds the query and each criterion
*separately*, so it never actually compares them -- it compares two summaries
of them. That is what makes searching a million vectors fast, and it is also
why the top-10 by cosine similarity is often not the best top-10.

A cross-encoder reads the query and the criterion *together* and scores the
pair directly. It is far too slow to run over a million criteria, but perfect
for reordering the ~50 the bi-encoder already shortlisted.

Used as: retrieve `candidate_k` with the vector index, rerank, keep top `k`.
Phase 4 measures the same questions with and without this step, so the
improvement is a recorded number rather than an article of faith.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Sequence

from ..config import settings
from .semantic import Hit


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def get_reranker():
    """Load the cross-encoder once per process."""
    from sentence_transformers import CrossEncoder

    cfg = settings()["reranker"]
    return CrossEncoder(cfg["model"], device=_device(), max_length=512)


def rerank(query: str, hits: Sequence[Hit], *, top_k: Optional[int] = None) -> List[Hit]:
    """Reorder ``hits`` by cross-encoder relevance to ``query``.

    The returned Hits carry the cross-encoder score in ``rerank_score`` and
    keep their original bi-encoder ``score``, so the two can be compared. The
    list is sorted by the reranker.
    """
    if not hits:
        return []

    model = get_reranker()
    pairs = [(query, hit.criterion_text) for hit in hits]
    scores = model.predict(pairs, show_progress_bar=False)

    for hit, score in zip(hits, scores):
        hit.rerank_score = float(score)

    ranked = sorted(hits, key=lambda h: h.rerank_score, reverse=True)
    return ranked[:top_k] if top_k else ranked


def describe() -> str:
    cfg = settings()["reranker"]
    return f"{cfg['model']} on {_device()}"
