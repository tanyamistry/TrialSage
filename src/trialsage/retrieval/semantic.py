"""Semantic search over eligibility criteria using pgvector.

Two things distinguish this from a textbook vector search, and both matter for
correctness rather than speed:

**Polarity is carried through.** Every hit reports whether it came from an
inclusion or an exclusion criterion. "Excludes patients with autoimmune
disease" and "includes patients with autoimmune disease" embed almost
identically -- the vectors cannot tell them apart, so the *metadata* has to.
Callers can also filter to one polarity, which is what the hybrid path does.

**Search can be restricted to a candidate set of trials.** This is the
mechanism behind the hybrid route: run the SQL filter first, then search only
within those NCT IDs. It is both more accurate (no semantically-similar trials
from the wrong phase or country) and cheaper (a much smaller search space).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence

from pgvector.psycopg import register_vector

from ..config import settings
from ..db import connect
from ..embed.model import embed_query

CriterionFilter = Optional[Literal["inclusion", "exclusion", "unspecified"]]


def _tune_hnsw(conn) -> None:
    """Configure the HNSW scan so filtered searches return complete results.

    This is a correctness fix, not a performance tweak.

    By default pgvector walks the HNSW graph for `ef_search` (40) candidates
    and only THEN applies the query's WHERE clause. So a search filtered to
    inclusion criteria first collects the 40 globally-nearest chunks and then
    discards every one that is an exclusion -- frequently leaving zero rows,
    even though thousands of matching inclusion criteria exist. It fails
    silently: you get an empty result set, not an error.

    That would break both of the things this module exists to do: polarity
    filtering, and restricting a search to a candidate set of trials (the
    hybrid route).

    `iterative_scan = relaxed_order` (pgvector 0.8+) makes the scan keep
    pulling batches from the index until the LIMIT is satisfied. Rows can come
    back slightly out of order, so callers re-sort by score.
    """
    cfg = settings()["retrieval"]
    try:
        conn.execute(f"SET hnsw.ef_search = {int(cfg.get('hnsw_ef_search', 100))}")
        conn.execute(
            "SET hnsw.iterative_scan = "
            f"{cfg.get('hnsw_iterative_scan', 'relaxed_order')}"
        )
    except Exception:
        # Older pgvector, or the extension's settings are not registered yet.
        # Searching still works; filtered recall is just worse.
        pass


@dataclass
class Hit:
    """One retrieved eligibility criterion with its provenance."""

    chunk_id: int
    nct_id: str
    criterion_type: str
    criterion_text: str
    score: float               # bi-encoder cosine similarity, 1.0 = identical
    # Cross-encoder score, populated only when the reranker has run. Kept
    # alongside `score` rather than overwriting it so the two rankings can be
    # compared directly -- which is the whole point of the Phase 4 before/after.
    rerank_score: Optional[float] = None
    brief_title: Optional[str] = None
    overall_status: Optional[str] = None
    phase_display: Optional[str] = None

    def citation(self) -> str:
        return self.nct_id

    def as_context(self) -> str:
        """Render for the synthesizer prompt, polarity made explicit."""
        label = {"inclusion": "INCLUSION", "exclusion": "EXCLUSION"}.get(
            self.criterion_type, "UNSPECIFIED")
        return f"[{self.nct_id}] ({label}) {self.criterion_text}"


def search(
    query: str,
    *,
    k: Optional[int] = None,
    nct_ids: Optional[Sequence[str]] = None,
    criterion_type: CriterionFilter = None,
    min_score: float = 0.0,
) -> List[Hit]:
    """Return the ``k`` criteria most similar to ``query``.

    ``nct_ids`` restricts the search to a candidate set -- pass the output of
    the SQL filter to get the hybrid behaviour. Passing an **empty list** means
    "the structured filter matched nothing", and correctly returns no hits
    rather than silently searching the whole corpus.
    """
    cfg = settings()["retrieval"]
    k = k or cfg["top_k"]

    if nct_ids is not None and len(nct_ids) == 0:
        return []

    vector = embed_query(query)

    where = ["c.embedding IS NOT NULL"]
    params: List[object] = [vector]
    if nct_ids is not None:
        where.append("c.nct_id = ANY(%s)")
        params.append(list(nct_ids))
    if criterion_type is not None:
        where.append("c.criterion_type = %s")
        params.append(criterion_type)

    # `1 - (embedding <=> query)` converts pgvector's cosine *distance* into a
    # similarity, so bigger is better and the number is readable in the UI.
    sql = f"""
        SELECT c.chunk_id, c.nct_id, c.criterion_type, c.criterion_text,
               1 - (c.embedding <=> %s) AS score,
               t.brief_title, t.overall_status, t.phase_display
        FROM eligibility_chunks c
        JOIN trials t USING (nct_id)
        WHERE {' AND '.join(where)}
        ORDER BY c.embedding <=> %s
        LIMIT %s
    """
    params.append(vector)   # ORDER BY needs the vector again
    params.append(k)

    with connect(autocommit=True) as conn:
        register_vector(conn)
        _tune_hnsw(conn)
        rows = conn.execute(sql, params).fetchall()

    hits = [
        Hit(chunk_id=r[0], nct_id=r[1], criterion_type=r[2], criterion_text=r[3],
            score=float(r[4]), brief_title=r[5], overall_status=r[6], phase_display=r[7])
        for r in rows
    ]
    # relaxed_order can return rows slightly out of distance order, so sort here
    # rather than trusting the SQL ORDER BY.
    hits.sort(key=lambda h: h.score, reverse=True)
    return [h for h in hits if h.score >= min_score]


def search_trials(
    query: str,
    *,
    k: Optional[int] = None,
    nct_ids: Optional[Sequence[str]] = None,
    criterion_type: CriterionFilter = None,
    candidate_k: Optional[int] = None,
    rerank: bool = False,
) -> List[Hit]:
    """Like :func:`search` but deduplicated to the best hit per trial.

    Retrieving criteria means one trial with several matching criteria can fill
    the entire result set. When the question is "which trials...", that is the
    wrong granularity, so we over-fetch and keep each trial's strongest hit.

    With ``rerank=True`` the shortlist is reordered by a cross-encoder before
    the top ``k`` are taken. The over-fetch (``candidate_k``) is what gives the
    reranker something to improve on -- reranking exactly ``k`` candidates can
    only reorder them, never bring a better one into view.
    """
    cfg = settings()["retrieval"]
    k = k or cfg["top_k"]
    candidate_k = candidate_k or max(cfg["candidate_k"], k * 5)

    hits = search(query, k=candidate_k, nct_ids=nct_ids, criterion_type=criterion_type)

    best: dict[str, Hit] = {}
    for hit in hits:
        if hit.nct_id not in best or hit.score > best[hit.nct_id].score:
            best[hit.nct_id] = hit
    ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)

    if rerank and ranked:
        from .rerank import rerank as _rerank
        return _rerank(query, ranked, top_k=k)
    return ranked[:k]


def format_hits(hits: Iterable[Hit]) -> str:
    lines = []
    for i, hit in enumerate(hits, 1):
        title = (hit.brief_title or "")[:60]
        lines.append(f"{i:>2}. {hit.score:.3f}  [{hit.criterion_type:<11}] {hit.nct_id}  {title}")
        lines.append(f"        {hit.criterion_text[:150]}")
    return "\n".join(lines)
