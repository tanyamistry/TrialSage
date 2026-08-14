"""The hybrid route: structured filter first, then semantic search inside it.

Ordering is the whole point. Running the SQL filter first and restricting the
vector search to the resulting NCT IDs is both more accurate and cheaper than
the reverse:

* **More accurate.** Searching 966k criteria for "history of autoimmune
  disease" returns the most semantically similar criteria in the entire
  registry -- overwhelmingly from trials in the wrong phase, status or country.
  Filtering afterwards throws most of them away and often leaves nothing.
* **Cheaper.** The scoped search compares against a few thousand vectors rather
  than a million.

The opposite ordering (search then filter) is exactly the failure mode that
makes naive RAG bad at this class of question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..config import settings
from .semantic import Hit, search_trials
from .sql_agent import SQLResult, generate_sql

# Asks for the identifier column specifically: the structured half of a hybrid
# question is a *filter*, not an answer, so all we need back is which trials
# qualify.
#
# The "ignore eligibility concepts" instruction is the second line of defence
# behind the router's semantic/structured split. Without it the agent tries to
# express a medical concept as a column filter -- for the Massachusetts
# question it produced `AND 'autoimmune disease' = ANY(conditions)`, which
# matches nothing and turns a good question into a false "no trials found".
# The eligibility half is handled by the vector search, not by SQL.
_FILTER_TEMPLATE = (
    "List the nct_id of every trial matching these criteria: {filters}. "
    "Return only the nct_id column. "
    "Filter ONLY on phase, status, therapeutic area, location, dates, sponsor, "
    "enrolment and age. IGNORE any mention of medical history, prior treatment, "
    "comorbidities or other eligibility-criteria concepts -- those are handled "
    "separately and must not appear in the WHERE clause."
)


@dataclass
class HybridResult:
    question: str
    semantic_query: str
    structured_query: str = ""
    sql: Optional[str] = None
    candidate_ids: List[str] = field(default_factory=list)
    hits: List[Hit] = field(default_factory=list)
    sql_result: Optional[SQLResult] = field(default=None, repr=False)
    error: Optional[str] = None
    latency_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def filtered_out(self) -> bool:
        """True when the structured filter matched nothing.

        Distinct from "found no relevant criteria": the honest answer is
        "no trials match those filters", not "nothing mentions that concept".
        """
        return self.ok and not self.candidate_ids


def retrieve_hybrid(
    question: str,
    *,
    semantic_query: Optional[str] = None,
    structured_query: Optional[str] = None,
    k: Optional[int] = None,
    max_candidates: Optional[int] = None,
) -> HybridResult:
    """Run the structured filter, then search only within the trials it returned.

    ``semantic_query`` and ``structured_query`` are the two halves of the
    question as separated by the router. Passing the whole question for both is
    supported but weaker -- see the note on ``_FILTER_TEMPLATE``.
    """
    import time

    started = time.perf_counter()
    cfg = settings()["retrieval"]
    k = k or cfg["top_k"]
    max_candidates = max_candidates or cfg.get("hybrid_max_candidates", 2000)

    sem_query = (semantic_query or question).strip()
    filters = (structured_query or question).strip()
    result = HybridResult(question=question, semantic_query=sem_query,
                          structured_query=filters)

    # 1. Structured filter -> candidate NCT IDs.
    sql_result = generate_sql(
        _FILTER_TEMPLATE.format(filters=filters),
        row_limit=max_candidates,
    )
    result.sql_result = sql_result
    result.sql = sql_result.sql

    if not sql_result.ok:
        result.error = f"structured filter failed: {sql_result.error}"
        result.latency_s = time.perf_counter() - started
        return result

    result.candidate_ids = sql_result.nct_ids()
    if not result.candidate_ids:
        # Empty is a real, reportable answer -- not a reason to widen the search.
        result.latency_s = time.perf_counter() - started
        return result

    # 2. Semantic search restricted to those trials.
    result.hits = search_trials(sem_query, k=k, nct_ids=result.candidate_ids)
    result.latency_s = time.perf_counter() - started
    return result


def explain(result: HybridResult) -> str:
    lines = [
        f"hybrid: {result.question!r}",
        f"  structured filter -> {len(result.candidate_ids)} candidate trials",
    ]
    if result.sql:
        lines.append(f"  sql: {result.sql}")
    lines.append(f"  structured half: {result.structured_query!r}")
    lines.append(f"  semantic half:   {result.semantic_query!r}")
    lines.append(f"  -> {len(result.hits)} hits ({result.latency_s:.1f}s)")
    if result.error:
        lines.append(f"  ERROR: {result.error}")
    return "\n".join(lines)
