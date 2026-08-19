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

import re
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
    "Find trials matching ONLY this description: {filters}\n"
    "Return the nct_id column.\n"
    "\n"
    "CRITICAL: include a WHERE condition ONLY for something stated explicitly "
    "above. Do not add a status, phase, area, country, state, date or any other "
    "condition that is not written there. A filter that is too narrow returns "
    "nothing and is much worse than one that is too broad.\n"
    "Ignore any mention of medical history, prior treatment or other "
    "eligibility-criteria wording -- that is handled separately and must not "
    "appear in the WHERE clause.\n"
    "\n"
    "Example: for \"phase 1 trials\" the whole query is\n"
    "  SELECT nct_id FROM v_trials WHERE phases @> ARRAY['PHASE1']\n"
    "-- nothing else, no status, no area, no location."
)


# Eligibility verbs left dangling once the medical concept is removed
# ("...trials exclude patients with" -> "...trials").
_DANGLING = re.compile(
    r"\b(that\s+|which\s+|who\s+)?(exclude|excludes|excluding|allow|allows|"
    r"allowing|mention|mentions|mentioning|require|requires|requiring|"
    r"permit|permits|accept|accepts)\b"
    r"(\s+(patients|participants|people|subjects|individuals))?"
    r"(\s+(with|who|that|having|of))?\s*[?.]?\s*$",
    re.IGNORECASE,
)


def strip_semantic(structured: str, semantic: Optional[str]) -> str:
    """Remove the eligibility concept from the structured half of a question.

    The router is asked to split a hybrid question in two, and usually does --
    but when it does not, it returns the *whole* question as the structured
    half while still extracting the semantic concept correctly. The SQL agent
    then dutifully tries to express that concept as a column filter
    (`conditions @> ARRAY['HIV']`), matches nothing, and the user gets a
    confident "no trials found" for a question with thousands of answers.

    Since we know the exact concept the router pulled out, removing it here is
    deterministic and does not depend on the model cooperating twice.
    """
    if not semantic:
        return structured
    text = structured
    needle = semantic.strip().rstrip("?.").strip()
    if needle and needle.lower() in text.lower():
        idx = text.lower().index(needle.lower())
        text = text[:idx] + text[idx + len(needle):]
    text = _DANGLING.sub("", text).strip()
    text = re.sub(r"\s{2,}", " ", text).strip(" ,;:?.")
    # If stripping consumed everything, fall back rather than send an empty
    # filter -- an empty description makes the agent invent one from nothing.
    return text or structured


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
    rerank: bool = False,
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
    # Mechanically remove the eligibility concept from the filter text, rather
    # than trusting the router to have done the split and the SQL agent to
    # honour an instruction to ignore it.
    filters = strip_semantic((structured_query or question).strip(), sem_query)
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
    result.hits = search_trials(sem_query, k=k, nct_ids=result.candidate_ids,
                                rerank=rerank)
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
