"""The three configurations compared in the evaluation.

* **router**      — the real pipeline: classify, then dispatch to SQL, vector,
                    or the hybrid SQL-then-vector path.
* **vector_only** — naive RAG. Every question goes to semantic search, no
                    matter what it asks for.
* **sql_only**    — every question goes to text-to-SQL, no matter what it asks.

The baselines exist to make the argument concrete rather than assumed. The
claim behind this whole project is that neither retriever alone can cover the
question space: vector search cannot count, and SQL cannot reason about
paraphrased medical prose. Running all three over the *same* questions turns
that claim into a table.

All three share the synthesizer and the citation guardrail, so the comparison
isolates retrieval strategy rather than measuring three different systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from trialsage.pipeline import AskResult, ask
from trialsage.retrieval.semantic import Hit, search_trials
from trialsage.retrieval.sql_agent import generate_sql
from trialsage.router.classify import RouteDecision
from trialsage.synth.synthesize import synthesize_from_hits, synthesize_from_rows
from trialsage.trace.tracer import Stopwatch


@dataclass
class RunResult:
    """One question answered by one configuration."""

    config: str
    question_id: str
    question: str
    predicted_route: str
    answer: str
    hits: List[Hit] = field(default_factory=list)
    sql: Optional[str] = None
    sql_rows: int = 0
    scalar: object = None
    decision: Optional[RouteDecision] = None
    cited: List[str] = field(default_factory=list)
    fabricated: List[str] = field(default_factory=list)
    uncited_claims: int = 0
    n_context_items: int = 0
    latency_s: float = 0.0
    total_tokens: int = 0
    error: Optional[str] = None

    @property
    def context_ids(self) -> List[str]:
        if self.hits:
            return [h.nct_id for h in self.hits]
        return []


def _from_ask(config: str, qid: str, result: AskResult) -> RunResult:
    audit = result.audit
    return RunResult(
        config=config,
        question_id=qid,
        question=result.question,
        predicted_route=result.route,
        answer=result.text,
        hits=result.hits,
        sql=(result.sql_result.sql if result.sql_result
             else (result.hybrid.sql if result.hybrid else None)),
        sql_rows=len(result.sql_result.rows) if result.sql_result else 0,
        scalar=result.sql_result.scalar() if result.sql_result else None,
        decision=result.decision,
        cited=sorted(audit.cited) if audit else [],
        fabricated=sorted(audit.fabricated) if audit else [],
        uncited_claims=len(audit.uncited_claims) if audit else 0,
        n_context_items=result.answer.n_context_items,
        latency_s=result.trace.latency_total_s if result.trace else 0.0,
        total_tokens=result.trace.total_tokens if result.trace else 0,
        error=result.answer.error,
    )


def run_router(qid: str, question: str, *, k: Optional[int] = None,
               rerank: bool = False) -> RunResult:
    result = ask(question, k=k, rerank=rerank)
    return _from_ask("router", qid, result)


def run_vector_only(qid: str, question: str, *, k: Optional[int] = None,
                    rerank: bool = False) -> RunResult:
    """Naive RAG: semantic search for everything, including counting questions."""
    watch = Stopwatch()
    hits = search_trials(question, k=k, rerank=rerank)
    answer = synthesize_from_hits(question, hits)
    audit = answer.audit
    return RunResult(
        config="vector_only",
        question_id=qid,
        question=question,
        predicted_route="semantic",
        answer=answer.text,
        hits=hits,
        cited=sorted(audit.cited) if audit else [],
        fabricated=sorted(audit.fabricated) if audit else [],
        uncited_claims=len(audit.uncited_claims) if audit else 0,
        n_context_items=answer.n_context_items,
        latency_s=round(watch.total(), 3),
        total_tokens=answer.total_tokens,
        error=answer.error,
    )


def run_sql_only(qid: str, question: str, **_: object) -> RunResult:
    """Text-to-SQL for everything, including questions about free-text prose."""
    watch = Stopwatch()
    sql_result = generate_sql(question)
    answer = synthesize_from_rows(question, sql_result)
    audit = answer.audit
    return RunResult(
        config="sql_only",
        question_id=qid,
        question=question,
        predicted_route="structured",
        answer=answer.text,
        sql=sql_result.sql,
        sql_rows=len(sql_result.rows),
        scalar=sql_result.scalar(),
        cited=sorted(audit.cited) if audit else [],
        fabricated=sorted(audit.fabricated) if audit else [],
        uncited_claims=len(audit.uncited_claims) if audit else 0,
        n_context_items=answer.n_context_items,
        latency_s=round(watch.total(), 3),
        total_tokens=sql_result.total_tokens + answer.total_tokens,
        error=answer.error or sql_result.error,
    )


CONFIGS: Dict[str, Callable[..., RunResult]] = {
    "router": run_router,
    "vector_only": run_vector_only,
    "sql_only": run_sql_only,
}
