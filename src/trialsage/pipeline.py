"""End-to-end: question -> route -> retrieve -> synthesize -> guard.

One function, :func:`ask`, is the single entry point. The CLI, the Phase 4
evaluation harness and the Phase 5 Streamlit UI all call it, so what gets
measured is exactly what gets shipped -- there is no separate "eval path" that
can drift from the real one.

The returned :class:`AskResult` carries the routing decision, the retrieval
internals and the citation audit alongside the answer, so a wrong answer can be
diagnosed without re-running anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .retrieval.hybrid import HybridResult, retrieve_hybrid
from .retrieval.semantic import Hit, search_trials
from .retrieval.sql_agent import SQLResult, generate_sql
from .router.classify import RouteDecision, classify
from .synth.citations import CitationAudit
from .synth.synthesize import Answer, synthesize_from_hits, synthesize_from_rows
from .trace import tracer
from .trace.tracer import Stopwatch, Trace


@dataclass
class AskResult:
    """Everything about one answered question."""

    question: str
    decision: RouteDecision
    answer: Answer
    sql_result: Optional[SQLResult] = None
    hybrid: Optional[HybridResult] = None
    hits: List[Hit] = field(default_factory=list)
    trace: Optional[Trace] = None

    @property
    def route(self) -> str:
        return self.decision.route

    @property
    def text(self) -> str:
        return self.answer.text

    @property
    def audit(self) -> Optional[CitationAudit]:
        return self.answer.audit

    @property
    def grounded(self) -> bool:
        return self.answer.grounded

    def explain(self) -> str:
        """Human-readable trace of how the answer was produced."""
        d = self.decision
        lines = [
            f"Q: {self.question}",
            "",
            "ROUTER",
            f"  route      : {d.route}",
            f"  confidence : {d.confidence:.2f}",
            f"  decided by : {d.source}",
            f"  reasoning  : {d.reasoning}",
        ]
        if d.semantic_query and d.semantic_query != self.question:
            lines.append(f"  semantic half  : {d.semantic_query!r}")
        if d.structured_query and d.structured_query != self.question:
            lines.append(f"  structured half: {d.structured_query!r}")

        lines.append("")
        lines.append("RETRIEVAL")
        if self.sql_result is not None:
            lines.append(f"  sql       : {self.sql_result.sql}")
            lines.append(f"  attempts  : {self.sql_result.n_attempts}")
            lines.append(f"  rows      : {len(self.sql_result.rows)}")
        if self.hybrid is not None:
            lines.append(f"  filter sql: {self.hybrid.sql}")
            lines.append(f"  candidates: {len(self.hybrid.candidate_ids)} trials")
            lines.append(f"  semantic  : {self.hybrid.semantic_query!r}")
        if self.hits:
            lines.append(f"  hits      : {len(self.hits)}")
            for hit in self.hits[:3]:
                lines.append(f"    {hit.score:.3f} [{hit.criterion_type}] {hit.nct_id}"
                             f"  {hit.criterion_text[:70]}")

        lines.append("")
        lines.append("ANSWER")
        for line in self.text.splitlines():
            lines.append(f"  {line}")

        if self.audit:
            lines.append("")
            lines.append("CITATION GUARDRAIL")
            lines.append(f"  {self.audit.summary()}")
            if self.audit.fabricated:
                lines.append(f"  FABRICATED (neutralised): {', '.join(sorted(self.audit.fabricated))}")
            for claim in self.audit.uncited_claims[:2]:
                lines.append(f"  uncited: \"{claim[:90]}\"")

        if self.trace:
            t = self.trace
            lines.append("")
            lines.append(f"TIMING  total {t.latency_total_s:.1f}s "
                         f"(route {t.latency_route_s:.1f}s, retrieve {t.latency_retrieve_s:.1f}s, "
                         f"synth {t.latency_synth_s:.1f}s)  tokens {t.total_tokens}")
        return "\n".join(lines)


def ask(
    question: str,
    *,
    k: Optional[int] = None,
    use_llm_router: bool = True,
    citation_mode: str = "flag",
    trace: bool = True,
) -> AskResult:
    """Answer a question end to end."""
    watch = Stopwatch()

    decision = classify(question, use_llm=use_llm_router)
    t_route = watch.lap()

    sql_result: Optional[SQLResult] = None
    hybrid: Optional[HybridResult] = None
    hits: List[Hit] = []

    if decision.route == "structured":
        sql_result = generate_sql(question)
        t_retrieve = watch.lap()
        answer = synthesize_from_rows(question, sql_result, citation_mode=citation_mode)

    elif decision.route == "semantic":
        query = decision.semantic_query or question
        hits = search_trials(query, k=k)
        t_retrieve = watch.lap()
        answer = synthesize_from_hits(question, hits, citation_mode=citation_mode)

    else:  # hybrid
        hybrid = retrieve_hybrid(question, semantic_query=decision.semantic_query,
                                 structured_query=decision.structured_query, k=k)
        hits = hybrid.hits
        t_retrieve = watch.lap()
        if hybrid.error:
            answer = synthesize_from_hits(question, [], citation_mode=citation_mode)
            answer.error = hybrid.error
            answer.text = f"Could not apply the structured filter: {hybrid.error}"
        elif hybrid.filtered_out:
            # Be specific: nothing matched the FILTER, which is a different
            # fact from "no trial mentions this concept".
            answer = synthesize_from_hits(question, [], citation_mode=citation_mode)
            answer.text = ("No matching trials found. No trials matched the "
                           "structured filters in this question.")
        else:
            answer = synthesize_from_hits(question, hits, citation_mode=citation_mode)

    t_synth = watch.lap()

    tr = Trace(
        question=question,
        route=decision.route,
        route_confidence=decision.confidence,
        route_source=decision.source,
        route_reasoning=decision.reasoning,
        latency_total_s=round(watch.total(), 3),
        latency_route_s=round(t_route, 3),
        latency_retrieve_s=round(t_retrieve, 3),
        latency_synth_s=round(t_synth, 3),
        prompt_tokens=decision.prompt_tokens + answer.prompt_tokens
        + (sql_result.prompt_tokens if sql_result else 0)
        + (hybrid.sql_result.prompt_tokens if hybrid and hybrid.sql_result else 0),
        completion_tokens=decision.completion_tokens + answer.completion_tokens
        + (sql_result.completion_tokens if sql_result else 0)
        + (hybrid.sql_result.completion_tokens if hybrid and hybrid.sql_result else 0),
        sql=(sql_result.sql if sql_result else (hybrid.sql if hybrid else None)),
        sql_attempts=(sql_result.n_attempts if sql_result
                      else (hybrid.sql_result.n_attempts if hybrid and hybrid.sql_result else 0)),
        n_candidates=len(hybrid.candidate_ids) if hybrid else 0,
        n_context_items=answer.n_context_items,
        citations_valid=len(answer.audit.cited) if answer.audit else 0,
        citations_fabricated=len(answer.audit.fabricated) if answer.audit else 0,
        uncited_claims=len(answer.audit.uncited_claims) if answer.audit else 0,
        grounded=answer.grounded,
        error=answer.error,
    )
    if trace:
        tracer.write(tr)

    return AskResult(question=question, decision=decision, answer=answer,
                     sql_result=sql_result, hybrid=hybrid, hits=hits, trace=tr)
