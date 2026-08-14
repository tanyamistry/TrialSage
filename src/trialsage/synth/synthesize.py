"""Turn retrieved context into a grounded, cited answer.

Three rules the synthesizer is built to keep:

1. **Answer only from the retrieved context.** No outside medical knowledge,
   however confident the model feels about it.
2. **Cite the NCT ID for every trial-specific claim.**
3. **Say "no matching trials found" rather than inventing something.**

Rules 1 and 3 are asked for in the prompt. Rule 2 is asked for in the prompt
*and* verified afterwards by ``citations.audit_citations`` -- a prompt is a
request, and this is healthcare, so it also gets a check.

Polarity is rendered explicitly in the context (INCLUSION / EXCLUSION) because
it is the single easiest thing for a model to invert. "Trials that exclude
patients with autoimmune disease" and "trials that allow them" are opposite
answers built from near-identical text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Set

from ..llm import LLMError, get_llm
from ..retrieval.semantic import Hit
from ..retrieval.sql_agent import SQLResult
from .citations import CitationAudit, audit_citations

NO_MATCH = "No matching trials found."

_SYSTEM = (
    "You are a clinical-trials research assistant. You answer ONLY from the "
    "context provided to you. You never use outside knowledge about specific "
    "trials, drugs or diseases. Every statement about a specific trial must "
    "cite that trial's NCT ID in the form NCT12345678. If the context does "
    "not answer the question, you say so plainly."
)

_PROMPT_CRITERIA = """\
Context: eligibility criteria retrieved from clinical trials. Each line is
tagged with the trial's NCT ID and whether the criterion is an INCLUSION
(who may join) or an EXCLUSION (who may not).

{context}

Question: {question}

Write the answer as a short bulleted list, following these rules exactly:
- Use ONLY the context above. Do not add trials or facts from your own knowledge.
- One bullet per trial. Start each bullet with the NCT ID.
- After the NCT ID, say in your own words what that trial's criterion means for
  the patient, and whether it ALLOWS or EXCLUDES them.
- INCLUSION and EXCLUSION are opposites. A trial whose criterion is an
  EXCLUSION does NOT allow those patients. State the direction correctly --
  this is the most important rule here.
- Cover every distinct trial in the context, up to 8 bullets.
- Open with ONE sentence summarising the overall picture, then the bullets.
  The opening sentence must agree with the bullets beneath it.
- If the question asks which trials ALLOW something, but every retrieved
  criterion EXCLUDES it, do not say the context is empty -- the trials are
  right there. Say plainly that none of the retrieved trials allow it, then
  list the trials that exclude it.
- Never claim something is absent from the context when it is present.
- If the context is genuinely empty, reply exactly: {no_match}

Answer:"""

_PROMPT_ROWS = """\
Context: the result of a database query over the clinical-trials registry.

Query: {sql}

Result:
{context}

Question: {question}

Write a short answer, following these rules exactly:
- Use ONLY the result above. Do not add numbers or trials from your own knowledge.
- If the result is a single number, state it directly and plainly.
- Cite the NCT ID for any specific trial you mention.
- If the result is empty, reply exactly: {no_match}
- Be concise. No preamble.

Answer:"""


@dataclass
class Answer:
    """A synthesized answer plus everything needed to audit it."""

    question: str
    text: str
    audit: Optional[CitationAudit] = None
    context_ids: Set[str] = field(default_factory=set)
    n_context_items: int = 0
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def grounded(self) -> bool:
        """True when no fabricated citation survived into the answer."""
        return self.audit is None or not self.audit.has_fabrications


def format_criteria_context(hits: Sequence[Hit], max_items: int = 20) -> str:
    lines = []
    for hit in hits[:max_items]:
        title = (hit.brief_title or "").strip()
        header = f"[{hit.nct_id}]"
        if title:
            header += f" {title[:80]}"
        if hit.overall_status:
            header += f" (status: {hit.overall_status}"
            header += f", {hit.phase_display})" if hit.phase_display else ")"
        label = {"inclusion": "INCLUSION", "exclusion": "EXCLUSION"}.get(
            hit.criterion_type, "UNSPECIFIED")
        lines.append(f"{header}\n  {label}: {hit.criterion_text}")
    return "\n".join(lines)


def format_rows_context(result: SQLResult, max_rows: int = 30) -> str:
    if not result.rows:
        return "(no rows)"
    header = " | ".join(result.columns)
    lines = [header, "-" * len(header)]
    for row in result.rows[:max_rows]:
        lines.append(" | ".join("" if v is None else str(v) for v in row))
    if len(result.rows) > max_rows:
        lines.append(f"... and {len(result.rows) - max_rows} more rows")
    return "\n".join(lines)


def _ids_from_rows(result: SQLResult) -> Set[str]:
    return {i for i in result.nct_ids() if i}


def synthesize_from_hits(
    question: str,
    hits: Sequence[Hit],
    *,
    max_items: int = 20,
    citation_mode: str = "flag",
) -> Answer:
    """Answer a semantic or hybrid question from retrieved criteria."""
    allowed = {hit.nct_id for hit in hits}
    answer = Answer(question=question, text=NO_MATCH,
                    context_ids=allowed, n_context_items=len(hits))
    if not hits:
        # Short-circuit: with no context there is nothing to ground an answer
        # in, and asking the model anyway invites it to fill the gap.
        answer.audit = audit_citations(NO_MATCH, allowed, require_citations=False)
        return answer

    prompt = _PROMPT_CRITERIA.format(
        context=format_criteria_context(hits, max_items=max_items),
        question=question,
        no_match=NO_MATCH,
    )
    return _run(prompt, question, allowed, len(hits), answer, citation_mode)


def synthesize_from_rows(
    question: str,
    result: SQLResult,
    *,
    citation_mode: str = "flag",
) -> Answer:
    """Answer a structured question from SQL results."""
    allowed = _ids_from_rows(result)
    answer = Answer(question=question, text=NO_MATCH,
                    context_ids=allowed, n_context_items=len(result.rows))

    if not result.ok:
        answer.error = result.error
        answer.text = f"The structured query could not be completed: {result.error}"
        return answer
    if not result.rows:
        answer.audit = audit_citations(NO_MATCH, allowed, require_citations=False)
        return answer

    prompt = _PROMPT_ROWS.format(
        sql=result.sql or "",
        context=format_rows_context(result),
        question=question,
        no_match=NO_MATCH,
    )
    # An aggregate result ("75") names no trial, so requiring a citation on it
    # would be nonsense. Citations are only required once trials are listed.
    require = bool(allowed)
    return _run(prompt, question, allowed, len(result.rows), answer,
                citation_mode, require_citations=require)


def _run(prompt: str, question: str, allowed: Set[str], n_items: int,
         answer: Answer, citation_mode: str,
         require_citations: bool = True) -> Answer:
    started = time.perf_counter()
    try:
        llm = get_llm("synth")
        response = llm.complete(prompt, system=_SYSTEM)
    except LLMError as exc:
        answer.error = str(exc)
        answer.text = f"Could not generate an answer: {exc}"
        answer.latency_s = time.perf_counter() - started
        return answer

    answer.latency_s = time.perf_counter() - started
    answer.prompt_tokens = response.prompt_tokens
    answer.completion_tokens = response.completion_tokens

    audit = audit_citations(
        response.text.strip(),
        allowed,
        mode=citation_mode,  # type: ignore[arg-type]
        require_citations=require_citations,
    )
    answer.audit = audit
    answer.text = audit.text
    answer.n_context_items = n_items
    return answer
