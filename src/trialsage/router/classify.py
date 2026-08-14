"""Classify a question as structured, semantic, or hybrid -- and explain why.

This is the piece the whole project is built around. Naive RAG cannot count, and
text-to-SQL cannot reason about "prior immunotherapy failure", so something has
to decide which machinery a question needs.

Two design commitments:

**The decision is a returned object, not a side effect.** ``RouteDecision``
carries the route, a confidence, a plain-language reason, and which mechanism
decided (LLM or rules). Nothing downstream has to guess why a question went
where it did, and the UI can show it.

**There is always a rule-based fallback.** A local 8B model asked for JSON will
occasionally return prose, malformed JSON, or an invented route name. When that
happens we do not fail and we do not silently default to one route -- we fall
back to keyword heuristics and say so in ``source``, so a degraded decision is
visible rather than disguised as a confident one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from ..llm import LLMError, get_llm

Route = Literal["structured", "semantic", "hybrid"]
VALID_ROUTES = ("structured", "semantic", "hybrid")


@dataclass
class RouteDecision:
    """Why a question was routed the way it was."""

    route: Route
    confidence: float
    reasoning: str
    source: Literal["llm", "rules", "llm+rules"] = "llm"
    # For hybrid questions the two halves are separated here. Keeping them
    # apart is what makes the hybrid route work: sending the whole question to
    # the SQL agent makes it try to express "history of autoimmune disease" as
    # a column filter, which matches nothing and produces a false negative.
    semantic_query: Optional[str] = None
    structured_query: Optional[str] = None
    llm_raw: Optional[str] = field(default=None, repr=False)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def explain(self) -> str:
        return (f"route={self.route} confidence={self.confidence:.2f} "
                f"source={self.source}\n  reason: {self.reasoning}")


# --- rule-based classifier -------------------------------------------------
#
# Used as a fallback when the LLM fails, and as a cross-check on low-confidence
# LLM answers. Deliberately simple and fully deterministic so its behaviour is
# obvious when you are debugging a bad route.

# Aggregation and filtering on fields that live in columns.
_STRUCTURED_PATTERNS = [
    r"\bhow many\b", r"\bcount\b", r"\bnumber of\b", r"\btotal\b",
    r"\baverage\b", r"\bmean\b", r"\bmedian\b", r"\bsum\b",
    r"\bmost\b", r"\bfewest\b", r"\bhighest\b", r"\blowest\b", r"\btop \d+\b",
    r"\benrol?ment\b", r"\bsponsor\b", r"\bstarted in\b", r"\bcompleted in\b",
    r"\bphase [1-4]\b", r"\bin \d{4}\b", r"\bsince \d{4}\b",
    r"\brecruiting\b", r"\bwithdrawn\b", r"\bterminated\b",
]

# Free-text concepts that only exist inside the eligibility prose.
_SEMANTIC_PATTERNS = [
    r"\beligibilit", r"\bcriteri", r"\bmentions?\b", r"\bmention(ing|ed)\b",
    r"\bhistory of\b", r"\bprior\b", r"\bpreviously\b", r"\brefractory\b",
    r"\bprogressed\b", r"\bfailure\b", r"\bfailed\b", r"\bintolerant\b",
    r"\ballow(s|ing|ed)?\b", r"\bexclude(s|d)?\b", r"\bexclusion\b",
    r"\binclusion\b", r"\bpatients with\b", r"\bpeople with\b",
    r"\bwho (can|cannot|can't)\b", r"\bcomorbid", r"\bcontraindicat",
]

# Structured concepts that, on their own, are strong location/status filters.
_FILTER_PATTERNS = [
    r"\bin (massachusetts|california|texas|new york|florida)\b",
    r"\bphase [1-4]\b", r"\brecruiting\b", r"\bactive\b",
    r"\boncology\b", r"\bdiabetes\b", r"\bcardiovascular\b",
]


def _hits(patterns: List[str], text: str) -> List[str]:
    found = []
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            found.append(match.group(0).strip())
    return found


def classify_by_rules(question: str) -> RouteDecision:
    """Deterministic keyword classifier. Never raises."""
    text = question.lower()
    structured = _hits(_STRUCTURED_PATTERNS, text)
    semantic = _hits(_SEMANTIC_PATTERNS, text)
    filters = _hits(_FILTER_PATTERNS, text)

    # An aggregation verb ("how many", "average") means the answer is a number
    # computed over rows -- vector search cannot produce that.
    counting = bool(re.search(r"\bhow many\b|\bcount\b|\baverage\b|\btotal\b|\bnumber of\b", text))

    if semantic and (structured or filters):
        # Both kinds of signal: filter first, then search the prose.
        route: Route = "hybrid"
        confidence = 0.75
        reason = (f"mentions free-text eligibility concepts ({', '.join(semantic[:3])}) "
                  f"and structured filters ({', '.join((structured + filters)[:3])})")
    elif semantic:
        route, confidence = "semantic", 0.7
        reason = f"asks about eligibility prose ({', '.join(semantic[:3])}) with no structured filter"
    elif structured or counting:
        route, confidence = "structured", 0.7
        reason = f"asks for a value computed over columns ({', '.join(structured[:3]) or 'aggregation'})"
    else:
        # Nothing matched. Semantic is the safer default: it returns cited
        # criteria the synthesizer can refuse to over-claim on, whereas a
        # wrong SQL query returns a confident number that happens to be wrong.
        route, confidence = "semantic", 0.35
        reason = "no clear structured or semantic signal; defaulting to semantic search"

    return RouteDecision(route=route, confidence=confidence, reasoning=reason,
                         source="rules", semantic_query=question)


# --- LLM classifier --------------------------------------------------------

_SYSTEM = (
    "You are a query router for a clinical-trials assistant. You classify a "
    "question into exactly one route and reply with a single JSON object. "
    "No prose, no markdown, no explanation outside the JSON."
)

_PROMPT = """\
A clinical-trials database has two kinds of information:

  STRUCTURED columns: phase, recruiting status, enrollment count, start date,
  sponsor, country, US state, therapeutic area, minimum/maximum age.

  FREE TEXT: the eligibility criteria describing who may or may not join --
  prior treatments, medical history, comorbidities, lab thresholds.

Choose the route:

  "structured" - answerable from columns alone. Counting, averaging, filtering
                 by phase/status/date/location. Example: "How many phase 3
                 diabetes trials started in 2024?"

  "semantic"   - needs the eligibility prose, with no structured filter.
                 Example: "Find trials whose eligibility mentions prior
                 immunotherapy failure."

  "hybrid"     - needs BOTH: a structured filter AND an eligibility concept.
                 Example: "Which recruiting phase 2 oncology trials in
                 Massachusetts allow patients with a history of autoimmune
                 disease?"

Reply with exactly this JSON shape:

{{"route": "structured|semantic|hybrid",
  "confidence": 0.0-1.0,
  "reasoning": "one short sentence",
  "semantic_query": "the eligibility concept alone, or null",
  "structured_query": "the column filters alone, or null"}}

For "hybrid" you MUST split the question into its two halves:

  semantic_query   - ONLY the medical/eligibility concept. No phase, no status,
                     no location, no therapeutic area.
  structured_query - ONLY the column filters. No medical/eligibility concept.

For the Massachusetts example above:
  semantic_query   = "history of autoimmune disease"
  structured_query = "recruiting phase 2 oncology trials in Massachusetts"

Getting this split wrong breaks the query: an eligibility concept left in
structured_query matches no column and returns zero trials.

Question: {question}

JSON:"""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of a possibly chatty response."""
    text = text.strip()
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if blocks:
            text = blocks[0].strip()
    start = text.find("{")
    if start == -1:
        return None
    # Walk braces so trailing prose after the object does not break parsing.
    depth = 0
    for i, char in enumerate(text[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def classify(question: str, *, use_llm: bool = True,
             min_confidence: float = 0.4) -> RouteDecision:
    """Classify a question, falling back to rules whenever the LLM is unusable.

    The fallback triggers on an unreachable model, unparseable output, an
    invented route name, or a confidence below ``min_confidence``. In every
    case ``source`` records what actually decided.
    """
    rules = classify_by_rules(question)
    if not use_llm:
        return rules

    try:
        llm = get_llm("router")
        response = llm.complete(_PROMPT.format(question=question), system=_SYSTEM)
    except LLMError as exc:
        rules.reasoning = f"{rules.reasoning} (LLM unavailable: {exc})"
        return rules

    payload = _extract_json(response.text)
    if not payload:
        rules.reasoning = f"{rules.reasoning} (LLM returned unparseable output)"
        rules.llm_raw = response.text
        rules.prompt_tokens = response.prompt_tokens
        rules.completion_tokens = response.completion_tokens
        rules.latency_s = response.latency_s
        return rules

    route = str(payload.get("route", "")).strip().lower()
    if route not in VALID_ROUTES:
        rules.reasoning = f"{rules.reasoning} (LLM proposed invalid route {route!r})"
        rules.llm_raw = response.text
        return rules

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reasoning = str(payload.get("reasoning") or "").strip() or "no reason given"

    def _field(name: str) -> Optional[str]:
        value = payload.get(name)
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not value or value.lower() in ("null", "none", "n/a"):
            return None
        return value

    semantic_query = _field("semantic_query")
    structured_query = _field("structured_query")
    if route in ("semantic", "hybrid") and not semantic_query:
        semantic_query = question
    if route == "hybrid" and not structured_query:
        # Falling back to the whole question here is safe: the SQL filter
        # template tells the agent to ignore eligibility prose.
        structured_query = question

    # Low LLM confidence: keep the LLM's route but record that the rules agreed
    # or disagreed, so a shaky decision is visible downstream.
    source: Literal["llm", "rules", "llm+rules"] = "llm"
    if confidence < min_confidence:
        if rules.route != route:
            reasoning = (f"LLM was unsure ({confidence:.2f}) and said {route}; "
                         f"rules say {rules.route} because {rules.reasoning}")
            route, confidence, source = rules.route, rules.confidence, "rules"
            semantic_query = rules.semantic_query if route != "structured" else None
            structured_query = question if route == "hybrid" else None
        else:
            source = "llm+rules"
            reasoning = f"{reasoning} (low LLM confidence, but rules agree)"
            confidence = max(confidence, rules.confidence)

    return RouteDecision(
        route=route,  # type: ignore[arg-type]
        confidence=confidence,
        reasoning=reasoning,
        source=source,
        semantic_query=semantic_query,
        structured_query=structured_query,
        llm_raw=response.text,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        latency_s=response.latency_s,
    )
