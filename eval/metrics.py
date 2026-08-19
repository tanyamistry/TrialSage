"""Scoring for the evaluation harness.

Deliberately mechanical: every metric here is computed from the gold set and
the run output, with no LLM judgement involved. RAGAS (eval/ragas_eval.py)
adds the judged metrics separately, so a wobble in the judge model cannot move
these numbers.

The metric that carries the most weight clinically is `polarity_correct`.
Everything else measures whether the system found the right trials; that one
measures whether it described them the right way round.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .baselines import RunResult
from .gold import EvalQuestion

# Wording that asserts a trial permits something. Used to detect a polarity
# inversion: saying a trial "allows" patients whose criterion is an EXCLUSION.
_ALLOW_VERBS = ("allows", "allow ", "permits", "permit ", "accepts", "accept ",
                "includes patients", "eligible", "can join", "may join")
_EXCLUDE_WORDS = ("exclude", "excludes", "excluded", "excluding", "not allow",
                  "does not allow", "none of", "not eligible", "ineligible")

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers_in(text: str) -> List[float]:
    out = []
    for raw in _NUMBER_RE.findall(text or ""):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


def answer_matches_gold(answer: str, gold: Any, *, tolerance: float = 0.01) -> bool:
    """True when the gold value appears in the answer.

    Numeric comparison with a small tolerance, because a model may round an
    average. Exact substring match for non-numeric golds.
    """
    if gold is None:
        return False
    try:
        target = float(gold)
    except (TypeError, ValueError):
        return str(gold).lower() in (answer or "").lower()

    for value in _numbers_in(answer):
        if abs(value - target) <= max(tolerance, abs(target) * tolerance):
            return True
    return False


def term_recall(hits: Sequence[Any], terms: Sequence[str]) -> float:
    """Fraction of retrieved criteria containing at least one expected term.

    A blunt proxy for retrieval precision that needs no human labels: if a
    search for "prior immunotherapy failure" returns criteria that never
    mention immunotherapy or anything adjacent, retrieval is off regardless of
    what the cosine score says.
    """
    if not hits or not terms:
        return 0.0
    relevant = sum(
        1 for h in hits
        if any(t in h.criterion_text.lower() for t in terms)
    )
    return relevant / len(hits)


def polarity_correct(answer: str, hits: Sequence[Any],
                     expected: Optional[str]) -> Optional[bool]:
    """Check the answer does not invert inclusion/exclusion.

    Returns None when the question has no polarity expectation.

    The specific failure this catches: every retrieved criterion is an
    EXCLUSION, but the answer says a trial ALLOWS those patients. That sends
    someone toward a trial that will turn them away, and it is the most
    damaging thing this system can get wrong.
    """
    if not expected or not hits:
        return None

    lowered = (answer or "").lower()
    matching = [h for h in hits if h.criterion_type == expected]
    if not matching:
        return None

    if expected == "exclusion":
        # Flag "NCTxxxxxxxx allows ..." for a trial whose criterion excludes.
        for hit in matching:
            nct = hit.nct_id.lower()
            idx = lowered.find(nct)
            while idx != -1:
                window = lowered[idx:idx + 90]
                if any(v in window for v in _ALLOW_VERBS) and \
                        not any(w in window for w in _EXCLUDE_WORDS):
                    return False
                idx = lowered.find(nct, idx + 1)
        # And require the answer to convey exclusion somewhere.
        return any(w in lowered for w in _EXCLUDE_WORDS)

    return True


def filter_precision(hits: Sequence[Any], candidate_ids: Sequence[str]) -> Optional[float]:
    """Fraction of retrieved trials that actually satisfy the structured filter.

    This is the metric that exposes how vector-only fails on hybrid questions.
    Asked for "recruiting phase 2 oncology trials in Massachusetts", naive RAG
    happily returns the most semantically similar criteria in the registry --
    from completed trials, the wrong phase, the wrong state. It cites real
    trials and reads fluently, so every citation-based metric says it did fine.
    Only checking the citations against the filter reveals the answer is about
    the wrong trials entirely.
    """
    if not hits or not candidate_ids:
        return None
    allowed = set(candidate_ids)
    return sum(1 for h in hits if h.nct_id in allowed) / len(hits)


@dataclass
class QuestionScore:
    question_id: str
    route_gold: str
    route_pred: str
    config: str
    routed_correctly: Optional[bool] = None
    answer_correct: Optional[bool] = None
    term_recall: Optional[float] = None
    filter_precision: Optional[float] = None
    polarity_ok: Optional[bool] = None
    n_cited: int = 0
    n_fabricated: int = 0
    uncited_claims: int = 0
    refused: bool = False
    latency_s: float = 0.0
    total_tokens: int = 0
    error: Optional[str] = None


def score_run(q: EvalQuestion, run: RunResult) -> QuestionScore:
    """Score one configuration's answer to one question."""
    answer = run.answer or ""
    refused = "no matching trials found" in answer.lower()

    score = QuestionScore(
        question_id=q.id,
        route_gold=q.route,
        route_pred=run.predicted_route,
        config=run.config,
        n_cited=len(run.cited),
        n_fabricated=len(run.fabricated),
        uncited_claims=run.uncited_claims,
        refused=refused,
        latency_s=run.latency_s,
        total_tokens=run.total_tokens,
        error=run.error,
    )

    # Routing accuracy only means something for the configuration that routes.
    if run.config == "router":
        score.routed_correctly = run.predicted_route == q.route

    # Structured questions have a checkable numeric answer.
    if q.route == "structured" and q.gold_answer is not None:
        score.answer_correct = answer_matches_gold(answer, q.gold_answer)

    # Semantic and hybrid questions are judged on what was retrieved.
    if q.route in ("semantic", "hybrid"):
        if q.expected_terms:
            score.term_recall = term_recall(run.hits, q.expected_terms)
        score.polarity_ok = polarity_correct(answer, run.hits, q.expected_polarity)
        # A retrieval-based question is "answered" if it cited any real trial.
        score.answer_correct = bool(run.cited) and not refused

    # Hybrid questions carry a gold structured filter, so we can check whether
    # the trials actually discussed are the ones the question asked about.
    if q.route == "hybrid" and q.gold_candidate_ids:
        score.filter_precision = filter_precision(run.hits, q.gold_candidate_ids)
        # Citing the wrong trials is not a correct answer, however fluent.
        if score.filter_precision is not None and score.filter_precision < 0.5:
            score.answer_correct = False

    return score


@dataclass
class ConfigSummary:
    config: str
    n: int = 0
    routing_accuracy: Optional[float] = None
    structured_accuracy: Optional[float] = None
    semantic_answered: Optional[float] = None
    hybrid_answered: Optional[float] = None
    mean_term_recall: Optional[float] = None
    mean_filter_precision: Optional[float] = None
    polarity_accuracy: Optional[float] = None
    citation_validity: Optional[float] = None
    total_fabricated: int = 0
    refusal_rate: float = 0.0
    mean_latency_s: float = 0.0
    mean_tokens: float = 0.0
    by_route: Dict[str, float] = field(default_factory=dict)


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def summarise(config: str, scores: List[QuestionScore]) -> ConfigSummary:
    s = ConfigSummary(config=config, n=len(scores))
    if not scores:
        return s

    routed = [x.routed_correctly for x in scores if x.routed_correctly is not None]
    s.routing_accuracy = _mean([float(x) for x in routed]) if routed else None

    for route, attr in (("structured", "structured_accuracy"),
                        ("semantic", "semantic_answered"),
                        ("hybrid", "hybrid_answered")):
        vals = [float(bool(x.answer_correct)) for x in scores
                if x.route_gold == route and x.answer_correct is not None]
        setattr(s, attr, _mean(vals))
        if vals:
            s.by_route[route] = _mean(vals)

    s.mean_term_recall = _mean([x.term_recall for x in scores
                                if x.term_recall is not None])
    s.mean_filter_precision = _mean([x.filter_precision for x in scores
                                     if x.filter_precision is not None])
    pol = [float(x.polarity_ok) for x in scores if x.polarity_ok is not None]
    s.polarity_accuracy = _mean(pol) if pol else None

    total_citations = sum(x.n_cited + x.n_fabricated for x in scores)
    s.total_fabricated = sum(x.n_fabricated for x in scores)
    s.citation_validity = (
        (total_citations - s.total_fabricated) / total_citations
        if total_citations else None
    )

    s.refusal_rate = _mean([float(x.refused) for x in scores]) or 0.0
    s.mean_latency_s = _mean([x.latency_s for x in scores]) or 0.0
    s.mean_tokens = _mean([float(x.total_tokens) for x in scores]) or 0.0
    return s


def confusion_matrix(scores: List[QuestionScore]) -> Dict[str, Dict[str, int]]:
    """gold route -> predicted route -> count. Router configuration only."""
    routes = ("structured", "semantic", "hybrid")
    matrix = {g: {p: 0 for p in routes} for g in routes}
    for score in scores:
        if score.route_gold in matrix and score.route_pred in matrix[score.route_gold]:
            matrix[score.route_gold][score.route_pred] += 1
    return matrix
