"""Enforce that every trial-specific claim is backed by a real retrieved trial.

The synthesizer prompt asks the model to cite an NCT ID for each claim. This
module assumes that request will sometimes be ignored, because it will be. It
is the mechanical check that runs after generation, in the same spirit as the
SQL guard: instructions are a request, verification is a control.

Two distinct failure modes, handled differently:

**Fabricated citations** -- an NCT ID that was never in the retrieved context.
These are the dangerous ones. An NCT ID looks authoritative and is exactly the
kind of thing a reader will not check. They are always neutralised.

**Uncited claims** -- a sentence that asserts something trial-specific without
naming a trial. Flagged rather than removed, because the detection is
heuristic and deleting real content would be worse than surfacing a warning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Literal, Set

# ClinicalTrials.gov identifiers are always NCT + 8 digits.
NCT_RE = re.compile(r"\bNCT\d{8}\b")

# Wording that makes a sentence a claim about a specific trial rather than a
# general statement. Used only to decide whether a missing citation matters.
_TRIAL_CLAIM_TERMS = (
    "trial", "study", "enrol", "recruit", "phase", "eligib", "criteri",
    "exclude", "include", "patient", "participant", "sponsor",
)

# Sentences that are commentary about the result set, not claims about a trial.
_META_TERMS = (
    "no matching trials", "no trials", "not found", "no results",
    "based on the retrieved", "the retrieved context", "context provided",
    "i cannot", "i don't have", "unable to", "the following trials",
    "here are", "in summary", "summary:",
)

# Quantifiers that make a sentence an aggregate statement about the whole
# retrieved set rather than a claim about one trial. "None of the trials allow
# X" is supported by the cited bullets beneath it, so demanding an NCT ID
# inside the summary line itself would be a false positive -- and a guardrail
# that cries wolf on correct answers is one people learn to ignore.
_AGGREGATE_PATTERNS = (
    # Quantifier-initial: "None of the trials allow X."
    r"^\s*(none|all|each|every|most|several|some|both|neither)\b",
    r"\b(none|all|each|every|neither) of (the|these|those)\b",
    r"^\s*(there are|there is)\b",
    r"^\s*\d+\s+(of\s+)?(the\s+)?\w*\s*trials?\b",
    # Lead-in sentences that introduce the cited bullets beneath them. These
    # were the bulk of the false positives in the Phase 4 sweep: 38 flagged
    # "uncited claims", almost all of the form "The eligibility criteria for X
    # are mentioned in several trials:" immediately followed by cited bullets.
    r"^\s*the (following|retrieved|above|context|trials?|eligibility|criteria|"
    r"clinical trials?|majority|studies)\b",
    r"^\s*(these|those) (trials?|criteria|studies)\b",
    r"^\s*based on\b",
    r"\bare mentioned in\b",
    r"\bhere are the (details|trials|results)\b",
)

Mode = Literal["flag", "strip"]


@dataclass
class CitationAudit:
    """Result of checking an answer against the trials actually retrieved."""

    text: str
    cited: Set[str] = field(default_factory=set)
    fabricated: Set[str] = field(default_factory=set)
    uncited_claims: List[str] = field(default_factory=list)
    unused_context: Set[str] = field(default_factory=set)

    @property
    def ok(self) -> bool:
        """True when nothing was fabricated and no claim is left uncited."""
        return not self.fabricated and not self.uncited_claims

    @property
    def has_fabrications(self) -> bool:
        return bool(self.fabricated)

    def summary(self) -> str:
        parts = [f"{len(self.cited)} valid citation(s)"]
        if self.fabricated:
            parts.append(f"{len(self.fabricated)} FABRICATED: {', '.join(sorted(self.fabricated))}")
        if self.uncited_claims:
            parts.append(f"{len(self.uncited_claims)} uncited claim(s)")
        return "; ".join(parts)


def extract_nct_ids(text: str) -> Set[str]:
    """Every NCT identifier appearing in a piece of text."""
    return set(NCT_RE.findall(text or ""))


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p.strip() for p in parts if p.strip()]


def _is_trial_claim(sentence: str, *, answer_has_citations: bool = False) -> bool:
    """True when a sentence asserts something about a *specific* trial.

    Meta-commentary never counts. An aggregate summary ("None of the trials
    allow X") counts only when the answer cites nothing anywhere -- a summary
    is legitimate when the specifics beneath it are cited, but a bare
    uncited generalisation is exactly the kind of unsupported claim this
    guardrail exists to surface.
    """
    lowered = sentence.lower().strip("•-* \t")
    if any(term in lowered for term in _META_TERMS):
        return False
    if answer_has_citations and any(re.search(p, lowered) for p in _AGGREGATE_PATTERNS):
        return False
    # A bullet or heading with no verb is not a claim.
    if len(lowered.split()) < 5:
        return False
    return any(term in lowered for term in _TRIAL_CLAIM_TERMS)


def audit_citations(
    answer: str,
    allowed_ids: Iterable[str],
    *,
    mode: Mode = "flag",
    require_citations: bool = True,
) -> CitationAudit:
    """Check ``answer`` against the trials that were actually retrieved.

    ``allowed_ids`` is the set of NCT IDs present in the retrieved context.
    Anything cited outside that set is a fabrication and is neutralised:

    * ``mode="flag"``   replaces it with ``[unverified: NCTxxxxxxxx]`` so the
      reader can see that something was claimed and rejected;
    * ``mode="strip"``  removes the identifier entirely.

    Flagging is the default. Silently deleting a fabricated citation would
    leave the surrounding sentence intact and looking sourced, which is worse
    than showing that the model made something up.
    """
    allowed = {i.upper() for i in allowed_ids}
    text = answer or ""

    found = extract_nct_ids(text)
    cited = found & allowed
    fabricated = found - allowed

    for bad in fabricated:
        replacement = "" if mode == "strip" else f"[unverified: {bad}]"
        text = re.sub(rf"\b{re.escape(bad)}\b", replacement, text)

    if mode == "strip" and fabricated:
        # Tidy the punctuation left behind by removed identifiers.
        text = re.sub(r"\(\s*[,;]?\s*\)", "", text)
        text = re.sub(r"\[\s*[,;]?\s*\]", "", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\s+([.,;:])", r"\1", text)

    uncited: List[str] = []
    if require_citations:
        answer_has_citations = bool(cited)
        for sentence in _split_sentences(text):
            if (_is_trial_claim(sentence, answer_has_citations=answer_has_citations)
                    and not NCT_RE.search(sentence)):
                uncited.append(sentence)

    return CitationAudit(
        text=text.strip(),
        cited=cited,
        fabricated=fabricated,
        uncited_claims=uncited,
        unused_context=allowed - cited,
    )


def append_warnings(audit: CitationAudit) -> str:
    """Return the answer with any guardrail warnings appended for display."""
    if audit.ok:
        return audit.text

    lines = [audit.text, "", "---", "**Citation guardrail**"]
    if audit.fabricated:
        lines.append(
            f"- Removed {len(audit.fabricated)} citation(s) not present in the "
            f"retrieved context: {', '.join(sorted(audit.fabricated))}. "
            "These trials were not retrieved and may not exist."
        )
    if audit.uncited_claims:
        lines.append(f"- {len(audit.uncited_claims)} statement(s) make a "
                     "trial-specific claim without citing an NCT ID:")
        for claim in audit.uncited_claims[:3]:
            lines.append(f"  - \"{claim[:120]}\"")
    return "\n".join(lines)
