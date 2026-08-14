"""Split raw eligibility text into individually tagged inclusion/exclusion criteria.

This is the most important parser in the project. "Trials that INCLUDE patients
with autoimmune disease" and "trials that EXCLUDE patients with autoimmune
disease" are opposite facts, and they are frequently phrased with nearly
identical wording. If a chunk is not tagged with which section it came from,
the vector search cannot tell them apart and the assistant will confidently
give the wrong answer.

We chunk per criterion rather than by fixed token window for the same reason:
a 512-token window routinely straddles the Inclusion/Exclusion boundary, which
produces a chunk that is half one polarity and half the other.

Everything here is driven by what the live API actually returns. Profiling 200
real oncology records showed:

* 82% bullet their criteria with ``* ``, 15% use ``1.``/``1)``, 2% use neither.
* 91% have both section headers, 3.5% have inclusion only, 5% have no headers
  at all -- so a "just split on the headers" approach silently drops 1 in 11.
* The text is escaped markdown: a real sample reads
  ``Patients must be \\> 365 days and \\< 18 years ... \\[COG\\]``.
  Left unescaped, criteria read as garbage and embed badly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Optional

CriterionType = Literal["inclusion", "exclusion", "unspecified"]

# Deliberately low. An earlier version used 15 chars to filter noise and was
# silently discarding ~0.5 real criteria per trial: "Pregnancy", "Hypertension",
# "Active cancer", "Prisoners", "Breastfeeding" are among the most common
# exclusions in clinical research and all fall under 15 characters. Junk is
# better identified by shape (see _is_junk) than by length.
DEFAULT_MIN_CHARS = 4
DEFAULT_MAX_CHARS = 2000


@dataclass(frozen=True)
class Criterion:
    """One eligibility criterion, tagged with the section it came from."""

    criterion_type: CriterionType
    chunk_index: int
    text: str

    @property
    def char_len(self) -> int:
        return len(self.text)


# --- markdown unescaping ---------------------------------------------------

# The registry escapes markdown punctuation with a backslash. Undo that.
# `^` is not markdown-special but shows up escaped in scientific notation
# ("5 \^ 10\^9"), so it is included here too.
_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!<>~|^])")


def unescape(text: str) -> str:
    """Turn ``\\> 365 days ... \\[COG\\]`` back into ``> 365 days ... [COG]``."""
    return _ESCAPE_RE.sub(r"\1", text)


# --- section header detection ----------------------------------------------

# Anchored form: the header sits on its own line, possibly bulleted or bolded.
#
# An earlier version required the line to end immediately after "Criteria"
# (plus an optional colon). That missed 894 trials (2.3% of the corpus) whose
# headers carry a trailing qualifier, and the consequence was severe rather
# than cosmetic: when "Key Exclusion Criteria with:" goes undetected, every
# exclusion below it inherits the preceding *inclusion* tag, and the system
# reports that a trial ALLOWS patients it in fact EXCLUDES.
#
# Real spellings this has to handle, all observed in the corpus:
#   "Inclusion Criteria:"      "Key Exclusion Criteria with:"
#   "Inclusion Criteria："     (full-width colon, from CJK input)
#   "Inclusion Criteria -"     "Inclusion Criteria:-"    "Inclusion Criteria."
#   "Inclusion Criteria (abbreviated):"   "Inclusion Criteria 1:"
#   "Inclusion Criteria include, but are not limited to:"
#
# And must NOT match, also observed:
#   "* Inclusion in another clinical trial"     <- a criterion, no "criteria"
#   "Inclusion of Women and Minorities"         <- a section, no "criteria"
#   "Note: Other protocol defined Inclusion/Exclusion criteria may apply."
#   "* Exclusion criteria will be assessed at screening"   <- prose
#
# The distinguishing signals: the word "criteria" must be present (unless the
# bare "Inclusion:" form is used), and a trailing qualifier is only accepted
# when the line ends in a colon or is short and verb-free.
_COLON = r"[:：]"          # ASCII and full-width colon
_VERBS = r"(?:will|would|may|might|must|shall|should|are|is|were|was|be|been|" \
         r"apply|applies|applied|assess|assessed|include[sd]|listed|meet|meets|" \
         r"see|refer|note[sd]?)"

_HEADER_ANCHORED = re.compile(
    rf"""^[ \t]*                          # leading indent
        (?:[*\-•]|\d+[.)])?          # optional bullet / number
        [ \t]*\**[ \t]*                   # optional bold markers
        (?:key[ \t]+)?                    # "Key Inclusion Criteria"
        (?P<kind>inclusion|exclusion)
        (?:
            [ \t]+criteria                # "Inclusion Criteria ..."
            (?:
                [^\n]{{0,60}}{_COLON}     #   any short qualifier, ending in a colon
              | [ \t]*[.\-–—]?  #   or nothing but trailing punctuation
              | (?:[ \t]+(?!{_VERBS}\b)[\w(),/&'-]+){{1,4}}   # or <=4 verb-free words
            )
          | [ \t]*{_COLON}                # or the bare "Inclusion:" form
        )
        [ \t]*{_COLON}?[ \t]*[.\-]?[ \t]*\**[ \t]*
        $""",
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)

# Unanchored fallback for the minority that run headers inline
# ("Inclusion Criteria: healthy  Exclusion Criteria: not healthy").
# A colon is REQUIRED here: without it, an ordinary sentence such as
# "patients meeting any exclusion criteria will be withdrawn" would be
# mistaken for a section break and split the document in the wrong place.
_HEADER_INLINE = re.compile(
    r"(?:key\s+)?(?P<kind>inclusion|exclusion)(?:\s+criteria)?\s*:",
    re.IGNORECASE,
)


def _find_headers(text: str) -> List[tuple[int, int, CriterionType]]:
    """Locate section headers as ``(start, end, kind)``, in document order.

    Prefers headers on their own line; falls back to the inline form only when
    the anchored pass finds nothing, so the loose pattern can never override a
    clean document structure.
    """
    matches = list(_HEADER_ANCHORED.finditer(text))
    if not matches:
        matches = list(_HEADER_INLINE.finditer(text))

    return [
        (m.start(), m.end(), m.group("kind").lower())  # type: ignore[misc]
        for m in matches
    ]


# --- criterion splitting ----------------------------------------------------

# A bullet or number starting a line. Requires following whitespace so that a
# decimal ("2.5 mg/dL") or a hyphenated range at line start is not mistaken for
# a list marker.
_BULLET_SPLIT = re.compile(r"^[ \t]*(?:[*\-•]|\d+[.)])[ \t]+", re.MULTILINE)

_BLANK_LINE_SPLIT = re.compile(r"\n[ \t]*\n+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")
_WHITESPACE = re.compile(r"\s+")


def _split_block(block: str, max_chars: int) -> List[str]:
    """Break one section's body into individual criteria.

    Tries progressively looser strategies, because the corpus is inconsistent:
    bullets (82% of trials), then blank-line paragraphs, then single newlines,
    then sentences for any remaining oversized run-on block.
    """
    block = block.strip()
    if not block:
        return []

    # 1. Bullets. `split` on a leading-marker pattern yields an empty or
    #    preamble first element ("Patients must meet all of:"), which we keep --
    #    it is either dropped later as too short, or is real context.
    if len(_BULLET_SPLIT.findall(block)) >= 2:
        parts = _BULLET_SPLIT.split(block)
    # 2. Paragraphs separated by blank lines.
    elif len(_BLANK_LINE_SPLIT.findall(block)) >= 1:
        parts = _BLANK_LINE_SPLIT.split(block)
    # 3. Plain newlines.
    elif "\n" in block:
        parts = block.split("\n")
    else:
        parts = [block]

    # 4. Anything still oversized is a run-on; fall back to sentences.
    out: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            out.extend(s for s in _SENTENCE_SPLIT.split(part) if s.strip())
        else:
            out.append(part)
    return out


def _normalise(text: str) -> str:
    """Collapse internal whitespace and trim stray list punctuation."""
    return _WHITESPACE.sub(" ", text).strip(" \t\n-*•")


_HAS_LETTER = re.compile(r"[A-Za-z]")


def _is_junk(text: str, min_chars: int) -> bool:
    """Reject non-criteria by shape rather than by length.

    Three things get dropped, and nothing else:

    * anything shorter than ``min_chars`` (stray punctuation, "A)");
    * anything with no letters at all ("18+", "1.", "---");
    * anything ending in a colon, which is always a sub-heading introducing the
      items below it ("Part 1:", "Cohort A:", "For all participants:"). The
      items themselves are captured as their own chunks, so nothing is lost.

    A short criterion with real words -- "Pregnancy", "Prisoners" -- survives.
    """
    if len(text) < min_chars:
        return True
    if not _HAS_LETTER.search(text):
        return True
    if text.endswith(":"):
        return True
    return False


def split_eligibility(
    raw: Optional[str],
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> List[Criterion]:
    """Parse raw eligibility text into tagged, per-criterion chunks.

    Text appearing before any header -- or in a trial with no headers at all --
    is tagged ``unspecified`` rather than guessed at. Roughly 5% of trials land
    here, and quietly calling them "inclusion" would inject false facts into
    the index.

    ``chunk_index`` restarts at 0 for each criterion type, matching the
    ``UNIQUE (nct_id, criterion_type, chunk_index)`` constraint on the table.
    """
    if not raw or not raw.strip():
        return []

    text = unescape(raw)
    headers = _find_headers(text)

    # Build (kind, body) sections from the header positions.
    sections: List[tuple[CriterionType, str]] = []
    if not headers:
        sections.append(("unspecified", text))
    else:
        preamble = text[: headers[0][0]].strip()
        if preamble:
            sections.append(("unspecified", preamble))
        for i, (_start, end, kind) in enumerate(headers):
            body_end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
            sections.append((kind, text[end:body_end]))

    # Split each section and number the results per type.
    counters: dict[str, int] = {}
    criteria: List[Criterion] = []
    for kind, body in sections:
        for piece in _split_block(body, max_chars):
            cleaned = _normalise(piece)
            if _is_junk(cleaned, min_chars):
                continue
            if len(cleaned) > max_chars:
                cleaned = cleaned[:max_chars].rsplit(" ", 1)[0]
            idx = counters.get(kind, 0)
            counters[kind] = idx + 1
            criteria.append(Criterion(kind, idx, cleaned))  # type: ignore[arg-type]

    return criteria
