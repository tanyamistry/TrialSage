"""Normalise ClinicalTrials.gov age strings into a numeric count of years.

The registry stores ages as free text with a unit: "18 Years", "6 Months",
"30 Days". Some trials state no bound at all, in which case the field is either
absent or the literal "N/A". To filter on age in SQL we need one comparable
numeric column, so everything is converted to years.

Two things this module is careful about:

* **Absent is not zero.** A trial with no stated minimum age returns ``None``,
  which becomes SQL ``NULL``. If we returned 0.0 instead, "trials open to
  newborns" would match every trial that simply never specified a lower bound.
* **Zero is a real value.** "0 Years" genuinely means birth, and appears in
  neonatal trials. So the function must distinguish ``None`` from ``0.0`` --
  which is why callers must use ``is None`` and never a plain truthiness check.
"""

from __future__ import annotations

import re
from typing import Optional

# Days per year and per month use the Gregorian average so that round-trips
# stay sane; the precision does not matter clinically but consistency does.
_DAYS_PER_YEAR = 365.25

_UNIT_TO_YEARS = {
    "year": 1.0,
    "month": 1.0 / 12.0,
    "week": 7.0 / _DAYS_PER_YEAR,
    "day": 1.0 / _DAYS_PER_YEAR,
    "hour": 1.0 / (_DAYS_PER_YEAR * 24),
    "minute": 1.0 / (_DAYS_PER_YEAR * 24 * 60),
}

# Values that explicitly mean "no bound stated".
_NULL_TOKENS = {"", "n/a", "na", "none", "not applicable", "-", "--"}

_AGE_RE = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z]+)?\s*$"
)


def parse_age_to_years(raw: Optional[str]) -> Optional[float]:
    """Convert an age string such as ``"6 Months"`` to years as a float.

    Returns ``None`` when no age was stated or the value cannot be understood.

    >>> parse_age_to_years("18 Years")
    18.0
    >>> parse_age_to_years("6 Months")
    0.5
    >>> parse_age_to_years("N/A") is None
    True
    """
    if raw is None:
        return None

    text = str(raw).strip()
    if text.lower() in _NULL_TOKENS:
        return None

    match = _AGE_RE.match(text)
    if not match:
        return None

    value = float(match.group("value"))
    unit_raw = match.group("unit")

    # A bare number ("18") is not something the API emits today, but tolerate it
    # rather than dropping the value -- years is the only sensible reading.
    if unit_raw is None:
        return round(value, 3)

    # Strip the plural: the API uses "Years"/"Months" but "1 Year" singular.
    unit = unit_raw.lower().rstrip("s")
    factor = _UNIT_TO_YEARS.get(unit)
    if factor is None:
        return None

    return round(value * factor, 3)


def describe_age(years: Optional[float]) -> str:
    """Render a normalised age back to something readable, for UI and logs."""
    if years is None:
        return "not stated"
    if years == 0:
        return "birth"
    if years < 1:
        months = years * 12
        return f"{months:.0f} months" if months >= 1 else f"{years * _DAYS_PER_YEAR:.0f} days"
    return f"{years:g} years"
