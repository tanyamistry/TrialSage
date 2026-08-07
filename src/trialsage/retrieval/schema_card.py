"""Build the schema description handed to the text-to-SQL model.

Two deliberate choices here, both aimed at making a small local model succeed:

**Generated from the live database, not hand-written.** Column names and types
come from ``information_schema``, so the card cannot drift out of sync with the
schema. A hand-maintained prompt silently rots the moment anyone adds a column.

**Includes real enum values.** The single most common text-to-SQL failure is
inventing a value that looks plausible but does not exist -- ``'Recruiting'``
instead of ``'RECRUITING'``, ``'Phase 3'`` instead of ``'PHASE3'``. Those
queries run fine and return zero rows, which is far worse than an error because
it looks like a real answer. We query the actual distinct values and put them in
the prompt.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from ..db import connect

# Columns worth describing beyond their name and type. Kept short: a small
# model degrades when the prompt gets long.
_COLUMN_NOTES = {
    "phases": "array, e.g. {PHASE2,PHASE3}. Use the @> operator.",
    "therapeutic_areas": "array: oncology, diabetes, cardiovascular",
    "conditions": "array of free-text condition names",
    "states": "array of US states with a trial site (any status)",
    "recruiting_states": "array of states where that SITE is recruiting",
    "countries": "array of countries with a site",
    "intervention_names": "array of drug/procedure names",
    "start_year": "integer, convenient shorthand for the start date year",
    "is_recruiting": "boolean, true when overall_status = 'RECRUITING'",
    "min_age_years": "numeric years; NULL means no minimum was stated",
    "max_age_years": "numeric years; NULL means no maximum was stated",
    "eligibility_text": "full raw eligibility criteria, unstructured",
    "n_sites": "number of trial sites",
}

_GUIDANCE = """\
Rules:
- PostgreSQL dialect. Output ONE SELECT statement and nothing else.
- Query only the views listed above. Never reference any other table.
- Array columns use the containment operator: phases @> ARRAY['PHASE3'].
  Do NOT use = or IN on an array column.
- CASING: only overall_status, phases and sex values are UPPERCASE.
  Every other text value uses normal capitalisation:
  'California', 'United States', 'Novo Nordisk A/S', 'diabetes'.
  Never uppercase a state, country, sponsor or condition name.
- COUNTING TRIALS: always count v_trials, which has one row per trial.
  v_trial_sites has one row per SITE, so a trial with 40 sites appears
  40 times -- counting it gives the wrong answer.
- For a location question, prefer v_trials with the states array:
  WHERE 'California' = ANY(states)
- Use ONLY the conditions the question actually states. Do not add extra
  filters (such as a status) that the question did not ask for.
- For "how many", use SELECT count(*).
- For "which trials", select nct_id and brief_title.
- Do not invent columns. Only use the ones listed."""

_EXAMPLES = """\
Examples:

Q: How many phase 3 diabetes trials started in 2024?
A: SELECT count(*) FROM v_trials WHERE phases @> ARRAY['PHASE3'] AND therapeutic_areas @> ARRAY['diabetes'] AND start_year = 2024

Q: List recruiting phase 2 oncology trials in Massachusetts.
A: SELECT nct_id, brief_title FROM v_trials WHERE overall_status = 'RECRUITING' AND phases @> ARRAY['PHASE2'] AND therapeutic_areas @> ARRAY['oncology'] AND 'Massachusetts' = ANY(states)

Q: What is the average enrollment of completed phase 3 trials?
A: SELECT avg(enrollment_count) FROM v_trials WHERE overall_status = 'COMPLETED' AND phases @> ARRAY['PHASE3']

Q: Which sponsors run the most cardiovascular trials?
A: SELECT lead_sponsor, count(*) FROM v_trials WHERE therapeutic_areas @> ARRAY['cardiovascular'] GROUP BY lead_sponsor ORDER BY count(*) DESC

Q: How many trials have a site in Texas?
A: SELECT count(*) FROM v_trials WHERE 'Texas' = ANY(states)

Q: How many trials enrol female participants only?
A: SELECT count(*) FROM v_trials WHERE sex = 'FEMALE'"""


def _columns(conn, view: str) -> List[tuple[str, str]]:
    """Column names and types for a table, view, or materialized view.

    Reads pg_catalog rather than information_schema on purpose: PostgreSQL
    does NOT list materialized views in information_schema.columns, so
    v_trials -- our main view -- comes back empty from there. That failure is
    silent, which is the dangerous part: the card renders fine, just without
    the most important table in it.
    """
    return conn.execute(
        """
        SELECT a.attname, format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relname = %s
          AND n.nspname = 'public'
          AND c.relkind IN ('r', 'v', 'm')   -- table, view, materialized view
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY a.attnum
        """,
        (view,),
    ).fetchall()


def _distinct(conn, sql: str, limit: int = 40) -> List[str]:
    return [str(r[0]) for r in conn.execute(sql).fetchall()[:limit] if r[0] is not None]


def _format_type(data_type: str) -> str:
    """Shorten Postgres type names so the prompt stays compact."""
    return {
        "character varying": "text",
        "timestamp with time zone": "timestamptz",
        "double precision": "float",
    }.get(data_type, data_type)


@lru_cache(maxsize=1)
def build_schema_card() -> str:
    """Return the schema description, built once per process from the database."""
    with connect(autocommit=True) as conn:
        parts: List[str] = ["Database schema (PostgreSQL). You may query ONLY these views.\n"]

        for view in ("v_trials", "v_trial_sites"):
            cols = _columns(conn, view)
            if not cols:
                continue
            parts.append(f"TABLE {view}:")
            for name, dtype in cols:
                note = _COLUMN_NOTES.get(name, "")
                line = f"  {name} ({_format_type(dtype)})"
                if note:
                    line += f"  -- {note}"
                parts.append(line)
            parts.append("")

        statuses = _distinct(
            conn, "SELECT DISTINCT overall_status FROM v_trials ORDER BY 1")
        phases = _distinct(
            conn, "SELECT DISTINCT unnest(phases) FROM v_trials ORDER BY 1")
        areas = _distinct(
            conn, "SELECT DISTINCT unnest(therapeutic_areas) FROM v_trials ORDER BY 1")

        sexes = _distinct(conn, "SELECT DISTINCT sex FROM v_trials ORDER BY 1")

        parts.append("Exact allowed values (case-sensitive):")
        parts.append(f"  overall_status: {', '.join(statuses)}")
        parts.append(f"  phases elements: {', '.join(phases)}")
        parts.append(f"  therapeutic_areas elements: {', '.join(areas)}")
        parts.append(f"  sex: {', '.join(sexes)}")
        parts.append("  states / countries / lead_sponsor: normal capitalisation,"
                     " e.g. 'California', 'United States'")
        parts.append("")
        parts.append(_GUIDANCE)
        parts.append("")
        parts.append(_EXAMPLES)

    return "\n".join(parts)


if __name__ == "__main__":  # pragma: no cover - manual inspection
    print(build_schema_card())
