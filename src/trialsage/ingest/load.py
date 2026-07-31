"""Load parsed trials into PostgreSQL.

Upsert semantics: re-running ingest for an area updates existing rows rather
than duplicating or failing. Child rows (phases, conditions, sites, criteria)
are deleted and re-inserted for the trials in the batch, which is the simplest
way to stay correct when a trial's site list shrinks between runs.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import psycopg

from .parse import ParsedTrial

BATCH_SIZE = 500

_UPSERT_TRIAL = """
INSERT INTO trials (
    nct_id, brief_title, official_title, brief_summary,
    study_type, overall_status, why_stopped,
    phase_display, enrollment_count, enrollment_type,
    start_date, start_date_precision, primary_completion_date,
    completion_date, last_update_posted,
    lead_sponsor, sponsor_class,
    sex, healthy_volunteers,
    min_age_years, max_age_years, min_age_raw, max_age_raw,
    eligibility_raw
) VALUES (
    %s, %s, %s, %s,
    %s, %s, %s,
    %s, %s, %s,
    %s, %s, %s,
    %s, %s,
    %s, %s,
    %s, %s,
    %s, %s, %s, %s,
    %s
)
ON CONFLICT (nct_id) DO UPDATE SET
    brief_title = EXCLUDED.brief_title,
    official_title = EXCLUDED.official_title,
    brief_summary = EXCLUDED.brief_summary,
    study_type = EXCLUDED.study_type,
    overall_status = EXCLUDED.overall_status,
    why_stopped = EXCLUDED.why_stopped,
    phase_display = EXCLUDED.phase_display,
    enrollment_count = EXCLUDED.enrollment_count,
    enrollment_type = EXCLUDED.enrollment_type,
    start_date = EXCLUDED.start_date,
    start_date_precision = EXCLUDED.start_date_precision,
    primary_completion_date = EXCLUDED.primary_completion_date,
    completion_date = EXCLUDED.completion_date,
    last_update_posted = EXCLUDED.last_update_posted,
    lead_sponsor = EXCLUDED.lead_sponsor,
    sponsor_class = EXCLUDED.sponsor_class,
    sex = EXCLUDED.sex,
    healthy_volunteers = EXCLUDED.healthy_volunteers,
    min_age_years = EXCLUDED.min_age_years,
    max_age_years = EXCLUDED.max_age_years,
    min_age_raw = EXCLUDED.min_age_raw,
    max_age_raw = EXCLUDED.max_age_raw,
    eligibility_raw = EXCLUDED.eligibility_raw,
    ingested_at = now()
"""


def _trial_row(t: ParsedTrial) -> Tuple:
    return (
        t.nct_id, t.brief_title, t.official_title, t.brief_summary,
        t.study_type, t.overall_status, t.why_stopped,
        t.phase_display, t.enrollment_count, t.enrollment_type,
        t.start_date, t.start_date_precision, t.primary_completion_date,
        t.completion_date, t.last_update_posted,
        t.lead_sponsor, t.sponsor_class,
        t.sex, t.healthy_volunteers,
        t.min_age_years, t.max_age_years, t.min_age_raw, t.max_age_raw,
        t.eligibility_raw,
    )


def _replace_children(cur: psycopg.Cursor, area: str, batch: Sequence[ParsedTrial]) -> int:
    """Delete then re-insert all child rows for the trials in this batch."""
    nct_ids = [t.nct_id for t in batch]
    for table in (
        "trial_areas", "trial_phases", "trial_conditions",
        "trial_interventions", "trial_locations", "eligibility_chunks",
    ):
        cur.execute(f"DELETE FROM {table} WHERE nct_id = ANY(%s)", (nct_ids,))

    cur.executemany(
        "INSERT INTO trial_areas (nct_id, area) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        [(t.nct_id, area) for t in batch],
    )
    cur.executemany(
        "INSERT INTO trial_phases (nct_id, phase) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        [(t.nct_id, p) for t in batch for p in t.phases],
    )
    cur.executemany(
        "INSERT INTO trial_conditions (nct_id, condition) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        [(t.nct_id, c) for t in batch for c in t.conditions],
    )
    cur.executemany(
        "INSERT INTO trial_interventions (nct_id, intervention_type, name) VALUES (%s, %s, %s)",
        [(t.nct_id, i.intervention_type, i.name) for t in batch for i in t.interventions],
    )
    cur.executemany(
        "INSERT INTO trial_locations (nct_id, facility, city, state, zip, country, location_status)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s)",
        [
            (t.nct_id, l.facility, l.city, l.state, l.zip, l.country, l.location_status)
            for t in batch for l in t.locations
        ],
    )

    chunk_rows = [
        (t.nct_id, c.criterion_type, c.chunk_index, c.text, c.char_len)
        for t in batch for c in t.criteria
    ]
    cur.executemany(
        "INSERT INTO eligibility_chunks (nct_id, criterion_type, chunk_index, criterion_text, char_len)"
        " VALUES (%s, %s, %s, %s, %s)"
        " ON CONFLICT (nct_id, criterion_type, chunk_index) DO NOTHING",
        chunk_rows,
    )
    return len(chunk_rows)


def load_trials(conn: psycopg.Connection, area: str, trials: Iterable[ParsedTrial]) -> Tuple[int, int]:
    """Insert/update trials in batches. Returns ``(trials_loaded, chunks_loaded)``."""
    loaded = 0
    chunks = 0
    batch: List[ParsedTrial] = []

    def flush() -> None:
        nonlocal loaded, chunks
        if not batch:
            return
        with conn.cursor() as cur:
            cur.executemany(_UPSERT_TRIAL, [_trial_row(t) for t in batch])
            chunks += _replace_children(cur, area, batch)
        conn.commit()
        loaded += len(batch)
        print(f"  loaded {loaded} trials, {chunks} criteria", end="\r", flush=True)
        batch.clear()

    for trial in trials:
        batch.append(trial)
        if len(batch) >= BATCH_SIZE:
            flush()
    flush()
    print(f"  loaded {loaded} trials, {chunks} criteria        ")
    return loaded, chunks


def refresh_views(conn: psycopg.Connection) -> None:
    """Rebuild the materialized view the SQL agent reads.

    Must run after every ingest, otherwise the agent queries stale data. Tries
    CONCURRENTLY first so readers are not blocked; that fails on a view that has
    never been populated, in which case we do a plain refresh.
    """
    conn.commit()
    with conn.cursor() as cur:
        try:
            cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY v_trials")
        except psycopg.errors.ObjectNotInPrerequisiteState:
            conn.rollback()
            cur.execute("REFRESH MATERIALIZED VIEW v_trials")
    conn.commit()
