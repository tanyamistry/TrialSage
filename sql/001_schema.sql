-- TrialSage base tables.
--
-- Design note: ClinicalTrials.gov gives us several genuinely multi-valued
-- fields -- a trial can be tagged PHASE1/PHASE2, list six conditions, and run
-- at 300 sites. Those cannot be columns, so each gets a child table here.
-- Phase 2 (sql/002_views.sql) flattens them back into arrays for the
-- text-to-SQL agent, which does much better with one wide table than joins.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS trials (
    nct_id                  text PRIMARY KEY,
    brief_title             text,
    official_title          text,
    brief_summary           text,

    study_type              text,
    overall_status          text,
    why_stopped             text,

    -- Human-readable join of the phases list, e.g. 'PHASE1/PHASE2'.
    -- Use the trial_phases child table (or v_trials.phases) to filter.
    phase_display           text,

    enrollment_count        integer,
    enrollment_type         text,          -- ACTUAL | ESTIMATED

    -- The API returns either '2024-03-15' or just '2024-03'. We always store a
    -- real date (missing day => the 1st) and keep the precision alongside so
    -- nothing downstream mistakes an imputed day for a reported one.
    start_date              date,
    start_date_precision    text CHECK (start_date_precision IN ('day','month','year')),
    primary_completion_date date,
    completion_date         date,
    last_update_posted      date,

    lead_sponsor            text,
    sponsor_class           text,

    sex                     text,
    healthy_volunteers      boolean,

    -- Normalised to years so they are numerically filterable.
    -- NULL means the trial did not state a bound -- which is common, and is
    -- NOT the same as zero or as "no age limit".
    min_age_years           numeric(6,3),
    max_age_years           numeric(6,3),
    min_age_raw             text,          -- kept for auditing the parser
    max_age_raw             text,

    eligibility_raw         text,
    ingested_at             timestamptz NOT NULL DEFAULT now()
);

-- A trial can legitimately belong to more than one therapeutic area
-- (a diabetic-cardiomyopathy trial is both diabetes and cardiovascular).
CREATE TABLE IF NOT EXISTS trial_areas (
    nct_id  text NOT NULL REFERENCES trials(nct_id) ON DELETE CASCADE,
    area    text NOT NULL,
    PRIMARY KEY (nct_id, area)
);

CREATE TABLE IF NOT EXISTS trial_phases (
    nct_id  text NOT NULL REFERENCES trials(nct_id) ON DELETE CASCADE,
    phase   text NOT NULL,
    PRIMARY KEY (nct_id, phase)
);

CREATE TABLE IF NOT EXISTS trial_conditions (
    nct_id     text NOT NULL REFERENCES trials(nct_id) ON DELETE CASCADE,
    condition  text NOT NULL,
    PRIMARY KEY (nct_id, condition)
);

CREATE TABLE IF NOT EXISTS trial_interventions (
    id                 bigserial PRIMARY KEY,
    nct_id             text NOT NULL REFERENCES trials(nct_id) ON DELETE CASCADE,
    intervention_type  text,
    name               text
);

CREATE TABLE IF NOT EXISTS trial_locations (
    id               bigserial PRIMARY KEY,
    nct_id           text NOT NULL REFERENCES trials(nct_id) ON DELETE CASCADE,
    facility         text,
    city             text,
    state            text,
    zip              text,
    country          text,
    -- Per-site recruiting status. A trial can be RECRUITING overall while an
    -- individual site is closed, so we keep this to allow the stricter
    -- "actually recruiting at this location" reading later.
    location_status  text
);

-- One row per individual eligibility criterion (not per fixed token window).
-- criterion_type is the whole point: "includes patients with X" and "excludes
-- patients with X" are opposite facts and must never be conflated.
CREATE TABLE IF NOT EXISTS eligibility_chunks (
    chunk_id        bigserial PRIMARY KEY,
    nct_id          text NOT NULL REFERENCES trials(nct_id) ON DELETE CASCADE,
    criterion_type  text NOT NULL CHECK (criterion_type IN ('inclusion','exclusion','unspecified')),
    chunk_index     integer NOT NULL,
    criterion_text  text NOT NULL,
    char_len        integer NOT NULL,
    embedding       vector(384),          -- populated in Phase 2
    UNIQUE (nct_id, criterion_type, chunk_index)
);

-- Records what the ingest job actually did, so a partial or failed run is
-- visible rather than silently leaving a half-loaded table.
CREATE TABLE IF NOT EXISTS ingest_runs (
    id              bigserial PRIMARY KEY,
    area            text NOT NULL,
    started_at      timestamptz NOT NULL DEFAULT now(),
    finished_at     timestamptz,
    trials_fetched  integer,
    trials_loaded   integer,
    chunks_loaded   integer,
    status          text,                 -- running | ok | failed
    notes           text
);
