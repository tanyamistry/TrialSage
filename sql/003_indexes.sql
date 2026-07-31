-- Indexes.
--
-- GIN on the array columns is what makes `'PHASE2' = ANY(phases)` fast; note
-- that the planner needs the containment form (`phases @> ARRAY['PHASE2']`) to
-- use it, so the SQL agent's prompt prefers that spelling. At 41k rows either
-- form is fine, but the index keeps it honest as the corpus grows.
--
-- REQUIRED: a materialized view needs at least one UNIQUE index to support
-- REFRESH ... CONCURRENTLY, which is how the ingest job avoids locking the
-- view out from under a running query.

CREATE UNIQUE INDEX IF NOT EXISTS v_trials_pk         ON v_trials (nct_id);

CREATE INDEX IF NOT EXISTS v_trials_status            ON v_trials (overall_status);
CREATE INDEX IF NOT EXISTS v_trials_start_year        ON v_trials (start_year);
CREATE INDEX IF NOT EXISTS v_trials_start_date        ON v_trials (start_date);
CREATE INDEX IF NOT EXISTS v_trials_enrollment        ON v_trials (enrollment_count);

CREATE INDEX IF NOT EXISTS v_trials_phases_gin        ON v_trials USING gin (phases);
CREATE INDEX IF NOT EXISTS v_trials_areas_gin         ON v_trials USING gin (therapeutic_areas);
CREATE INDEX IF NOT EXISTS v_trials_conditions_gin    ON v_trials USING gin (conditions);
CREATE INDEX IF NOT EXISTS v_trials_states_gin        ON v_trials USING gin (states);
CREATE INDEX IF NOT EXISTS v_trials_countries_gin     ON v_trials USING gin (countries);

CREATE INDEX IF NOT EXISTS trial_locations_state      ON trial_locations (state);
CREATE INDEX IF NOT EXISTS trial_locations_nct        ON trial_locations (nct_id);
CREATE INDEX IF NOT EXISTS elig_chunks_nct            ON eligibility_chunks (nct_id);
CREATE INDEX IF NOT EXISTS elig_chunks_type           ON eligibility_chunks (criterion_type);

-- The pgvector HNSW index is deliberately NOT created here. Building it before
-- the embeddings exist is wasted work, and it is far faster to build once over
-- a fully populated table. Phase 2 creates it after the embedding run.
