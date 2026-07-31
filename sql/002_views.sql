-- The surface the text-to-SQL agent is allowed to see.
--
-- Why views rather than the base tables: an 8B local model writing five-table
-- joins hallucinates badly. v_trials collapses every multi-valued field into
-- an array column, so the overwhelming majority of structured questions become
-- single-table, zero-join queries. The agent's prompt teaches one idiom for
-- arrays -- `'PHASE2' = ANY(phases)` -- and that covers phases, conditions,
-- states and countries alike.
--
-- It is also the security boundary: the read-only role is granted SELECT on
-- these two views and nothing else, so the base tables are unreachable.
--
-- MATERIALIZED so the array aggregation is paid once at ingest time and the
-- arrays can be GIN-indexed. sql/003_indexes.sql builds those indexes, and the
-- ingest job refreshes the view when it finishes.

DROP MATERIALIZED VIEW IF EXISTS v_trials CASCADE;

CREATE MATERIALIZED VIEW v_trials AS
SELECT
    t.nct_id,
    t.brief_title,
    t.overall_status,
    (t.overall_status = 'RECRUITING')                AS is_recruiting,
    t.phase_display,
    COALESCE(ph.phases,  ARRAY[]::text[])            AS phases,
    COALESCE(ar.areas,   ARRAY[]::text[])            AS therapeutic_areas,
    COALESCE(co.conds,   ARRAY[]::text[])            AS conditions,
    COALESCE(iv.names,   ARRAY[]::text[])            AS intervention_names,
    COALESCE(lo.countries, ARRAY[]::text[])          AS countries,
    COALESCE(lo.states,    ARRAY[]::text[])          AS states,
    -- States where a site is itself recruiting, for the stricter reading of
    -- "recruiting in <state>". The default reading uses is_recruiting + states.
    COALESCE(lo.recruiting_states, ARRAY[]::text[])  AS recruiting_states,
    COALESCE(lo.n_sites, 0)                          AS n_sites,
    t.enrollment_count,
    t.enrollment_type,
    t.start_date,
    EXTRACT(YEAR FROM t.start_date)::int             AS start_year,
    t.primary_completion_date,
    t.completion_date,
    t.lead_sponsor,
    t.sponsor_class,
    t.sex,
    t.healthy_volunteers,
    t.min_age_years,
    t.max_age_years,
    t.brief_summary,
    t.eligibility_raw                                AS eligibility_text
FROM trials t
LEFT JOIN (SELECT nct_id, array_agg(DISTINCT phase     ORDER BY phase)     AS phases FROM trial_phases     GROUP BY nct_id) ph USING (nct_id)
LEFT JOIN (SELECT nct_id, array_agg(DISTINCT area      ORDER BY area)      AS areas  FROM trial_areas      GROUP BY nct_id) ar USING (nct_id)
LEFT JOIN (SELECT nct_id, array_agg(DISTINCT condition ORDER BY condition) AS conds  FROM trial_conditions GROUP BY nct_id) co USING (nct_id)
LEFT JOIN (SELECT nct_id, array_agg(DISTINCT name      ORDER BY name)      AS names  FROM trial_interventions WHERE name IS NOT NULL GROUP BY nct_id) iv USING (nct_id)
LEFT JOIN (
    SELECT nct_id,
           array_agg(DISTINCT country) FILTER (WHERE country IS NOT NULL) AS countries,
           array_agg(DISTINCT state)   FILTER (WHERE state   IS NOT NULL) AS states,
           array_agg(DISTINCT state)   FILTER (WHERE state   IS NOT NULL
                                                AND location_status = 'RECRUITING') AS recruiting_states,
           count(*)::int AS n_sites
    FROM trial_locations GROUP BY nct_id
) lo USING (nct_id);

-- Site-level detail, for questions that genuinely need one row per location
-- ("which cities", "how many sites in Texas").
CREATE OR REPLACE VIEW v_trial_sites AS
SELECT
    l.nct_id,
    t.brief_title,
    t.overall_status               AS trial_status,
    t.phase_display,
    l.facility,
    l.city,
    l.state,
    l.country,
    l.location_status              AS site_status
FROM trial_locations l
JOIN trials t USING (nct_id);
