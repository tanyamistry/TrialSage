-- The database-level half of the text-to-SQL safety story.
--
-- There are three independent layers guarding the SQL agent, because prompt
-- instructions ("only write SELECTs") are not a security control -- they are a
-- suggestion that a jailbreak or a confused model will ignore. This file is the
-- layer that holds even if every layer above it is bypassed:
--
--   1. THIS FILE   -- trialsage_ro has no INSERT/UPDATE/DELETE grant anywhere,
--                     and can see only the two whitelisted views. A DROP or
--                     DELETE fails inside the engine.
--   2. guard.py    -- parses the generated SQL and rejects anything that is not
--                     a single SELECT over whitelisted relations (Phase 2).
--   3. session     -- read-only transactions + a statement timeout, set below
--                     as role defaults so they apply on every connection.
--
-- Expects a psql variable:  -v ro_password='...'   (the Makefile passes it.)

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'trialsage_ro') THEN
        CREATE ROLE trialsage_ro LOGIN;
    END IF;
END
$$;

-- Set the password from the variable without ever interpolating it as raw SQL.
SELECT format('ALTER ROLE trialsage_ro WITH PASSWORD %L', :'ro_password') \gexec

-- Belt and braces: strip anything this role may have picked up previously.
REVOKE ALL ON ALL TABLES    IN SCHEMA public FROM trialsage_ro;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM trialsage_ro;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA public FROM trialsage_ro;
REVOKE CREATE ON SCHEMA public FROM trialsage_ro;

SELECT format('GRANT CONNECT ON DATABASE %I TO trialsage_ro', current_database()) \gexec
GRANT USAGE ON SCHEMA public TO trialsage_ro;

-- The entire allowed surface: two views, SELECT only.
GRANT SELECT ON v_trials      TO trialsage_ro;
GRANT SELECT ON v_trial_sites TO trialsage_ro;

-- Applied automatically to every session this role opens.
ALTER ROLE trialsage_ro SET default_transaction_read_only = on;
ALTER ROLE trialsage_ro SET statement_timeout = '15s';
-- Never let a runaway agent query monopolise a connection slot.
ALTER ROLE trialsage_ro SET idle_in_transaction_session_timeout = '30s';
