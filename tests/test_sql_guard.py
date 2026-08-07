"""Adversarial tests for the generated-SQL guard.

This is a security boundary, so the tests are written as attacks rather than as
happy-path coverage. Anything that gets past the guard reaches the database
with whatever privileges the read-only role has -- so the two layers are tested
independently (see also tests/test_readonly_role.py).
"""

import pytest

from trialsage.retrieval.guard import validate_sql

WHITELIST = ["v_trials", "v_trial_sites"]


def check(sql, row_limit=500):
    return validate_sql(sql, whitelist=WHITELIST, row_limit=row_limit)


class TestAllowsLegitimateQueries:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT count(*) FROM v_trials WHERE start_year = 2024",
            "SELECT nct_id, brief_title FROM v_trials WHERE phases @> ARRAY['PHASE3']",
            "SELECT nct_id FROM v_trials WHERE 'Massachusetts' = ANY(states)",
            "SELECT t.nct_id, s.city FROM v_trials t JOIN v_trial_sites s USING (nct_id)",
            "SELECT overall_status, count(*) FROM v_trials GROUP BY 1 ORDER BY 2 DESC",
            "WITH ph3 AS (SELECT nct_id FROM v_trials WHERE phases @> ARRAY['PHASE3'])"
            " SELECT count(*) FROM ph3",
            "SELECT * FROM v_trials UNION SELECT * FROM v_trials",
        ],
    )
    def test_accepted(self, sql):
        result = check(sql)
        assert result.ok, result.reason

    def test_cte_name_is_not_mistaken_for_a_table(self):
        result = check("WITH x AS (SELECT nct_id FROM v_trials) SELECT * FROM x")
        assert result.ok, result.reason

    def test_strips_markdown_fences(self):
        """Small models wrap output in ```sql fences unprompted."""
        result = check("```sql\nSELECT count(*) FROM v_trials\n```")
        assert result.ok, result.reason
        assert "```" not in result.sql

    def test_tolerates_trailing_semicolon(self):
        assert check("SELECT count(*) FROM v_trials;").ok


class TestBlocksWrites:
    @pytest.mark.parametrize(
        "sql",
        [
            "INSERT INTO trials (nct_id) VALUES ('x')",
            "UPDATE v_trials SET brief_title = 'x'",
            "DELETE FROM v_trials",
            "DROP TABLE trials",
            "TRUNCATE trials",
            "ALTER TABLE trials ADD COLUMN pwned text",
            "CREATE TABLE pwned (id int)",
            "GRANT ALL ON v_trials TO public",
            "CREATE VIEW pwned AS SELECT * FROM v_trials",
        ],
    )
    def test_rejected(self, sql):
        result = check(sql)
        assert not result.ok
        assert result.sql is None


class TestBlocksEvasion:
    def test_stacked_statement(self):
        """The classic: a valid SELECT followed by something destructive."""
        result = check("SELECT 1 FROM v_trials; DROP TABLE trials")
        assert not result.ok
        assert "1 statement" in result.reason

    def test_select_into_creates_a_table(self):
        """Parses as a Select, so a root-type check alone would allow it."""
        result = check("SELECT * INTO evil FROM v_trials")
        assert not result.ok

    def test_comment_hidden_second_statement(self):
        result = check("SELECT count(*) FROM v_trials -- \n; DELETE FROM trials")
        assert not result.ok

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM trials",
            "SELECT * FROM eligibility_chunks",
            "SELECT * FROM pg_catalog.pg_tables",
            "SELECT * FROM information_schema.tables",
            "SELECT * FROM pg_shadow",
        ],
    )
    def test_non_whitelisted_relations(self, sql):
        result = check(sql)
        assert not result.ok
        assert "not allowed" in result.reason or "not accessible" in result.reason

    def test_whitelisted_table_in_subquery_is_fine_but_others_are_not(self):
        assert check("SELECT * FROM v_trials WHERE nct_id IN"
                     " (SELECT nct_id FROM v_trial_sites)").ok
        assert not check("SELECT * FROM v_trials WHERE nct_id IN"
                         " (SELECT nct_id FROM trials)").ok

    def test_cte_cannot_launder_a_forbidden_table(self):
        result = check("WITH x AS (SELECT * FROM trials) SELECT * FROM x")
        assert not result.ok

    @pytest.mark.parametrize("fn", ["pg_sleep(10)", "pg_read_file('/etc/passwd')"])
    def test_dangerous_functions(self, fn):
        result = check(f"SELECT {fn} FROM v_trials")
        assert not result.ok
        assert "forbidden function" in result.reason

    def test_case_and_whitespace_do_not_bypass(self):
        assert not check("dElEtE   FROM    v_trials").ok
        assert not check("SeLeCt * FrOm TrIaLs").ok

    def test_garbage_is_rejected_not_crashed(self):
        for junk in ["", "   ", "not sql at all", "SELECT FROM WHERE", "'; --"]:
            result = check(junk)
            assert not result.ok
            assert result.reason


class TestRowCap:
    def test_limit_injected_when_absent(self):
        result = check("SELECT nct_id FROM v_trials")
        assert result.ok
        assert "LIMIT 500" in result.sql.upper()

    def test_oversized_limit_is_reduced(self):
        result = check("SELECT nct_id FROM v_trials LIMIT 100000")
        assert result.ok
        assert "LIMIT 500" in result.sql.upper()
        assert "100000" not in result.sql

    def test_smaller_limit_is_preserved(self):
        result = check("SELECT nct_id FROM v_trials LIMIT 10")
        assert result.ok
        assert "LIMIT 10" in result.sql.upper()

    def test_custom_cap_is_respected(self):
        result = check("SELECT nct_id FROM v_trials", row_limit=25)
        assert "LIMIT 25" in result.sql.upper()

    def test_aggregate_query_still_gets_a_cap(self):
        """A COUNT returns one row, but capping costs nothing and keeps the
        rule simple -- no special-casing means no gap to exploit."""
        result = check("SELECT count(*) FROM v_trials")
        assert result.ok
        assert "LIMIT" in result.sql.upper()


class TestGuardResult:
    def test_is_falsy_when_rejected(self):
        assert not check("DROP TABLE trials")

    def test_is_truthy_when_accepted(self):
        assert check("SELECT 1 FROM v_trials")

    def test_rejection_always_explains_why(self):
        for sql in ["DROP TABLE trials", "SELECT * FROM trials", "SELECT 1; SELECT 2"]:
            result = check(sql)
            assert result.reason, f"no reason given for {sql!r}"
