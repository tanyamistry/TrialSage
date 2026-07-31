"""Verify the database-level guarantees the text-to-SQL agent depends on.

This is the layer that has to hold when everything above it fails. Phase 2 adds
a SQL parser that rejects non-SELECT statements, but a parser can be fooled;
these grants cannot. If any of these tests fail, the SQL agent is unsafe to
enable regardless of how good its prompt is.

Skipped automatically when the database is not running.
"""

import psycopg
import pytest

from trialsage.config import readonly_dsn

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def ro_conn():
    try:
        conn = psycopg.connect(readonly_dsn(), autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as exc:
        pytest.skip(f"database not reachable: {exc}")
    yield conn
    conn.close()


def _fails(conn, sql: str) -> str:
    """Run `sql` expecting it to be refused; return the error message."""
    with pytest.raises(psycopg.Error) as excinfo:
        conn.execute(sql)
    return str(excinfo.value)


class TestAllowed:
    def test_can_select_from_whitelisted_views(self, ro_conn):
        (count,) = ro_conn.execute("SELECT count(*) FROM v_trials").fetchone()
        assert count > 0
        ro_conn.execute("SELECT nct_id FROM v_trial_sites LIMIT 1").fetchone()


class TestWritesBlocked:
    @pytest.mark.parametrize(
        "sql",
        [
            "INSERT INTO trials (nct_id) VALUES ('NCT99999999')",
            "UPDATE trials SET brief_title = 'x'",
            "DELETE FROM trials",
            "DROP TABLE trials",
            "TRUNCATE trials",
            "ALTER TABLE trials ADD COLUMN pwned text",
            "CREATE TABLE pwned (id int)",
            "GRANT ALL ON trials TO trialsage_ro",
        ],
    )
    def test_write_statements_are_refused(self, ro_conn, sql):
        _fails(ro_conn, sql)

    def test_base_tables_are_not_readable(self, ro_conn):
        """Only the two views are granted; the underlying tables are invisible."""
        message = _fails(ro_conn, "SELECT * FROM trials LIMIT 1")
        assert "permission denied" in message.lower()

    def test_stacked_statement_cannot_smuggle_a_write(self, ro_conn):
        """A guard that only inspects the first statement would miss this one."""
        _fails(ro_conn, "SELECT 1; DELETE FROM trials")
        # And the data is still there.
        (count,) = ro_conn.execute("SELECT count(*) FROM v_trials").fetchone()
        assert count > 0


class TestSessionDefaults:
    def test_transactions_are_read_only_by_default(self, ro_conn):
        (value,) = ro_conn.execute("SHOW default_transaction_read_only").fetchone()
        assert value == "on"

    def test_statement_timeout_is_set(self, ro_conn):
        (value,) = ro_conn.execute("SHOW statement_timeout").fetchone()
        assert value not in ("0", "")
