"""Validate and rewrite LLM-generated SQL before it reaches the database.

This is security layer 2 of 3 (see sql/004_readonly_role.sql for the full
picture). It exists because a language model told "only write SELECT" will
sometimes write something else -- through confusion, or because a user asked it
to. Prompt instructions are not a security control.

The approach is a parse-and-inspect allowlist, not a regex or a keyword
blocklist. Keyword filtering is trivially defeated (comments, casing,
whitespace, string literals containing keywords); parsing the statement into a
syntax tree and rejecting anything that is not a plain SELECT over approved
relations is not.

Rejections are returned rather than raised, because the SQL agent feeds the
reason back to the model for a repair attempt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Set

import sqlglot
from sqlglot import exp

# Node types that must never appear anywhere in the tree. `Into` is the
# subtle one: `SELECT * INTO evil FROM v_trials` parses as a Select, so a
# root-type check alone would wave it through.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.Grant, exp.TruncateTable, exp.Into, exp.Merge,
    # `Command` is sqlglot's catch-all for statements it does not model
    # (VACUUM, SET, CALL, COPY, ...). Anything landing here is not a SELECT.
    exp.Command,
)

# Functions that read the filesystem, sleep, open connections, or otherwise do
# something other than return data. Most already require privileges the
# read-only role lacks, but defence in depth is cheap here.
_FORBIDDEN_FUNCTIONS: Set[str] = {
    "pg_sleep", "pg_read_file", "pg_read_binary_file", "pg_ls_dir",
    "pg_stat_file", "lo_import", "lo_export", "dblink", "dblink_exec",
    "query_to_xml", "set_config", "pg_terminate_backend", "pg_cancel_backend",
    "pg_reload_conf", "pg_logical_emit_message", "copy_from", "pg_file_write",
}

_ALLOWED_SCHEMAS = {"", "public"}


@dataclass
class GuardResult:
    """Outcome of validation. ``sql`` is the rewritten statement when ``ok``."""

    ok: bool
    sql: Optional[str] = None
    reason: Optional[str] = None

    def __bool__(self) -> bool:
        return self.ok


def _strip_fences(sql: str) -> str:
    """Remove markdown code fences, which small models add unprompted."""
    text = sql.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]                      # drop ```sql
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text.rstrip().rstrip(";").strip()


def validate_sql(
    sql: str,
    *,
    whitelist: Iterable[str],
    row_limit: int = 500,
) -> GuardResult:
    """Check a generated statement and return it rewritten with a row cap.

    Enforced, in order:

    1. exactly one statement (blocks ``SELECT 1; DROP TABLE trials``);
    2. the root is a SELECT or a UNION of SELECTs;
    3. no forbidden node type anywhere in the tree;
    4. every table reference is whitelisted, ignoring CTE names;
    5. no schema qualifier other than ``public`` (blocks ``pg_catalog``);
    6. no forbidden function call;
    7. a LIMIT is present and no larger than ``row_limit``.
    """
    allowed = {name.lower() for name in whitelist}
    cleaned = _strip_fences(sql)

    if not cleaned:
        return GuardResult(False, reason="empty statement")

    # 1. Exactly one statement.
    try:
        statements = [s for s in sqlglot.parse(cleaned, dialect="postgres") if s is not None]
    except Exception as exc:  # sqlglot raises several types; treat all as invalid
        return GuardResult(False, reason=f"could not parse SQL: {exc}")

    if not statements:
        return GuardResult(False, reason="no statement found")
    if len(statements) > 1:
        return GuardResult(
            False,
            reason=f"expected 1 statement, found {len(statements)}; "
                   "multiple statements are not allowed",
        )

    tree = statements[0]

    # 2. Root must be a query.
    if not isinstance(tree, (exp.Select, exp.Union)):
        return GuardResult(
            False,
            reason=f"only SELECT is allowed, got {type(tree).__name__.upper()}",
        )

    # 3. Forbidden node types anywhere in the tree.
    for node_type in _FORBIDDEN_NODES:
        found = tree.find(node_type)
        if found is not None:
            return GuardResult(
                False,
                reason=f"forbidden operation: {node_type.__name__.upper()}",
            )

    # 4/5. Table references.
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        schema = (table.db or "").lower()
        if name in cte_names and not schema:
            continue
        if schema not in _ALLOWED_SCHEMAS:
            return GuardResult(
                False,
                reason=f"schema '{schema}' is not accessible; "
                       f"query only: {', '.join(sorted(allowed))}",
            )
        if name not in allowed:
            return GuardResult(
                False,
                reason=f"table '{name}' is not allowed; "
                       f"query only: {', '.join(sorted(allowed))}",
            )

    # 6. Function calls.
    for func in tree.find_all(exp.Anonymous):
        fname = (func.this or "").lower() if isinstance(func.this, str) else ""
        if fname in _FORBIDDEN_FUNCTIONS:
            return GuardResult(False, reason=f"forbidden function: {fname}")

    # 7. Row cap. Replace any existing limit that is missing or too large.
    existing = tree.args.get("limit")
    needs_limit = True
    if existing is not None:
        try:
            if int(existing.expression.name) <= row_limit:
                needs_limit = False
            else:
                tree.set("limit", None)
        except (AttributeError, ValueError):
            tree.set("limit", None)   # non-literal LIMIT; replace with our own

    if needs_limit:
        tree = tree.limit(row_limit)

    return GuardResult(True, sql=tree.sql(dialect="postgres", pretty=False))


def whitelist_from_config() -> Sequence[str]:
    from ..config import settings
    return settings()["sql_whitelist"]
