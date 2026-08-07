"""Text-to-SQL: natural language question -> validated SQL -> rows.

Flow, with a repair loop:

    question -> LLM -> guard -> execute
                 ^                 |
                 |                 v
                 +---- error fed back as feedback

The repair loop matters far more with a small local model than with a hosted
one. An 8B model frequently produces SQL that is *nearly* right -- a made-up
column, `= ANY` where containment was needed -- and a single round of concrete
feedback ("column x does not exist") fixes a large share of those. We feed back
the real Postgres error, not a paraphrase.

Everything runs through the read-only role, so even a query that slips past the
guard cannot modify anything.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psycopg

from ..config import settings
from ..db import connect_readonly
from ..llm import LLMError, get_llm
from .guard import GuardResult, validate_sql
from .schema_card import build_schema_card

_SYSTEM = (
    "You are a PostgreSQL expert. You translate questions into a single SELECT "
    "statement. You reply with SQL only -- no explanation, no markdown, no "
    "commentary. Never write anything except a SELECT."
)

# Small models like to explain themselves despite instructions. This pulls the
# statement out of whatever prose surrounds it.
_SQL_RE = re.compile(r"\b(WITH|SELECT)\b.*", re.IGNORECASE | re.DOTALL)


@dataclass
class SQLAttempt:
    """One generate -> validate -> execute cycle, kept for inspection."""

    raw_output: str
    sql: Optional[str] = None
    guard_reason: Optional[str] = None
    db_error: Optional[str] = None
    ok: bool = False


@dataclass
class SQLResult:
    """Final outcome of the SQL path, including everything needed for tracing."""

    question: str
    ok: bool
    sql: Optional[str] = None
    columns: List[str] = field(default_factory=list)
    rows: List[Tuple[Any, ...]] = field(default_factory=list)
    error: Optional[str] = None
    attempts: List[SQLAttempt] = field(default_factory=list)
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def scalar(self) -> Any:
        """First cell of the first row -- convenient for count(*) questions."""
        return self.rows[0][0] if self.rows and self.rows[0] else None

    def nct_ids(self) -> List[str]:
        """NCT IDs from the result set, for handing to the hybrid retriever."""
        if "nct_id" not in self.columns:
            return []
        idx = self.columns.index("nct_id")
        return [r[idx] for r in self.rows if r[idx]]


def _extract_sql(text: str) -> str:
    """Pull a bare statement out of a chatty model response."""
    cleaned = text.strip()
    if "```" in cleaned:
        blocks = re.findall(r"```(?:sql)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
        if blocks:
            cleaned = blocks[0].strip()
    match = _SQL_RE.search(cleaned)
    if match:
        cleaned = match.group(0)
    # Cut anything after the first statement terminator.
    return cleaned.split(";")[0].strip()


def _build_prompt(question: str, feedback: Optional[str]) -> str:
    parts = [build_schema_card(), "", f"Question: {question}"]
    if feedback:
        parts += [
            "",
            "Your previous attempt failed with this error:",
            f"  {feedback}",
            "Write a corrected query that fixes exactly this problem.",
        ]
    parts += ["", "SQL:"]
    return "\n".join(parts)


def generate_sql(
    question: str,
    *,
    max_attempts: int = 3,
    row_limit: Optional[int] = None,
    execute: bool = True,
) -> SQLResult:
    """Answer a structured question by generating and running SQL."""
    cfg = settings()
    row_limit = row_limit or cfg["retrieval"]["sql_row_limit"]
    whitelist = cfg["sql_whitelist"]

    llm = get_llm("sql")
    result = SQLResult(question=question, ok=False)
    started = time.perf_counter()
    feedback: Optional[str] = None

    for _ in range(max_attempts):
        try:
            response = llm.complete(_build_prompt(question, feedback), system=_SYSTEM)
        except LLMError as exc:
            result.error = str(exc)
            break

        result.prompt_tokens += response.prompt_tokens
        result.completion_tokens += response.completion_tokens

        attempt = SQLAttempt(raw_output=response.text)
        result.attempts.append(attempt)

        candidate = _extract_sql(response.text)
        guard: GuardResult = validate_sql(candidate, whitelist=whitelist, row_limit=row_limit)
        if not guard.ok:
            attempt.guard_reason = guard.reason
            feedback = guard.reason
            continue

        attempt.sql = guard.sql
        if not execute:
            attempt.ok = True
            result.ok, result.sql = True, guard.sql
            break

        try:
            with connect_readonly() as conn:
                cur = conn.execute(guard.sql)
                result.columns = [d.name for d in (cur.description or [])]
                result.rows = cur.fetchall()
            attempt.ok = True
            result.ok, result.sql, result.error = True, guard.sql, None
            break
        except psycopg.Error as exc:
            # The raw Postgres message is the most useful repair signal there
            # is -- "column x does not exist" tells the model exactly what to fix.
            message = str(exc).strip().splitlines()[0]
            attempt.db_error = message
            feedback = message
            result.error = message

    result.latency_s = time.perf_counter() - started
    if not result.ok and not result.error:
        last = result.attempts[-1] if result.attempts else None
        result.error = (last.guard_reason or last.db_error) if last else "no attempts made"
    return result


def explain(result: SQLResult) -> str:
    """Readable trace of what the agent did -- used by the CLI and Phase 5 UI."""
    lines = [f"Q: {result.question}",
             f"   attempts={result.n_attempts} ok={result.ok} "
             f"latency={result.latency_s:.1f}s tokens={result.total_tokens}"]
    for i, attempt in enumerate(result.attempts, 1):
        status = "OK" if attempt.ok else (attempt.guard_reason or attempt.db_error or "failed")
        lines.append(f"   [{i}] {status}")
        if attempt.sql:
            lines.append(f"       {attempt.sql}")
        elif not attempt.ok:
            lines.append(f"       raw: {attempt.raw_output.strip()[:120]!r}")
    if result.ok:
        preview = result.rows[:3]
        lines.append(f"   columns={result.columns} rows={len(result.rows)}")
        for row in preview:
            lines.append(f"       {row}")
    return "\n".join(lines)
