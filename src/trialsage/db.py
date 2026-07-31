"""Database connection helpers.

Two entry points on purpose, mirroring the two privilege levels:

* :func:`connect` -- the read/write application role, for ingest and embedding.
* :func:`connect_readonly` -- the restricted role the SQL agent uses in Phase 2.

Anything that generates SQL from a language model must go through
``connect_readonly``. That role has no write grants at all, so even a
successful prompt injection cannot modify the database.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg

from .config import app_dsn, readonly_dsn


@contextmanager
def connect(autocommit: bool = False) -> Iterator[psycopg.Connection]:
    """Open a read/write connection as the application role."""
    with psycopg.connect(app_dsn(), autocommit=autocommit) as conn:
        yield conn


@contextmanager
def connect_readonly() -> Iterator[psycopg.Connection]:
    """Open a connection as the restricted, read-only role."""
    with psycopg.connect(readonly_dsn(), autocommit=True) as conn:
        yield conn


def ping() -> str:
    """Return the server version, or raise if the database is unreachable."""
    with connect(autocommit=True) as conn:
        row = conn.execute("SELECT version()").fetchone()
        return row[0] if row else "unknown"
