"""Central configuration: `config.yaml` for tunables, `.env` for secrets.

Keeping these separate means config.yaml can be committed and shared while
credentials stay local. Everything in the project reads settings from here
rather than calling os.environ directly, so there is one place to look.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=1)
def settings() -> Dict[str, Any]:
    """Parsed config.yaml (cached -- it never changes during a run)."""
    with CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default) or default


def app_dsn() -> str:
    """Read/write connection string, used by ingest and embedding jobs."""
    return (
        f"postgresql://{env('POSTGRES_USER', 'trialsage')}:"
        f"{env('POSTGRES_PASSWORD', 'trialsage')}@"
        f"{env('POSTGRES_HOST', 'localhost')}:"
        f"{env('POSTGRES_PORT', '5433')}/"
        f"{env('POSTGRES_DB', 'trialsage')}"
    )


def readonly_dsn() -> str:
    """Restricted connection string for the text-to-SQL agent (Phase 2).

    This role can only SELECT from the whitelisted views. Nothing in the
    ingest path should ever use it, and the agent should never use anything
    else.
    """
    return (
        f"postgresql://{env('READONLY_USER', 'trialsage_ro')}:"
        f"{env('READONLY_PASSWORD', 'change_me_readonly')}@"
        f"{env('POSTGRES_HOST', 'localhost')}:"
        f"{env('POSTGRES_PORT', '5433')}/"
        f"{env('POSTGRES_DB', 'trialsage')}"
    )
