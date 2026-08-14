"""Per-query tracing: route, latency, token counts, and guardrail outcome.

Written as JSONL to ``logs/traces.jsonl`` -- one self-contained JSON object per
query, appended. That format is deliberate: it survives crashes (no closing
bracket needed), it is greppable, and pandas reads it in one line for the
Phase 4 evaluation.

Latency is broken down by stage rather than reported as one number, because
"the answer took 14 seconds" is not actionable but "routing 1.2s, retrieval
0.3s, synthesis 12.5s" tells you exactly where to look.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import LOGS_DIR

TRACE_PATH = LOGS_DIR / "traces.jsonl"


@dataclass
class Trace:
    """One end-to-end query, from question to guarded answer."""

    question: str
    route: str = ""
    route_confidence: float = 0.0
    route_source: str = ""
    route_reasoning: str = ""

    latency_total_s: float = 0.0
    latency_route_s: float = 0.0
    latency_retrieve_s: float = 0.0
    latency_synth_s: float = 0.0

    prompt_tokens: int = 0
    completion_tokens: int = 0

    sql: Optional[str] = None
    sql_attempts: int = 0
    n_candidates: int = 0
    n_context_items: int = 0

    citations_valid: int = 0
    citations_fabricated: int = 0
    uncited_claims: int = 0
    grounded: bool = True

    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["total_tokens"] = self.total_tokens
        return data


def write(trace: Trace, path: Optional[Path] = None) -> None:
    """Append a trace. Never raises -- tracing must not break a query."""
    target = path or TRACE_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a") as fh:
            fh.write(json.dumps(trace.to_dict()) + "\n")
    except OSError:
        pass


def read_all(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    target = path or TRACE_PATH
    if not target.exists():
        return []
    out = []
    with target.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


class Stopwatch:
    """Tiny timing helper so stage timings read cleanly at the call site."""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._mark = self._start

    def lap(self) -> float:
        now = time.perf_counter()
        elapsed = now - self._mark
        self._mark = now
        return elapsed

    def total(self) -> float:
        return time.perf_counter() - self._start
