"""Fetch studies from the ClinicalTrials.gov API v2.

Results are cached to ``data/raw/<area>.jsonl`` so that re-running the parser or
loader does not re-download thousands of records. Delete that file (or pass
``--refresh``) to force a fresh pull.

Paging uses ``nextPageToken``; ``pageSize`` maxes out at 1000 (the API silently
caps anything larger, verified against the live service).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import DATA_DIR, settings

RAW_DIR = DATA_DIR / "raw"


def build_filter(cfg: Dict[str, Any]) -> str:
    """Compose the ``filter.advanced`` expression from config.yaml.

    Produces, for example::

        AREA[StudyType]INTERVENTIONAL
          AND AREA[StartDate]RANGE[2018-01-01,MAX]
          AND (AREA[Phase]PHASE1 OR AREA[Phase]PHASE2 OR ...)
    """
    clauses = [f"AREA[StudyType]{cfg['study_type']}"]
    if cfg.get("start_date_from"):
        clauses.append(f"AREA[StartDate]RANGE[{cfg['start_date_from']},MAX]")
    if cfg.get("phases"):
        phase_expr = " OR ".join(f"AREA[Phase]{p}" for p in cfg["phases"])
        clauses.append(f"({phase_expr})")
    return " AND ".join(clauses)


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _get_page(client: httpx.Client, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    response = client.get(url, params=params)
    response.raise_for_status()
    return response.json()


def iter_studies(
    area: str,
    *,
    limit: Optional[int] = None,
    progress: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Yield raw study dicts for one therapeutic area, following pagination."""
    cfg = settings()["ingest"]
    area_cfg = cfg["therapeutic_areas"][area]

    params: Dict[str, Any] = {
        "query.cond": area_cfg["condition_query"],
        "filter.advanced": build_filter(cfg),
        "pageSize": cfg["page_size"],
        "countTotal": "true",
    }

    fetched = 0
    total: Optional[int] = None
    token: Optional[str] = None

    with httpx.Client(timeout=cfg["request_timeout_s"]) as client:
        while True:
            page_params = dict(params)
            if token:
                page_params["pageToken"] = token

            payload = _get_page(client, cfg["api_base"], page_params)

            if total is None:
                total = payload.get("totalCount")
                if progress:
                    print(f"  {area}: {total} trials match the corpus filter")

            studies: List[Dict[str, Any]] = payload.get("studies", [])
            if not studies:
                break

            for study in studies:
                yield study
                fetched += 1
                if limit is not None and fetched >= limit:
                    return

            if progress:
                print(f"  fetched {fetched}/{total}", end="\r", flush=True)

            token = payload.get("nextPageToken")
            if not token:
                break

    if progress:
        print(f"  fetched {fetched}/{total}        ")


def cache_path(area: str) -> Path:
    return RAW_DIR / f"{area}.jsonl"


def fetch_to_cache(area: str, *, limit: Optional[int] = None, refresh: bool = False) -> Path:
    """Download an area to a JSONL cache file and return its path.

    Writes to a temporary file first so an interrupted download can never leave
    a truncated cache that later looks complete.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(area)

    if path.exists() and not refresh:
        print(f"  using cached {path.relative_to(DATA_DIR.parent)} "
              f"({sum(1 for _ in path.open())} records) -- pass --refresh to re-download")
        return path

    tmp = path.with_suffix(".jsonl.tmp")
    count = 0
    with tmp.open("w") as fh:
        for study in iter_studies(area, limit=limit):
            fh.write(json.dumps(study) + "\n")
            count += 1
    tmp.replace(path)
    print(f"  wrote {count} records to {path.relative_to(DATA_DIR.parent)}")
    return path


def read_cache(area: str) -> Iterator[Dict[str, Any]]:
    with cache_path(area).open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)
