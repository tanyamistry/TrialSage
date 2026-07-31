"""Ingest CLI: fetch -> parse -> load, for one therapeutic area.

    python -m trialsage.ingest.run --area diabetes
    python -m trialsage.ingest.run --area diabetes --refresh   # re-download
    python -m trialsage.ingest.run --area oncology --limit 200 # quick smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Iterator, Optional

from ..config import settings
from ..db import connect
from .fetch import fetch_to_cache, read_cache
from .load import load_trials, refresh_views
from .parse import ParsedTrial, parse_study


def _parsed_trials(area: str) -> Iterator[ParsedTrial]:
    skipped = 0
    for study in read_cache(area):
        trial = parse_study(study)
        if trial is None:
            skipped += 1
            continue
        yield trial
    if skipped:
        print(f"  skipped {skipped} records with no NCT ID")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest ClinicalTrials.gov data")
    parser.add_argument("--area", required=True, help="therapeutic area from config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="cap records (for smoke tests)")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and re-download")
    args = parser.parse_args(argv)

    areas = settings()["ingest"]["therapeutic_areas"]
    if args.area not in areas:
        print(f"Unknown area '{args.area}'. Available: {', '.join(areas)}", file=sys.stderr)
        return 2

    started = time.time()
    print(f"Ingesting '{args.area}'")

    print("1/3 fetch")
    fetch_to_cache(args.area, limit=args.limit, refresh=args.refresh)

    print("2/3 parse + load")
    with connect() as conn:
        # Record the run so a crash leaves visible evidence rather than a
        # half-loaded table that looks complete.
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ingest_runs (area, status) VALUES (%s, 'running') RETURNING id",
                (args.area,),
            )
            run_id = cur.fetchone()[0]
        conn.commit()

        try:
            loaded, chunks = load_trials(conn, args.area, _parsed_trials(args.area))
            print("3/3 refresh materialized view")
            refresh_views(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE ingest_runs SET finished_at = now(), trials_loaded = %s,"
                    " chunks_loaded = %s, status = 'ok' WHERE id = %s",
                    (loaded, chunks, run_id),
                )
            conn.commit()
        except Exception as exc:  # noqa: BLE001 -- we re-raise after recording
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE ingest_runs SET finished_at = now(), status = 'failed',"
                    " notes = %s WHERE id = %s",
                    (str(exc)[:2000], run_id),
                )
            conn.commit()
            raise

    print(f"Done in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
