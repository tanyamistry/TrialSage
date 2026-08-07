"""Rebuild `trial_areas` from the cached API responses.

Area membership is the one thing we know from *which query returned a trial*
rather than from the trial's own JSON, so it cannot be recomputed from the
trials table -- it has to come from the per-area cache files.

This exists as a separate repair command because re-running the full ingest
would delete and re-insert `eligibility_chunks`, discarding every embedding
computed so far. Retagging touches only `trial_areas`.

    python -m trialsage.ingest.retag
"""

from __future__ import annotations

import json
import sys
from typing import Dict, List, Optional, Set

from ..config import settings
from ..db import connect
from .fetch import cache_path
from .load import refresh_views


def areas_from_cache() -> Dict[str, Set[str]]:
    """Map area -> set of NCT IDs, read from data/raw/<area>.jsonl."""
    result: Dict[str, Set[str]] = {}
    for area in settings()["ingest"]["therapeutic_areas"]:
        path = cache_path(area)
        if not path.exists():
            print(f"  {area}: no cache file, skipping")
            continue
        ids: Set[str] = set()
        with path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                nct = json.loads(line)["protocolSection"]["identificationModule"].get("nctId")
                if nct:
                    ids.add(nct)
        result[area] = ids
        print(f"  {area}: {len(ids):,} trials in cache")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    print("Rebuilding trial_areas from cached API responses")
    by_area = areas_from_cache()
    if not by_area:
        print("no caches found; nothing to do")
        return 1

    with connect() as conn:
        with conn.cursor() as cur:
            before = cur.execute("SELECT count(*) FROM trial_areas").fetchone()[0]

            # Only insert for trials that actually exist, so a cache that is
            # ahead of the database cannot violate the foreign key.
            existing = {
                r[0] for r in cur.execute("SELECT nct_id FROM trials").fetchall()
            }
            rows = [
                (nct, area)
                for area, ids in by_area.items()
                for nct in ids
                if nct in existing
            ]
            cur.executemany(
                "INSERT INTO trial_areas (nct_id, area) VALUES (%s, %s)"
                " ON CONFLICT DO NOTHING",
                rows,
            )
            after = cur.execute("SELECT count(*) FROM trial_areas").fetchone()[0]
        conn.commit()

        print(f"\n  memberships: {before:,} -> {after:,}  (+{after - before:,} restored)")

        with conn.cursor() as cur:
            for area, n in cur.execute(
                "SELECT area, count(*) FROM trial_areas GROUP BY 1 ORDER BY 2 DESC"
            ).fetchall():
                expected = len(by_area.get(area, ()))
                flag = "" if n >= expected else f"   <-- still short by {expected - n}"
                print(f"    {area:<16} {n:>7,} / {expected:,}{flag}")
            multi = cur.execute(
                "SELECT count(*) FROM (SELECT nct_id FROM trial_areas"
                " GROUP BY 1 HAVING count(*) > 1) s"
            ).fetchone()[0]
            print(f"    trials in >1 area: {multi:,}")

        print("\n  refreshing v_trials so the SQL agent sees the corrected areas")
        refresh_views(conn)

    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
