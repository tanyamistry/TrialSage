"""Print a health report on what is actually in the database.

Run after ingest (`make verify`). This is deliberately a data-quality report,
not just row counts: it surfaces how well the two tricky parsers did, which is
the thing most likely to be silently wrong.
"""

from __future__ import annotations

from ..db import connect

QUERIES: list[tuple[str, str]] = [
    ("Trials", "SELECT count(*) FROM trials"),
    ("Eligibility criteria", "SELECT count(*) FROM eligibility_chunks"),
    ("Sites", "SELECT count(*) FROM trial_locations"),
    ("Conditions", "SELECT count(*) FROM trial_conditions"),
    ("Interventions", "SELECT count(*) FROM trial_interventions"),
]


def main() -> int:
    with connect(autocommit=True) as conn:
        print("=" * 62)
        print("ROW COUNTS")
        print("=" * 62)
        for label, sql in QUERIES:
            (count,) = conn.execute(sql).fetchone()
            print(f"  {label:<24} {count:>10,}")

        print()
        print("=" * 62)
        print("ELIGIBILITY SPLIT QUALITY  (the parser most likely to be wrong)")
        print("=" * 62)
        rows = conn.execute(
            "SELECT criterion_type, count(*), round(avg(char_len))"
            " FROM eligibility_chunks GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
        for kind, count, avg_len in rows:
            print(f"  {kind:<14} {count:>8,} criteria   avg {int(avg_len):>4} chars")

        (no_chunks,) = conn.execute(
            "SELECT count(*) FROM trials t WHERE NOT EXISTS"
            " (SELECT 1 FROM eligibility_chunks c WHERE c.nct_id = t.nct_id)"
        ).fetchone()
        (untagged_trials,) = conn.execute(
            "SELECT count(DISTINCT nct_id) FROM eligibility_chunks"
            " WHERE criterion_type = 'unspecified'"
        ).fetchone()
        (total,) = conn.execute("SELECT count(*) FROM trials").fetchone()
        if total:
            print(f"\n  trials with no criteria extracted : {no_chunks:,} ({no_chunks / total:.1%})")
            print(f"  trials with untagged criteria     : {untagged_trials:,} ({untagged_trials / total:.1%})")

        print()
        print("=" * 62)
        print("AGE NORMALISATION")
        print("=" * 62)
        row = conn.execute(
            "SELECT count(*) FILTER (WHERE min_age_years IS NOT NULL),"
            "       count(*) FILTER (WHERE max_age_years IS NOT NULL),"
            "       count(*) FILTER (WHERE min_age_raw IS NOT NULL AND min_age_years IS NULL),"
            "       min(min_age_years), max(max_age_years)"
            " FROM trials"
        ).fetchone()
        parsed_min, parsed_max, failed, lo, hi = row
        print(f"  min_age parsed : {parsed_min:,}")
        print(f"  max_age parsed : {parsed_max:,}")
        print(f"  FAILED to parse a stated age : {failed:,}   <-- should be 0")
        print(f"  observed range : {lo} .. {hi} years")

        print()
        print("=" * 62)
        print("STRUCTURED FIELDS")
        print("=" * 62)
        for label, sql in [
            ("phases", "SELECT phase, count(*) FROM trial_phases GROUP BY 1 ORDER BY 1"),
            ("top statuses", "SELECT overall_status, count(*) FROM trials GROUP BY 1 ORDER BY 2 DESC LIMIT 5"),
            ("start years", "SELECT EXTRACT(YEAR FROM start_date)::int, count(*) FROM trials"
                            " WHERE start_date IS NOT NULL GROUP BY 1 ORDER BY 1"),
        ]:
            print(f"\n  {label}:")
            for key, count in conn.execute(sql).fetchall():
                print(f"    {str(key):<24} {count:>8,}")

        print()
        print("=" * 62)
        print("SAMPLE CRITERIA (spot-check the split by eye)")
        print("=" * 62)
        for nct, kind, text in conn.execute(
            "SELECT nct_id, criterion_type, criterion_text FROM eligibility_chunks"
            " WHERE criterion_type IN ('inclusion','exclusion') AND char_len BETWEEN 40 AND 160"
            " ORDER BY nct_id LIMIT 6"
        ).fetchall():
            print(f"  [{kind:<9}] {nct}  {text[:110]}")

        print()
        print("=" * 62)
        print("INGEST RUNS")
        print("=" * 62)
        for area, status, loaded, chunks, started in conn.execute(
            "SELECT area, status, trials_loaded, chunks_loaded, started_at"
            " FROM ingest_runs ORDER BY id DESC LIMIT 5"
        ).fetchall():
            print(f"  {started:%Y-%m-%d %H:%M}  {area:<16} {status:<8} "
                  f"{loaded or 0:>7,} trials  {chunks or 0:>8,} criteria")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
