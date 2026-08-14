"""Re-split eligibility text and repair criterion polarity, keeping embeddings.

Needed whenever the eligibility parser changes. Re-running the full ingest
would work, but it deletes and re-inserts `eligibility_chunks`, discarding
every embedding -- roughly three hours of GPU time.

The saving insight: a parser fix of this kind changes which *section* a
criterion belongs to, not the criterion's *text*. Embeddings are computed from
the text alone, so they can be carried across by matching on it. Only genuinely
new or reworded criteria need embedding afterwards, which is usually a handful.

    python -m trialsage.ingest.rechunk --dry-run
    python -m trialsage.ingest.rechunk
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

from pgvector.psycopg import register_vector

from ..db import connect
from .eligibility import split_eligibility

BATCH = 500


def _current_chunks(cur, nct_id: str) -> List[Tuple[str, str, object]]:
    """(criterion_type, criterion_text, embedding) rows currently stored."""
    return cur.execute(
        "SELECT criterion_type, criterion_text, embedding FROM eligibility_chunks"
        " WHERE nct_id = %s ORDER BY criterion_type, chunk_index",
        (nct_id,),
    ).fetchall()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Re-split eligibility criteria")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    stats = Counter()
    changed_trials: List[str] = []

    with connect() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            rows = cur.execute(
                "SELECT nct_id, eligibility_raw FROM trials"
                " WHERE eligibility_raw IS NOT NULL ORDER BY nct_id"
                + (f" LIMIT {int(args.limit)}" if args.limit else "")
            ).fetchall()

        print(f"Checking {len(rows):,} trials against the current parser")

        for i, (nct_id, raw) in enumerate(rows, 1):
            new = split_eligibility(raw)
            with conn.cursor() as cur:
                old = _current_chunks(cur, nct_id)

            old_map = {(t, txt) for t, txt, _ in old}
            new_map = {(c.criterion_type, c.text) for c in new}
            if old_map == new_map:
                stats["unchanged_trials"] += 1
                continue

            changed_trials.append(nct_id)
            stats["changed_trials"] += 1

            # Which criteria flipped polarity? This is the number that matters:
            # a flip is the difference between "allows" and "excludes".
            old_by_text = {txt: t for t, txt, _ in old}
            for criterion in new:
                previous = old_by_text.get(criterion.text)
                if previous is None:
                    stats["new_criteria"] += 1
                elif previous != criterion.criterion_type:
                    stats[f"flipped_{previous}_to_{criterion.criterion_type}"] += 1
                    stats["flipped_total"] += 1

            if args.dry_run:
                continue

            # Carry embeddings across by text, then replace the rows.
            embeddings = {txt: emb for _, txt, emb in old if emb is not None}
            with conn.cursor() as cur:
                cur.execute("DELETE FROM eligibility_chunks WHERE nct_id = %s", (nct_id,))
                cur.executemany(
                    "INSERT INTO eligibility_chunks"
                    " (nct_id, criterion_type, chunk_index, criterion_text, char_len, embedding)"
                    " VALUES (%s, %s, %s, %s, %s, %s)",
                    [
                        (nct_id, c.criterion_type, c.chunk_index, c.text, c.char_len,
                         embeddings.get(c.text))
                        for c in new
                    ],
                )
                stats["embeddings_reused"] += sum(1 for c in new if c.text in embeddings)
                stats["need_embedding"] += sum(1 for c in new if c.text not in embeddings)

            if i % BATCH == 0:
                conn.commit()
                print(f"  {i:,}/{len(rows):,} checked, {stats['changed_trials']:,} changed",
                      end="\r", flush=True)

        if not args.dry_run:
            conn.commit()

    print(f"  {len(rows):,} checked                                        ")
    print()
    print("=" * 62)
    print("RESULT" + ("  (dry run -- nothing written)" if args.dry_run else ""))
    print("=" * 62)
    print(f"  trials unchanged        : {stats['unchanged_trials']:,}")
    print(f"  trials re-chunked       : {stats['changed_trials']:,}")
    print(f"  criteria that FLIPPED polarity : {stats['flipped_total']:,}")
    for key in sorted(k for k in stats if k.startswith("flipped_") and k != "flipped_total"):
        print(f"      {key.replace('flipped_', '').replace('_', ' '):<34} {stats[key]:,}")
    print(f"  brand-new criteria      : {stats['new_criteria']:,}")
    if not args.dry_run:
        print(f"  embeddings reused       : {stats['embeddings_reused']:,}")
        print(f"  need embedding          : {stats['need_embedding']:,}"
              "   <- run `make embed`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
