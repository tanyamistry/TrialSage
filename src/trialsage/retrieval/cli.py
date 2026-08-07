"""Command-line access to the individual retrievers.

Phase 2 keeps the two retrievers separately runnable on purpose -- being able
to interrogate each one in isolation is what makes the Phase 3 router
debuggable. If a hybrid answer looks wrong, you need to know which half is at
fault.

    python -m trialsage.retrieval.cli sql "how many phase 3 trials started in 2024"
    python -m trialsage.retrieval.cli semantic "prior immunotherapy failure"
    python -m trialsage.retrieval.cli semantic "autoimmune disease" --type exclusion
    python -m trialsage.retrieval.cli hybrid "trials allowing autoimmune history" \
        --where "phases @> ARRAY['PHASE2'] AND overall_status = 'RECRUITING'"
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from ..db import connect
from .semantic import format_hits, search, search_trials
from .sql_agent import explain, generate_sql


def _cmd_sql(args) -> int:
    result = generate_sql(args.question, max_attempts=args.max_attempts)
    print(explain(result))
    if result.ok and len(result.rows) > 3:
        print(f"   ... {len(result.rows) - 3} more rows")
    return 0 if result.ok else 1


def _cmd_semantic(args) -> int:
    fn = search if args.per_criterion else search_trials
    hits = fn(args.query, k=args.k, criterion_type=args.type)
    if not hits:
        print("no matching criteria found")
        return 1
    print(f"{len(hits)} hits for {args.query!r}"
          f"{f' (type={args.type})' if args.type else ''}\n")
    print(format_hits(hits))
    return 0


def _cmd_hybrid(args) -> int:
    """SQL filter first, then vector search restricted to those trials.

    This is the hybrid mechanism in its raw form -- Phase 3 wraps it behind the
    router, but running it manually is the clearest way to see why the ordering
    matters. The filter shrinks the search space before any embedding is
    compared.
    """
    sql = f"SELECT nct_id FROM v_trials WHERE {args.where}"
    with connect(autocommit=True) as conn:
        nct_ids = [r[0] for r in conn.execute(sql).fetchall()]
    print(f"structured filter: {args.where}")
    print(f"  -> {len(nct_ids)} candidate trials")

    if not nct_ids:
        print("no trials match the structured filter; "
              "returning no hits rather than searching everything")
        return 1

    hits = search_trials(args.query, k=args.k, nct_ids=nct_ids,
                         criterion_type=args.type)
    print(f"semantic search within those {len(nct_ids)} trials: {args.query!r}")
    print(f"  -> {len(hits)} hits\n")
    print(format_hits(hits))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sql = sub.add_parser("sql", help="text-to-SQL")
    p_sql.add_argument("question")
    p_sql.add_argument("--max-attempts", type=int, default=3)
    p_sql.set_defaults(func=_cmd_sql)

    p_sem = sub.add_parser("semantic", help="vector search over eligibility criteria")
    p_sem.add_argument("query")
    p_sem.add_argument("-k", type=int, default=10)
    p_sem.add_argument("--type", choices=["inclusion", "exclusion", "unspecified"],
                       default=None, help="restrict to one criterion polarity")
    p_sem.add_argument("--per-criterion", action="store_true",
                       help="return raw criteria instead of one hit per trial")
    p_sem.set_defaults(func=_cmd_semantic)

    p_hyb = sub.add_parser("hybrid", help="SQL filter, then vector search within it")
    p_hyb.add_argument("query")
    p_hyb.add_argument("--where", required=True, help="SQL WHERE clause on v_trials")
    p_hyb.add_argument("-k", type=int, default=10)
    p_hyb.add_argument("--type", choices=["inclusion", "exclusion", "unspecified"],
                       default=None)
    p_hyb.set_defaults(func=_cmd_hybrid)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
