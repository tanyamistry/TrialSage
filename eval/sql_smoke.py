"""Measure text-to-SQL execution accuracy against hand-written gold SQL.

Answers the Phase 2 question: is the local 8B model good enough at SQL, or does
the SQL agent need to point at a hosted model?

"Execution accuracy" compares the *result* of the generated query against the
result of gold SQL written by hand. That is the right metric here -- there are
many correct ways to phrase the same query, so comparing SQL strings would
punish valid answers. What matters is whether the number is right.

Run:  make sql-smoke        (or: python -m eval.sql_smoke)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, List, Optional

from trialsage.db import connect
from trialsage.llm import describe_roles
from trialsage.retrieval.sql_agent import generate_sql


@dataclass
class Case:
    question: str
    gold_sql: str
    note: str = ""


CASES: List[Case] = [
    # The area filter here is load-bearing now that all three therapeutic
    # areas are loaded. Without it the gold answer silently counts oncology
    # and cardiovascular trials too, and marks the model wrong for being right.
    Case("How many phase 3 diabetes trials started in 2024?",
         "SELECT count(*) FROM v_trials WHERE phases @> ARRAY['PHASE3']"
         " AND therapeutic_areas @> ARRAY['diabetes'] AND start_year = 2024",
         "the headline structured example question"),
    Case("How many trials are currently recruiting?",
         "SELECT count(*) FROM v_trials WHERE overall_status = 'RECRUITING'"),
    Case("How many phase 2 trials are there?",
         "SELECT count(*) FROM v_trials WHERE phases @> ARRAY['PHASE2']",
         "tests the array containment idiom"),
    Case("How many trials started in 2023?",
         "SELECT count(*) FROM v_trials WHERE start_year = 2023"),
    Case("How many trials were withdrawn?",
         "SELECT count(*) FROM v_trials WHERE overall_status = 'WITHDRAWN'",
         "tests exact enum casing"),
    Case("What is the average enrollment of completed trials?",
         "SELECT avg(enrollment_count) FROM v_trials WHERE overall_status = 'COMPLETED'"),
    Case("What is the total enrollment across all phase 3 trials?",
         "SELECT sum(enrollment_count) FROM v_trials WHERE phases @> ARRAY['PHASE3']"),
    Case("How many trials have a site in California?",
         "SELECT count(*) FROM v_trials WHERE 'California' = ANY(states)",
         "tests array membership on states"),
    Case("How many phase 4 trials are recruiting?",
         "SELECT count(*) FROM v_trials WHERE phases @> ARRAY['PHASE4']"
         " AND overall_status = 'RECRUITING'",
         "two conditions combined"),
    Case("How many trials accept healthy volunteers?",
         "SELECT count(*) FROM v_trials WHERE healthy_volunteers = true"),
    Case("How many trials have a minimum age under 18 years?",
         "SELECT count(*) FROM v_trials WHERE min_age_years < 18",
         "NULL handling: trials with no stated minimum must not count"),
    # Deliberately asks for the COUNT, not the sponsor name: Eli Lilly and
    # Novo Nordisk are tied at 138 trials, so "which sponsor runs the most"
    # has two equally correct answers and the tie-break is arbitrary. Asking
    # for the number keeps GROUP BY + ORDER BY coverage without a gold answer
    # that depends on row ordering luck.
    Case("What is the highest number of trials run by any single lead sponsor?",
         "SELECT count(*) FROM v_trials GROUP BY lead_sponsor"
         " ORDER BY count(*) DESC LIMIT 1",
         "GROUP BY + ORDER BY"),
]


def _norm(value: Any) -> Any:
    """Compare numbers loosely and everything else as a string."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    try:
        return round(float(str(value)), 2)
    except (TypeError, ValueError):
        return str(value).strip()


def gold_answer(sql: str) -> Any:
    with connect(autocommit=True) as conn:
        row = conn.execute(sql).fetchone()
    return _norm(row[0]) if row else None


def run(max_attempts: int = 3, only: Optional[int] = None) -> int:
    print("LLM configuration:")
    print(describe_roles())
    print()

    cases = CASES[:only] if only else CASES
    passed = 0
    lenient_passed = 0
    total_latency = 0.0
    total_tokens = 0
    first_try = 0
    guard_blocks = 0
    failures = []

    for i, case in enumerate(cases, 1):
        expected = gold_answer(case.gold_sql)
        result = generate_sql(case.question, max_attempts=max_attempts)
        total_latency += result.latency_s
        total_tokens += result.total_tokens

        actual = _norm(result.scalar()) if result.ok else None
        ok = result.ok and actual == expected

        # Lenient: the right value is present in the first row, but not in
        # column 0 -- typically the model returned an extra label column
        # alongside the number. Wrong shape, right answer. A downstream
        # synthesizer reading the rows would still answer correctly, so this
        # is worth measuring separately rather than scoring as a flat miss.
        lenient_ok = ok or (
            result.ok
            and bool(result.rows)
            and any(_norm(cell) == expected for cell in result.rows[0])
        )
        if lenient_ok:
            lenient_passed += 1

        if ok:
            passed += 1
            if result.n_attempts == 1:
                first_try += 1
        else:
            failures.append((case, result, expected, actual, lenient_ok))

        guard_blocks += sum(1 for a in result.attempts if a.guard_reason)

        mark = "PASS" if ok else "FAIL"
        print(f"[{i:>2}/{len(cases)}] {mark}  expected={expected!r} got={actual!r}  "
              f"tries={result.n_attempts} {result.latency_s:.1f}s")
        print(f"          Q: {case.question}")
        if result.sql:
            print(f"          SQL: {result.sql}")
        if not ok:
            reason = result.error or "wrong answer"
            print(f"          !! {reason}")
        print()

    n = len(cases)
    print("=" * 70)
    print(f"Execution accuracy (strict)  : {passed}/{n}  ({passed / n:.0%})")
    print(f"Execution accuracy (lenient) : {lenient_passed}/{n}  ({lenient_passed / n:.0%})"
          "   [right value present, possibly with an extra column]")
    print(f"Correct first try            : {first_try}/{n}  ({first_try / n:.0%})")
    print(f"Guard rejections             : {guard_blocks}")
    print(f"Mean latency                 : {total_latency / n:.1f}s per question")
    print(f"Mean tokens                  : {total_tokens // n} per question")
    print("=" * 70)

    if failures:
        print("\nFAILURES\n" + "-" * 70)
        for case, result, expected, actual, lenient_ok in failures:
            print(f"Q: {case.question}")
            if case.note:
                print(f"   ({case.note})")
            tag = "  (LENIENT PASS: right value in another column)" if lenient_ok else ""
            print(f"   expected {expected!r}, got {actual!r}{tag}")
            print(f"   gold: {case.gold_sql}")
            print(f"   gen : {result.sql or '<none produced>'}")
            for j, attempt in enumerate(result.attempts, 1):
                problem = attempt.guard_reason or attempt.db_error
                if problem:
                    print(f"   [{j}] {problem}")
            print()

    return 0 if passed == n else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--only", type=int, default=None, help="run only the first N cases")
    args = parser.parse_args(argv)
    return run(max_attempts=args.max_attempts, only=args.only)


if __name__ == "__main__":
    sys.exit(main())
