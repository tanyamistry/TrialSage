"""Load the evaluation set and compute draft gold answers from the gold SQL.

Separated from the harness so the question set can be validated cheaply --
without running a single LLM call -- before committing an hour of compute to
scoring against it. A question whose gold SQL returns zero rows is usually a
bad question rather than an interesting negative, and it is much better to
find that now.

Nothing here decides truth. `gold_sql` is hand-written and the answers it
produces are drafts; `reviewed` stays false until a human says otherwise.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from trialsage.db import connect

QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"


@dataclass
class EvalQuestion:
    id: str
    question: str
    route: str
    reviewed: bool = False
    gold_sql: Optional[str] = None
    gold_filter_sql: Optional[str] = None
    expect: Dict[str, Any] = field(default_factory=dict)
    note: str = ""

    # Filled in by compute_gold()
    gold_answer: Any = None
    gold_candidate_ids: List[str] = field(default_factory=list)
    gold_error: Optional[str] = None

    @property
    def expected_terms(self) -> List[str]:
        return [t.lower() for t in self.expect.get("terms", [])]

    @property
    def expected_polarity(self) -> Optional[str]:
        return self.expect.get("polarity")


def load_questions(path: Optional[Path] = None) -> List[EvalQuestion]:
    with (path or QUESTIONS_PATH).open() as fh:
        data = yaml.safe_load(fh)
    return [
        EvalQuestion(
            id=item["id"],
            question=item["question"],
            route=item["route"],
            reviewed=bool(item.get("reviewed", False)),
            gold_sql=item.get("gold_sql"),
            gold_filter_sql=item.get("gold_filter_sql"),
            expect=item.get("expect") or {},
            note=item.get("note", ""),
        )
        for item in data["questions"]
    ]


def compute_gold(questions: List[EvalQuestion]) -> List[EvalQuestion]:
    """Execute each question's gold SQL to produce its draft answer."""
    with connect(autocommit=True) as conn:
        for q in questions:
            try:
                if q.gold_sql:
                    row = conn.execute(q.gold_sql).fetchone()
                    q.gold_answer = row[0] if row else None
                if q.gold_filter_sql:
                    q.gold_candidate_ids = [
                        r[0] for r in conn.execute(q.gold_filter_sql).fetchall()
                    ]
            except Exception as exc:  # noqa: BLE001 -- report, do not abort
                q.gold_error = str(exc).splitlines()[0]
    return questions


def unreviewed_count(questions: List[EvalQuestion]) -> int:
    return sum(1 for q in questions if not q.reviewed)


def review_banner(questions: List[EvalQuestion]) -> str:
    n = unreviewed_count(questions)
    if not n:
        return ""
    return (
        "\n" + "!" * 74 + "\n"
        f"  {n} of {len(questions)} gold answers are UNREVIEWED drafts.\n"
        "  Every metric below is provisional until they are checked by hand.\n"
        "  Auto-generated gold answers grade the system against its own output.\n"
        + "!" * 74
    )


def main() -> int:
    questions = compute_gold(load_questions())

    by_route: Dict[str, int] = {}
    for q in questions:
        by_route[q.route] = by_route.get(q.route, 0) + 1

    print(f"Loaded {len(questions)} questions: "
          + ", ".join(f"{r} {n}" for r, n in sorted(by_route.items())))
    print(review_banner(questions))

    problems = []
    print(f"\n{'id':<5} {'route':<11} {'gold answer / candidates':<28} question")
    print("-" * 100)
    for q in questions:
        if q.gold_error:
            detail = f"ERROR: {q.gold_error[:24]}"
            problems.append((q.id, q.gold_error))
        elif q.gold_sql:
            detail = f"= {q.gold_answer}"
            if q.gold_answer in (None, 0):
                problems.append((q.id, f"gold SQL returned {q.gold_answer}"))
        elif q.gold_filter_sql:
            detail = f"{len(q.gold_candidate_ids)} candidate trials"
            if not q.gold_candidate_ids:
                problems.append((q.id, "gold filter matched no trials"))
        else:
            detail = f"terms: {', '.join(q.expected_terms[:3])}"
        print(f"{q.id:<5} {q.route:<11} {detail:<28} {q.question[:52]}")

    print()
    if problems:
        print(f"{len(problems)} question(s) need attention:")
        for qid, why in problems:
            print(f"  {qid}: {why}")
    else:
        print("All gold SQL executed and returned a usable value.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
