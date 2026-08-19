"""Run the evaluation: three configurations over the same 50 questions.

    python -m eval.run_eval                      # router + both baselines
    python -m eval.run_eval --configs router     # just the router
    python -m eval.run_eval --rerank             # router with the reranker on
    python -m eval.run_eval --limit 6            # quick smoke test

Writes per-question JSON to eval/results/<tag>.json so scoring, RAGAS and the
report can all be re-run without paying for generation again. On a local 8B
model a full three-configuration sweep is roughly 45 minutes, so caching the
raw runs matters.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from .baselines import CONFIGS, RunResult
from .gold import EvalQuestion, compute_gold, load_questions, review_banner
from .metrics import QuestionScore, score_run, summarise

RESULTS_DIR = Path(__file__).parent / "results"


def _serialise(run: RunResult) -> Dict:
    data = asdict(run)
    data.pop("decision", None)
    data["hits"] = [
        {"nct_id": h.nct_id, "criterion_type": h.criterion_type,
         "criterion_text": h.criterion_text, "score": h.score,
         "rerank_score": getattr(h, "rerank_score", None)}
        for h in run.hits
    ]
    data["scalar"] = str(run.scalar) if run.scalar is not None else None
    if run.decision is not None:
        data["route_confidence"] = run.decision.confidence
        data["route_source"] = run.decision.source
        data["route_reasoning"] = run.decision.reasoning
    return data


def run_config(config: str, questions: List[EvalQuestion], *,
               rerank: bool = False) -> List[RunResult]:
    fn = CONFIGS[config]
    runs: List[RunResult] = []
    started = time.perf_counter()

    for i, q in enumerate(questions, 1):
        try:
            run = fn(q.id, q.question, rerank=rerank)
        except Exception as exc:  # noqa: BLE001 -- one bad question must not
            #                        abort a 45-minute sweep
            run = RunResult(config=config, question_id=q.id, question=q.question,
                            predicted_route="error", answer="", error=str(exc)[:300])
        runs.append(run)
        elapsed = time.perf_counter() - started
        eta = (elapsed / i) * (len(questions) - i)
        print(f"  [{config}] {i:>2}/{len(questions)} {q.id} "
              f"({run.latency_s:.0f}s, eta {eta / 60:.0f}m)", flush=True)
    return runs


def save(tag: str, runs: List[RunResult]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{tag}.json"
    with path.open("w") as fh:
        json.dump([_serialise(r) for r in runs], fh, indent=2)
    return path


def print_summary(questions: List[EvalQuestion],
                  all_scores: Dict[str, List[QuestionScore]]) -> None:
    by_id = {q.id: q for q in questions}

    print("\n" + "=" * 92)
    print("COMPARISON — same 50 questions through all three configurations")
    print("=" * 92)
    header = (f"{'metric':<34} " + " ".join(f"{c:>17}" for c in all_scores))
    print(header)
    print("-" * len(header))

    summaries = {c: summarise(c, s) for c, s in all_scores.items()}

    def row(label: str, fn, pct: bool = True) -> None:
        cells = []
        for config in all_scores:
            value = fn(summaries[config])
            if value is None:
                cells.append(f"{'n/a':>17}")
            elif pct:
                cells.append(f"{value:>16.0%} ")
            else:
                cells.append(f"{value:>17.1f}")
        print(f"{label:<34} " + " ".join(cells))

    row("routing accuracy", lambda s: s.routing_accuracy)
    print("-" * len(header))
    row("structured answered correctly", lambda s: s.structured_accuracy)
    row("semantic answered with citations", lambda s: s.semantic_answered)
    row("hybrid answered with citations", lambda s: s.hybrid_answered)
    print("-" * len(header))
    row("retrieval term recall", lambda s: s.mean_term_recall)
    row("hybrid filter precision", lambda s: s.mean_filter_precision)
    row("polarity correct", lambda s: s.polarity_accuracy)
    row("citation validity", lambda s: s.citation_validity)
    row("refusal rate", lambda s: s.refusal_rate)
    print("-" * len(header))
    row("mean latency (s)", lambda s: s.mean_latency_s, pct=False)
    row("mean tokens", lambda s: s.mean_tokens, pct=False)

    # Where each configuration falls over, question by question.
    print("\n" + "=" * 92)
    print("WHERE EACH CONFIGURATION FAILS, BY ROUTE")
    print("=" * 92)
    print(f"{'route':<12} {'n':>3}  " + " ".join(f"{c:>17}" for c in all_scores))
    for route in ("structured", "semantic", "hybrid"):
        n = sum(1 for q in questions if q.route == route)
        cells = []
        for config, scores in all_scores.items():
            vals = [float(bool(x.answer_correct)) for x in scores
                    if by_id[x.question_id].route == route
                    and x.answer_correct is not None]
            cells.append(f"{(sum(vals) / len(vals)):>16.0%} " if vals else f"{'n/a':>17}")
        print(f"{route:<12} {n:>3}  " + " ".join(cells))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS),
                        choices=list(CONFIGS))
    parser.add_argument("--rerank", action="store_true",
                        help="enable the cross-encoder reranker")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default=None, help="suffix for the results files")
    args = parser.parse_args(argv)

    questions = compute_gold(load_questions())
    if args.limit:
        # Keep the route mix when sampling, otherwise a smoke test can end up
        # all-structured and prove nothing.
        per_route = max(1, args.limit // 3)
        sampled: List[EvalQuestion] = []
        for route in ("structured", "semantic", "hybrid"):
            sampled += [q for q in questions if q.route == route][:per_route]
        questions = sampled[:args.limit]

    print(f"Evaluating {len(questions)} questions "
          f"x {len(args.configs)} configs"
          f"{' (reranker ON)' if args.rerank else ''}")
    print(review_banner(questions))
    print()

    suffix = args.tag or ("rerank" if args.rerank else "base")
    all_scores: Dict[str, List[QuestionScore]] = {}

    for config in args.configs:
        runs = run_config(config, questions, rerank=args.rerank)
        path = save(f"{config}_{suffix}", runs)
        print(f"  -> {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
        by_id = {q.id: q for q in questions}
        all_scores[config] = [score_run(by_id[r.question_id], r) for r in runs]

    print_summary(questions, all_scores)
    print(review_banner(questions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
