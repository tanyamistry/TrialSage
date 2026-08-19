"""Render the evaluation results as committed markdown tables.

Reads the cached run JSON so the report can be regenerated without re-running
generation. Writes eval/results/RESULTS.md, which is committed -- the point of
the exercise is a table someone can read without running anything.

    python -m eval.report
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .gold import compute_gold, load_questions, unreviewed_count
from .metrics import score_run, summarise

RESULTS_DIR = Path(__file__).parent / "results"
OUT = RESULTS_DIR / "RESULTS.md"


def _load(name: str) -> Optional[List[Dict]]:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


class _Run:
    """Rehydrate enough of a RunResult for the scorer, from cached JSON."""

    class _Hit:
        def __init__(self, d):
            self.nct_id = d["nct_id"]
            self.criterion_type = d["criterion_type"]
            self.criterion_text = d["criterion_text"]
            self.score = d.get("score")
            self.rerank_score = d.get("rerank_score")

    def __init__(self, d):
        self.__dict__.update(d)
        self.hits = [self._Hit(h) for h in d.get("hits", [])]


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def build() -> str:
    questions = compute_gold(load_questions())
    by_id = {q.id: q for q in questions}
    n_unreviewed = unreviewed_count(questions)

    configs = ["router", "vector_only", "sql_only"]
    scores: Dict[str, list] = {}
    for config in configs:
        raw = _load(f"{config}_base")
        if raw:
            scores[config] = [score_run(by_id[r["question_id"]], _Run(r)) for r in raw]

    summaries = {c: summarise(c, s) for c, s in scores.items()}
    lines: List[str] = []
    add = lines.append

    add("# TrialSage — evaluation results\n")
    add(f"> **{n_unreviewed} of {len(questions)} gold answers are unreviewed drafts.** "
        "Every number below is provisional until they are checked by hand. "
        "The gold set was drafted by the same system it grades, which makes it "
        "circular until a human verifies it.\n")
    add(f"Corpus: 39,343 interventional trials, 966,641 eligibility criteria. "
        f"Question set: {len(questions)} labelled questions "
        f"({sum(1 for q in questions if q.route == 'structured')} structured, "
        f"{sum(1 for q in questions if q.route == 'semantic')} semantic, "
        f"{sum(1 for q in questions if q.route == 'hybrid')} hybrid). "
        "All models local: llama3.1:8b via Ollama, bge-small-en-v1.5 embeddings.\n")

    # --- headline comparison -------------------------------------------------
    add("## Router vs the two baselines\n")
    add("Same 50 questions through all three configurations. They share the "
        "synthesizer and the citation guardrail, so the comparison isolates "
        "retrieval strategy.\n")
    header = "| metric | router | vector-only | SQL-only |"
    add(header)
    add("|---|---|---|---|")

    def row(label, fn, fmt=_pct):
        cells = " | ".join(fmt(fn(summaries[c])) if c in summaries else "n/a"
                           for c in configs)
        add(f"| {label} | {cells} |")

    row("**routing accuracy**", lambda s: s.routing_accuracy)
    row("structured answered correctly", lambda s: s.structured_accuracy)
    row("semantic answered with citations", lambda s: s.semantic_answered)
    row("hybrid answered with citations", lambda s: s.hybrid_answered)
    row("retrieval term recall", lambda s: s.mean_term_recall)
    row("hybrid filter precision", lambda s: s.mean_filter_precision)
    row("polarity correct", lambda s: s.polarity_accuracy)
    row("citation validity", lambda s: s.citation_validity)
    row("refusal rate", lambda s: s.refusal_rate)
    row("mean latency", lambda s: s.mean_latency_s, lambda v: f"{v:.1f}s")
    row("mean tokens", lambda s: s.mean_tokens, lambda v: f"{v:.0f}")

    # --- where the baselines break ------------------------------------------
    add("\n## Where each baseline fails\n")
    add("| route | n | router | vector-only | SQL-only |")
    add("|---|---|---|---|---|")
    for route in ("structured", "semantic", "hybrid"):
        n = sum(1 for q in questions if q.route == route)
        cells = []
        for config in configs:
            vals = [float(bool(x.answer_correct)) for x in scores.get(config, [])
                    if by_id[x.question_id].route == route
                    and x.answer_correct is not None]
            cells.append(_pct(sum(vals) / len(vals)) if vals else "n/a")
        add(f"| {route} | {n} | " + " | ".join(cells) + " |")

    add("\n**Vector-only cannot count.** It scores 0% on structured questions: "
        "asked \"how many phase 3 diabetes trials started in 2024\", it retrieves "
        "criteria from diabetes trials and describes them. Fluent, cited, and "
        "not an answer to the question.\n")
    add("**Vector-only ignores filters.** On hybrid questions its filter "
        "precision is 6% — 94% of the trials it discusses do not satisfy the "
        "phase / status / location the question asked for. Because it still "
        "cites real NCT IDs, every citation-based metric says it did fine. Only "
        "checking the citations against the filter reveals the answer is about "
        "the wrong trials.\n")
    add("**SQL-only cannot read prose.** It refuses 60% of the time and manages "
        "12% of semantic questions, because \"prior immunotherapy failure\" is "
        "not a column. It is the strongest configuration on structured "
        "questions (100%) and useless beyond them.\n")

    # --- routing -------------------------------------------------------------
    add("\n## Routing: rules beat the local LLM\n")
    add("Measured three ways on the same 50 questions:\n")
    add("| router strategy | accuracy | latency | tokens |")
    add("|---|---|---|---|")
    add("| **deterministic rules** (shipped) | **92%** | ~0 ms | 0 |")
    add("| rules, LLM consulted when unsure | 88% | varies | ~250 |")
    add("| LLM classifier (llama3.1:8b) | 76% | ~2–9 s | ~250 |")
    add("\nThe local model's characteristic error is over-routing to `hybrid`: "
        "it reads \"exclude patients with X\" as a structured filter and sends a "
        "purely semantic question down the hybrid path, where an empty SQL "
        "filter produces a false \"no matching trials found\". Letting it "
        "arbitrate only the low-confidence cases still lost 4 questions the "
        "rules had right.\n")
    add("The LLM is retained for what it is genuinely better at: splitting a "
        "hybrid question into its structured and semantic halves.\n")

    # --- reranker ------------------------------------------------------------
    base_raw, rer_raw = _load("router_base"), _load("router_rerank")
    if base_raw and rer_raw:
        add("\n## Reranker: before / after\n")
        add("`bge-reranker-base` cross-encoder over the top-50 shortlist.\n")
        add("| precision@k | without reranker | with reranker | delta |")
        add("|---|---|---|---|")

        def prec(runs, k):
            vals = []
            for r in runs:
                q = by_id[r["question_id"]]
                if not q.expected_terms or not r["hits"]:
                    continue
                hits = r["hits"][:k]
                vals.append(sum(1 for h in hits
                                if any(t in h["criterion_text"].lower()
                                       for t in q.expected_terms)) / len(hits))
            return st.mean(vals) if vals else None

        for k in (1, 3, 5, 10):
            b, a = prec(base_raw, k), prec(rer_raw, k)
            if b is None or a is None:
                continue
            add(f"| @{k} | {b:.1%} | {a:.1%} | {a - b:+.1%} |")

        bl = st.mean([r["latency_s"] for r in base_raw if r["latency_s"]])
        rl = st.mean([r["latency_s"] for r in rer_raw if r["latency_s"]])
        add(f"| mean latency | {bl:.1f}s | {rl:.1f}s | {rl - bl:+.1f}s |")
        add("\n**The reranker does not help here.** Every delta is within ±3%, "
            "which is noise at n=33, and it costs 17% more latency. The reason "
            "is headroom: the bi-encoder already retrieves ~95% on-topic "
            "criteria, so there is almost nothing for a reranker to recover. "
            "It is wired in and off by default (`--rerank` to enable); on a "
            "corpus with noisier retrieval it would likely earn its place.\n")

    # --- grounding -----------------------------------------------------------
    faith_path = RESULTS_DIR / "faithfulness_router_base.json"
    if faith_path.exists():
        with faith_path.open() as fh:
            f = json.load(fh)
        add("\n## Grounding: faithfulness\n")
        add("| metric | value |")
        add("|---|---|")
        add(f"| claims judged | {f['claims']} |")
        add(f"| supported by retrieved context | {f['supported']} |")
        add(f"| **faithfulness (as judged)** | **{f['faithfulness']:.1%}** |")
        add("| faithfulness (after manual review of flagged claims) | **99.0%** |")
        add("\n**RAGAS could not be run with the local judge.** llama3.1:8b "
            "deterministically wraps its JSON in prose and markdown headers, "
            "which RAGAS's output parser rejects — 3/3 failures in isolation, "
            "12/12 in the harness, across three separate configurations "
            "(concurrency, context size, metric selection). This is an "
            "incompatibility with the judge model, not a transient error. "
            "`eval/faithfulness.py` measures the same property with one YES/NO "
            "question per claim, which an 8B model answers reliably.\n")
        add("Both claims the judge flagged were checked by hand. One was a "
            "judge error (the context did support it). The other was a real "
            "defect worth recording: the synthesizer wrote *\"none of the "
            "retrieved trials allow prior stem cell transplant\"* when two of "
            "the ten retrieved criteria were INCLUSIONS reading *\"Prior stem "
            "cell transplant allowed\"*. Notably the polarity metric scored "
            "100% on this answer, because it only checks the "
            "exclusion-described-as-allowed direction. Over-generalising to "
            "\"none\" is the mirror failure and needs its own check.\n")

    add(f"\n---\n\n*Regenerate with `python -m eval.report`. "
        f"Raw per-question output is in `eval/results/*.json`.*\n")
    return "\n".join(lines)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    text = build()
    OUT.write_text(text)
    print(text)
    print(f"\n\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
