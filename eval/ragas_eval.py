"""RAGAS faithfulness and context precision, judged by a local model.

Two metrics, both about grounding rather than correctness:

* **faithfulness** — of the claims the answer makes, how many are supported by
  the retrieved context? This is the direct measure of hallucination. Our
  citation guardrail catches fabricated *NCT IDs*; faithfulness catches
  fabricated *statements*, which is the larger surface.

* **context precision** — of the context we retrieved, how much was actually
  relevant to answering? Low precision means the synthesizer is wading through
  noise, which is exactly what the reranker is supposed to fix. This is the
  metric the before/after reranker comparison turns on.

The judge is llama3.1:8b over Ollama, so this costs nothing. That is also its
main weakness and it is worth stating plainly: an 8B judge is noisier than a
frontier model, and small differences between configurations should not be
over-read. The relative comparison (reranker on vs off, scored by the same
judge on the same questions) is far more trustworthy than the absolute number.

    python -m eval.ragas_eval --tag base
    python -m eval.ragas_eval --tag rerank --routes semantic hybrid
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from .gold import compute_gold, load_questions

RESULTS_DIR = Path(__file__).parent / "results"

warnings.filterwarnings("ignore")


def _build_judge():
    """LLM + embeddings for RAGAS, both local."""
    from langchain_ollama import ChatOllama
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from trialsage.config import env, settings

    model = env("LLM_MODEL", "llama3.1:8b")
    base_url = env("OLLAMA_BASE_URL", "http://localhost:11434")

    # temperature=0 so a re-run gives the same verdict; num_predict caps the
    # judge's output so one rambling response cannot stall the whole sweep.
    #
    # num_ctx=8192 is not optional. Ollama's default 4096 is smaller than a
    # RAGAS faithfulness prompt carrying ten retrieved criteria plus the
    # answer plus RAGAS's own few-shot examples, so the server silently
    # context-shifts ("n_discard = 2045") and the judge scores a *truncated*
    # prompt. That produces numbers, which is worse than producing an error:
    # they look like measurements. It is also far slower -- calls were taking
    # 54-77s each with shifting versus roughly half that without.
    llm = LangchainLLMWrapper(ChatOllama(
        model=model, base_url=base_url, temperature=0.0,
        num_predict=512, num_ctx=8192))

    from langchain_huggingface import HuggingFaceEmbeddings
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=settings()["embedding"]["model"]))
    return llm, emb


def load_runs(tag: str, config: str = "router") -> List[Dict]:
    path = RESULTS_DIR / f"{config}_{tag}.json"
    if not path.exists():
        raise SystemExit(f"No results at {path}. Run `python -m eval.run_eval` first.")
    with path.open() as fh:
        return json.load(fh)


def build_dataset(runs: List[Dict], routes: List[str]) -> "object":
    """Assemble the RAGAS input from cached runs.

    Only questions with retrieved text contexts can be scored -- a structured
    answer ("75") has no passages to be faithful to, so scoring it would be
    meaningless rather than merely hard.
    """
    from datasets import Dataset

    questions = {q.id: q for q in compute_gold(load_questions())}
    rows = {"user_input": [], "response": [], "retrieved_contexts": [], "_id": []}

    for run in runs:
        q = questions.get(run["question_id"])
        if q is None or q.route not in routes:
            continue
        contexts = [
            f"[{h['nct_id']}] ({h['criterion_type'].upper()}) {h['criterion_text']}"
            for h in run.get("hits", [])
        ]
        if not contexts or not (run.get("answer") or "").strip():
            continue
        rows["user_input"].append(run["question"])
        rows["response"].append(run["answer"])
        rows["retrieved_contexts"].append(contexts)
        rows["_id"].append(run["question_id"])

    ids = rows.pop("_id")
    dataset = Dataset.from_dict(rows)
    return dataset, ids


def run_ragas(tag: str, config: str, routes: List[str],
              sample: Optional[int] = None,
              with_context_precision: bool = False) -> Optional[Dict]:
    from ragas import evaluate
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference

    runs = load_runs(tag, config)
    dataset, ids = build_dataset(runs, routes)
    if sample and len(dataset) > sample:
        # A local 8B judge costs roughly a minute per answer. Sampling keeps
        # the run tractable; the sample is deterministic (first N) so the
        # before/after reranker comparison scores the same questions.
        dataset = dataset.select(range(sample))
        ids = ids[:sample]
    if len(dataset) == 0:
        print(f"  no scorable rows for {config}/{tag} (routes={routes})")
        return None

    print(f"  scoring {len(dataset)} answers from {config}_{tag} "
          f"(routes: {', '.join(routes)})")
    llm, emb = _build_judge()

    # Concurrency must be 1 and the timeout generous.
    #
    # RAGAS defaults to 16 parallel workers and a 180s timeout, which is right
    # for a hosted API and completely wrong for Ollama: Ollama serialises
    # requests, so 16 "concurrent" calls just queue behind one another and
    # every single one hits the deadline. The first run of this scored 0 of 64
    # jobs -- not a low score, no score at all, which is a far easier failure
    # to misread as "the metric says nothing" than as "the harness is broken".
    from ragas.run_config import RunConfig

    run_config = RunConfig(timeout=900, max_workers=1, max_retries=2)

    # Context precision costs one judge call *per retrieved criterion*
    # (10 contexts x 12 answers = 120 calls at ~50s each on a local 8B model),
    # against roughly 3 calls per answer for faithfulness. On a hosted judge
    # both are cheap; here the difference is 100 minutes versus 15. Faithfulness
    # is the metric that answers "is the synthesizer making things up", so it is
    # the default and context precision is opt-in.
    metrics = [Faithfulness(llm=llm)]
    if with_context_precision:
        metrics.append(LLMContextPrecisionWithoutReference(llm=llm))

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=emb,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=True,
    )

    df = result.to_pandas()
    df.insert(0, "question_id", ids)
    out = RESULTS_DIR / f"ragas_{config}_{tag}.csv"
    df.to_csv(out, index=False)

    scores = {}
    for column in ("faithfulness", "llm_context_precision_without_reference"):
        if column in df.columns:
            series = df[column].dropna()
            scores[column] = float(series.mean()) if len(series) else None
            scores[f"{column}_n"] = int(len(series))
    scores["rows"] = len(df)
    scores["csv"] = str(out)
    return scores


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="base")
    parser.add_argument("--configs", nargs="+", default=["router"])
    parser.add_argument("--context-precision", action="store_true",
                        help="also score context precision (much slower locally)")
    parser.add_argument("--sample", type=int, default=None,
                        help="score only the first N answers (local judge is slow)")
    parser.add_argument("--routes", nargs="+", default=["semantic", "hybrid"],
                        choices=["structured", "semantic", "hybrid"])
    args = parser.parse_args(argv)

    print("RAGAS (judge: local llama3.1:8b — noisier than a frontier judge;\n"
          "       trust the relative comparison more than the absolute value)\n")

    summary = {}
    for config in args.configs:
        scores = run_ragas(args.tag, config, args.routes, sample=args.sample,
                           with_context_precision=args.context_precision)
        if scores:
            summary[config] = scores
            print(f"\n  {config}_{args.tag}:")
            for key, value in scores.items():
                if isinstance(value, float):
                    print(f"    {key:<45} {value:.3f}")
                elif key != "csv":
                    print(f"    {key:<45} {value}")

    if summary:
        out = RESULTS_DIR / f"ragas_summary_{args.tag}.json"
        with out.open("w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
