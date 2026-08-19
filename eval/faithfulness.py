"""Claim-level faithfulness, measured with a local model.

Written because RAGAS could not be made to work with llama3.1:8b. RAGAS asks
the judge for structured JSON; this model deterministically prefixes it with
prose ("Here is the JSON output for each statement:") and markdown headers, so
every job failed with a parser exception -- 3/3 in isolation, 12/12 in the
harness. That is a limitation of the judge, not a bug to retry around.

The metric here is the same idea, built to the model's actual capability:
split the answer into claims and ask one YES/NO question per claim. A binary
answer is something an 8B model returns reliably, and the parsing is ours.

    faithfulness = claims supported by the retrieved context / total claims

Definitions kept deliberately narrow so the number means something:
* only sentences making a substantive claim are scored (bullets and prose,
  not headings);
* the judge sees only the retrieved context, never the trial database, so
  "supported" means supported *by what we actually retrieved*.

    python -m eval.faithfulness --tag base --sample 12
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from trialsage.llm import LLMError, get_llm

RESULTS_DIR = Path(__file__).parent / "results"

_JUDGE_SYSTEM = (
    "You verify whether a claim is supported by a context. "
    "You reply with exactly one word: YES or NO."
)

_JUDGE_PROMPT = """\
Context:
{context}

Claim: {claim}

Is this claim fully supported by the context above? Consider a claim
unsupported if it states a fact the context does not contain, or if it
reverses the meaning (saying a trial allows something the context says it
excludes).

Reply with exactly one word, YES or NO."""

_YES = re.compile(r"\byes\b", re.IGNORECASE)
_NO = re.compile(r"\bno\b", re.IGNORECASE)


@dataclass
class AnswerScore:
    question_id: str
    claims: int = 0
    supported: int = 0
    unsupported_examples: List[str] = field(default_factory=list)
    unparseable: int = 0

    @property
    def faithfulness(self) -> Optional[float]:
        return self.supported / self.claims if self.claims else None


def split_claims(answer: str) -> List[str]:
    """Sentences and bullets that assert something checkable."""
    parts = re.split(r"(?:\n+|(?<=[.!?])\s+)", answer or "")
    claims = []
    for part in parts:
        text = part.strip().strip("•-*").strip()
        if len(text.split()) < 4:
            continue
        if text.lower().startswith(("here are", "the following", "based on")):
            continue
        claims.append(text)
    return claims


def judge(claim: str, context: str) -> Optional[bool]:
    llm = get_llm("default")
    try:
        response = llm.complete(_JUDGE_PROMPT.format(context=context, claim=claim),
                                system=_JUDGE_SYSTEM, max_tokens=8)
    except LLMError:
        return None
    text = response.text.strip()
    # Check NO first: "NO" is a substring risk in words like "not", but the
    # word-boundary regex handles that, and a leading NO should win over a
    # trailing hedge.
    if _NO.search(text[:20]):
        return False
    if _YES.search(text[:20]):
        return True
    return None


def score_answer(run: Dict) -> AnswerScore:
    score = AnswerScore(question_id=run["question_id"])
    contexts = [f"[{h['nct_id']}] ({h['criterion_type'].upper()}) {h['criterion_text']}"
                for h in run.get("hits", [])]
    if not contexts:
        return score
    context = "\n".join(contexts)

    for claim in split_claims(run.get("answer") or ""):
        verdict = judge(claim, context)
        if verdict is None:
            score.unparseable += 1
            continue
        score.claims += 1
        if verdict:
            score.supported += 1
        elif len(score.unsupported_examples) < 3:
            score.unsupported_examples.append(claim[:130])
    return score


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="base")
    parser.add_argument("--config", default="router")
    parser.add_argument("--sample", type=int, default=12)
    args = parser.parse_args(argv)

    path = RESULTS_DIR / f"{args.config}_{args.tag}.json"
    if not path.exists():
        raise SystemExit(f"no results at {path}")
    with path.open() as fh:
        runs = [r for r in json.load(fh) if r.get("hits")]
    runs = runs[:args.sample]

    print(f"Claim-level faithfulness, judge=llama3.1:8b, "
          f"{len(runs)} answers from {args.config}_{args.tag}\n")

    scores: List[AnswerScore] = []
    for i, run in enumerate(runs, 1):
        score = score_answer(run)
        scores.append(score)
        value = score.faithfulness
        print(f"  [{i:>2}/{len(runs)}] {score.question_id}  "
              f"{score.supported}/{score.claims} claims supported"
              f"{f' = {value:.0%}' if value is not None else ''}", flush=True)

    total_claims = sum(s.claims for s in scores)
    total_supported = sum(s.supported for s in scores)
    unparseable = sum(s.unparseable for s in scores)
    overall = total_supported / total_claims if total_claims else None

    print("\n" + "=" * 66)
    print("FAITHFULNESS")
    print("=" * 66)
    print(f"  claims judged        : {total_claims}")
    print(f"  supported by context : {total_supported}")
    print(f"  unparseable verdicts : {unparseable}")
    if overall is not None:
        print(f"  faithfulness         : {overall:.1%}")
    per = [s.faithfulness for s in scores if s.faithfulness is not None]
    if per:
        print(f"  per-answer worst     : {min(per):.0%}   best: {max(per):.0%}")

    unsupported = [(s.question_id, c) for s in scores for c in s.unsupported_examples]
    if unsupported:
        print(f"\n  unsupported claims ({len(unsupported)} shown):")
        for qid, claim in unsupported[:10]:
            print(f"    [{qid}] {claim}")

    out = RESULTS_DIR / f"faithfulness_{args.config}_{args.tag}.json"
    with out.open("w") as fh:
        json.dump({
            "faithfulness": overall,
            "claims": total_claims,
            "supported": total_supported,
            "unparseable": unparseable,
            "per_answer": [{"question_id": s.question_id, "claims": s.claims,
                            "supported": s.supported,
                            "unsupported_examples": s.unsupported_examples}
                           for s in scores],
        }, fh, indent=2)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
