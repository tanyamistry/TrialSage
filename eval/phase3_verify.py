"""Phase 3 sign-off: the three example questions, end to end, plus the guardrail.

Checks that must be green before Phase 4:

1. Each example question is routed correctly, with an inspectable reason.
2. The structured question answers 75.
3. The semantic question returns relevant oncology trials, all citations valid.
4. The hybrid question respects EXCLUSION polarity -- the answer must not say a
   trial allows patients it in fact excludes. This is the one that matters
   clinically, and it is the failure a fluent model produces most readily.
5. The citation guardrail neutralises a fabricated NCT ID.
6. The router still works with the LLM disabled (rule-based fallback).

    python -m eval.phase3_verify
"""

from __future__ import annotations

import sys
from typing import List, Tuple

from trialsage.pipeline import ask
from trialsage.router.classify import classify_by_rules
from trialsage.synth.citations import audit_citations

PASS, FAIL = "PASS", "FAIL"
results: List[Tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}\n        {detail}")


def h(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def show_router(result) -> None:
    d = result.decision
    print(f"  router -> {d.route}  (confidence {d.confidence:.2f}, via {d.source})")
    print(f"           reason: {d.reasoning}")
    if d.semantic_query and d.semantic_query != result.question:
        print(f"           semantic half  : {d.semantic_query!r}")
    if d.structured_query and d.structured_query != result.question:
        print(f"           structured half: {d.structured_query!r}")


def check_structured() -> None:
    h("1. STRUCTURED — 'how many phase 3 diabetes trials started in 2024?'")
    result = ask("how many phase 3 diabetes trials started in 2024?")
    show_router(result)
    print(f"  sql: {result.sql_result.sql if result.sql_result else None}")
    print(f"  answer: {result.text.strip()[:200]}")

    record("routed to structured", result.route == "structured",
           f"route={result.route}")
    record("answer contains 75", "75" in result.text,
           f"answer: {result.text.strip()[:80]!r}")
    record("no fabricated citations", result.grounded,
           "grounded" if result.grounded else str(result.audit.fabricated))


def check_semantic() -> None:
    h("2. SEMANTIC — 'trials whose eligibility mentions prior immunotherapy failure'")
    result = ask("find trials whose eligibility mentions prior immunotherapy failure")
    show_router(result)
    print(f"  hits: {len(result.hits)}")
    for hit in result.hits[:3]:
        print(f"    {hit.score:.3f} [{hit.criterion_type}] {hit.nct_id} "
              f"{hit.criterion_text[:70]}")
    print(f"  answer:\n    " + result.text.strip().replace("\n", "\n    ")[:600])

    record("routed to semantic", result.route == "semantic", f"route={result.route}")
    record("retrieved several trials", len(result.hits) >= 5, f"{len(result.hits)} hits")
    record("hits are topically relevant",
           sum(1 for x in result.hits
               if any(t in x.criterion_text.lower()
                      for t in ("immunotherapy", "checkpoint", "pd-1", "pd-l1",
                                "anti-pd", "immune therap"))) >= len(result.hits) * 0.6,
           "most hits mention immunotherapy")
    record("every citation is real", result.grounded,
           f"{len(result.audit.cited)} valid, "
           f"{len(result.audit.fabricated)} fabricated")
    record("answer cites at least one trial", len(result.audit.cited) >= 1,
           f"cited: {sorted(result.audit.cited)[:4]}")


def check_hybrid() -> None:
    h("3. HYBRID — 'recruiting phase 2 oncology trials in Massachusetts "
      "allowing autoimmune history'")
    question = ("which recruiting phase 2 oncology trials in Massachusetts allow "
                "patients with a history of autoimmune disease?")
    result = ask(question)
    show_router(result)
    hy = result.hybrid
    print(f"  filter sql: {hy.sql if hy else None}")
    print(f"  candidates: {len(hy.candidate_ids) if hy else 0} trials")
    print(f"  hits: {len(result.hits)}")
    print(f"  answer:\n    " + result.text.strip().replace("\n", "\n    ")[:700])

    record("routed to hybrid", result.route == "hybrid", f"route={result.route}")
    record("router split the question in two",
           bool(result.decision.semantic_query)
           and "massachusetts" not in (result.decision.semantic_query or "").lower(),
           f"semantic half = {result.decision.semantic_query!r}")
    record("structured filter ran first and matched trials",
           bool(hy and hy.candidate_ids),
           f"{len(hy.candidate_ids) if hy else 0} candidate trials")
    record("filter SQL contains no eligibility concept",
           bool(hy and hy.sql and "autoimmune" not in hy.sql.lower()),
           "eligibility prose stayed out of the WHERE clause")
    record("semantic search stayed inside the candidate set",
           all(x.nct_id in set(hy.candidate_ids) for x in result.hits) if hy else False,
           "no trial leaked in from outside the filter")

    # The clinical safety check. Every retrieved criterion here is an EXCLUSION,
    # so an answer claiming a trial ALLOWS these patients is inverted -- the
    # single most dangerous output this system can produce.
    exclusion_ids = {x.nct_id for x in result.hits if x.criterion_type == "exclusion"}
    lowered = result.text.lower()
    wrongly_allowed = [
        nct for nct in exclusion_ids
        if any(f"{nct.lower()} {verb}" in lowered
               for verb in ("allows", "allow", "permits", "includes", "accepts"))
    ]
    record("no exclusion trial is described as allowing patients",
           not wrongly_allowed,
           f"polarity correct for all {len(exclusion_ids)} exclusion trials"
           if not wrongly_allowed else f"INVERTED: {wrongly_allowed}")
    record("answer states exclusion, not permission",
           "exclude" in lowered or "none" in lowered or "not allow" in lowered,
           "answer conveys that these trials exclude such patients")
    record("every citation is real", result.grounded,
           f"{len(result.audit.cited)} valid, {len(result.audit.fabricated)} fabricated")


def check_guardrail() -> None:
    h("4. CITATION GUARDRAIL")
    fabricated = ("NCT01234567 excludes autoimmune disease. "
                  "NCT99999999 allows these patients.")
    audit = audit_citations(fabricated, {"NCT01234567"})
    record("fabricated NCT ID is detected", audit.fabricated == {"NCT99999999"},
           f"detected: {sorted(audit.fabricated)}")
    record("fabricated NCT ID is neutralised in the text",
           "[unverified: NCT99999999]" in audit.text,
           "replaced with a visible marker rather than silently deleted")
    record("real citation survives untouched", "NCT01234567" in audit.text,
           "valid citation preserved")

    stripped = audit_citations(fabricated, {"NCT01234567"}, mode="strip")
    record("strip mode removes it entirely", "NCT99999999" not in stripped.text,
           "strip mode available for UI contexts")

    empty = audit_citations("No matching trials found.", set())
    record("refusal path does not trip the guardrail", empty.ok,
           "'no matching trials' is clean")


def check_router_fallback() -> None:
    h("5. ROUTER FALLBACK (LLM disabled)")
    cases = [
        ("how many phase 3 diabetes trials started in 2024?", "structured"),
        ("find trials whose eligibility mentions prior immunotherapy failure", "semantic"),
        ("which recruiting phase 2 oncology trials in Massachusetts allow "
         "patients with a history of autoimmune disease?", "hybrid"),
    ]
    correct = 0
    for question, expected in cases:
        decision = classify_by_rules(question)
        ok = decision.route == expected
        correct += ok
        print(f"    {'ok ' if ok else 'NO '} {expected:<11} <- got {decision.route:<11} "
              f"({decision.confidence:.2f})  {decision.reasoning[:60]}")
    record("rule-based fallback routes all three correctly", correct == 3,
           f"{correct}/3 without any LLM call")

    result = ask("how many phase 3 diabetes trials started in 2024?",
                 use_llm_router=False)
    record("pipeline runs end to end with the LLM router disabled",
           result.route == "structured" and "75" in result.text,
           f"route={result.route}, answer contains 75: {'75' in result.text}")


def main() -> int:
    check_structured()
    check_semantic()
    check_hybrid()
    check_guardrail()
    check_router_fallback()

    failed = [r for r in results if r[0] == FAIL]
    h("SUMMARY")
    print(f"  {len(results) - len(failed)}/{len(results)} checks passed")
    for _, name, detail in failed:
        print(f"  FAIL  {name} -- {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
