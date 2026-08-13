"""Phase 2 sign-off checks against the fully-populated index.

Three things have to be green before Phase 3 starts:

1. The semantic example question returns sensible oncology hits.
2. The headline structured question is back to 75 after the multi-area fix.
3. Polarity filtering and candidate-set scoping still work at full scale.

(3) is the one most likely to regress silently. pgvector's HNSW walks the graph
for `ef_search` candidates and applies the WHERE clause afterwards, so a
filtered search can return zero rows even when thousands of matching chunks
exist -- and it returns them as an empty result, not an error. That failure
mode gets worse as the corpus grows, which is exactly why it is re-checked here
rather than trusted from the 21%-populated run.

    python -m eval.phase2_verify
"""

from __future__ import annotations

import sys
import time
from typing import List, Tuple

from trialsage.db import connect
from trialsage.retrieval.semantic import search, search_trials

PASS, FAIL = "PASS", "FAIL"
results: List[Tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}\n        {detail}")


def h(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def check_corpus() -> None:
    h("0. Corpus state")
    with connect(autocommit=True) as conn:
        done, total = conn.execute(
            "SELECT count(*) FILTER (WHERE embedding IS NOT NULL), count(*)"
            " FROM eligibility_chunks").fetchone()
        indexed = conn.execute(
            "SELECT count(*) FROM pg_indexes"
            " WHERE indexname = 'elig_chunks_embedding_hnsw'").fetchone()[0] == 1
        trials = conn.execute("SELECT count(*) FROM trials").fetchone()[0]

    record("every criterion is embedded", done == total,
           f"{done:,}/{total:,} embedded across {trials:,} trials")
    record("HNSW index exists", indexed,
           "elig_chunks_embedding_hnsw present" if indexed else "index MISSING")


def check_semantic_example() -> None:
    h("1. Semantic example question")
    query = "eligibility mentions prior immunotherapy failure"

    started = time.perf_counter()
    hits = search_trials(query, k=10)
    elapsed = time.perf_counter() - started

    record("returns hits", len(hits) >= 5,
           f"{len(hits)} distinct trials in {elapsed * 1000:.0f} ms")
    if not hits:
        return

    record("top hit is strongly similar", hits[0].score >= 0.75,
           f"top score {hits[0].score:.3f}")

    # These should be oncology trials: immunotherapy failure is an oncology
    # concept, and nothing in the query names a therapeutic area. If diabetes
    # or cardiovascular trials dominate, retrieval is not working.
    with connect(autocommit=True) as conn:
        rows = conn.execute(
            "SELECT nct_id, array_to_string(therapeutic_areas, ',')"
            " FROM v_trials WHERE nct_id = ANY(%s)",
            ([hit.nct_id for hit in hits],),
        ).fetchall()
    areas = {nct: a for nct, a in rows}
    onc = sum(1 for hit in hits if "oncology" in areas.get(hit.nct_id, ""))
    record("hits are predominantly oncology", onc >= len(hits) * 0.7,
           f"{onc}/{len(hits)} tagged oncology")

    # Relevance: the retrieved text should actually be about prior therapy.
    terms = ("immunotherapy", "checkpoint", "pd-1", "pd-l1", "ctla-4",
             "anti-pd", "immune therapy", "progressed on", "refractory")
    relevant = sum(1 for hit in hits
                   if any(t in hit.criterion_text.lower() for t in terms))
    record("retrieved text is topically relevant", relevant >= len(hits) * 0.7,
           f"{relevant}/{len(hits)} mention immunotherapy or progression terms")

    print("\n  Top 5:")
    for i, hit in enumerate(hits[:5], 1):
        print(f"   {i}. {hit.score:.3f} [{hit.criterion_type}] {hit.nct_id}"
              f" ({areas.get(hit.nct_id, '?')})")
        print(f"      {hit.criterion_text[:130]}")


def check_headline() -> None:
    h("2. Headline structured question")
    sql = ("SELECT count(*) FROM v_trials WHERE phases @> ARRAY['PHASE3']"
           " AND therapeutic_areas @> ARRAY['diabetes'] AND start_year = 2024")
    with connect(autocommit=True) as conn:
        n = conn.execute(sql).fetchone()[0]
    record("phase 3 diabetes trials started 2024 == 75", n == 75,
           f"got {n} (live ClinicalTrials.gov API reports 75)")

    with connect(autocommit=True) as conn:
        multi = conn.execute(
            "SELECT count(*) FROM (SELECT nct_id FROM trial_areas"
            " GROUP BY 1 HAVING count(*) > 1) s").fetchone()[0]
    record("multi-area memberships intact", multi > 1800,
           f"{multi:,} trials tagged with more than one area")


def check_polarity_and_scoping() -> None:
    h("3. Polarity filtering and candidate-set scoping")
    query = "history of autoimmune disease"

    unfiltered = search(query, k=20)
    inclusion = search(query, k=20, criterion_type="inclusion")
    exclusion = search(query, k=20, criterion_type="exclusion")

    record("unfiltered search returns results", len(unfiltered) >= 15,
           f"{len(unfiltered)} criteria")
    record("inclusion filter returns a full page", len(inclusion) >= 15,
           f"{len(inclusion)} criteria (0 here = the HNSW post-filter bug is back)")
    record("exclusion filter returns a full page", len(exclusion) >= 15,
           f"{len(exclusion)} criteria")
    record("polarity filter is honoured",
           all(x.criterion_type == "inclusion" for x in inclusion)
           and all(x.criterion_type == "exclusion" for x in exclusion),
           "every returned chunk matches the requested polarity")

    # Autoimmune disease is overwhelmingly an exclusion criterion in reality,
    # so an unfiltered search should lean that way. This is what makes the
    # polarity tagging worth having at all.
    exc_share = sum(1 for x in unfiltered if x.criterion_type == "exclusion") / len(unfiltered)
    print(f"        (unfiltered mix: {exc_share:.0%} exclusion -- expected, "
          "autoimmune history is usually an exclusion)")

    # Candidate-set scoping: the hybrid mechanism.
    with connect(autocommit=True) as conn:
        candidates = [r[0] for r in conn.execute(
            "SELECT nct_id FROM v_trials WHERE overall_status = 'RECRUITING'"
            " AND phases @> ARRAY['PHASE2'] AND therapeutic_areas @> ARRAY['oncology']"
            " AND 'Massachusetts' = ANY(states)").fetchall()]

    record("structured filter finds candidates", len(candidates) > 20,
           f"{len(candidates)} recruiting phase 2 oncology trials in Massachusetts")

    scoped = search_trials(query, k=10, nct_ids=candidates)
    record("scoped search returns hits", len(scoped) >= 5,
           f"{len(scoped)} hits within the {len(candidates)}-trial candidate set")
    record("scoped search stays inside the candidate set",
           all(hit.nct_id in set(candidates) for hit in scoped),
           "no trial leaked in from outside the filter")

    empty = search_trials(query, k=10, nct_ids=[])
    record("empty candidate set returns nothing", empty == [],
           "an empty structured filter does not silently fall back to the whole corpus")

    print("\n  Hybrid top 3 (the third example question):")
    for i, hit in enumerate(scoped[:3], 1):
        print(f"   {i}. {hit.score:.3f} [{hit.criterion_type}] {hit.nct_id}")
        print(f"      {hit.criterion_text[:130]}")


def main() -> int:
    check_corpus()
    check_semantic_example()
    check_headline()
    check_polarity_and_scoping()

    failed = [r for r in results if r[0] == FAIL]
    h("SUMMARY")
    print(f"  {len(results) - len(failed)}/{len(results)} checks passed")
    for status, name, detail in failed:
        print(f"  FAIL  {name} -- {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
