"""Integration tests for semantic search.

Requires a database with embeddings present; skipped automatically otherwise.

The headline test here is `test_filtered_search_is_not_truncated_by_hnsw`.
pgvector's HNSW index applies the WHERE clause *after* collecting `ef_search`
nearest neighbours, so a filtered search silently returns far fewer rows than
requested -- often zero -- while thousands of matching rows exist. It produces
an empty result set rather than an error, so nothing about it looks broken.
Both features this module exists for (polarity filtering and candidate-set
restriction for the hybrid route) depend on the fix staying in place.
"""

import psycopg
import pytest

from trialsage.config import app_dsn

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedded_db():
    try:
        conn = psycopg.connect(app_dsn(), autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as exc:
        pytest.skip(f"database not reachable: {exc}")
    (n,) = conn.execute(
        "SELECT count(*) FROM eligibility_chunks WHERE embedding IS NOT NULL"
    ).fetchone()
    if n == 0:
        conn.close()
        pytest.skip("no embeddings present; run `make embed` first")
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def sem(embedded_db):
    from trialsage.retrieval import semantic
    return semantic


class TestBasicSearch:
    def test_returns_hits(self, sem):
        hits = sem.search("severe renal impairment", k=5)
        assert len(hits) == 5
        assert all(0.0 <= h.score <= 1.0 for h in hits)

    def test_results_are_sorted_by_score(self, sem):
        scores = [h.score for h in sem.search("kidney disease", k=10)]
        assert scores == sorted(scores, reverse=True)

    def test_hits_carry_polarity_and_citation(self, sem):
        for hit in sem.search("pregnancy", k=5):
            assert hit.criterion_type in {"inclusion", "exclusion", "unspecified"}
            assert hit.nct_id.startswith("NCT")
            assert hit.citation() == hit.nct_id

    def test_as_context_makes_polarity_explicit(self, sem):
        hit = sem.search("pregnancy", k=1)[0]
        rendered = hit.as_context()
        assert hit.nct_id in rendered
        assert any(tag in rendered for tag in ("INCLUSION", "EXCLUSION", "UNSPECIFIED"))

    def test_search_trials_deduplicates_by_trial(self, sem):
        hits = sem.search_trials("insulin therapy", k=10)
        assert len({h.nct_id for h in hits}) == len(hits)


class TestPolarityFilter:
    @pytest.mark.parametrize("polarity", ["inclusion", "exclusion"])
    def test_only_that_polarity_is_returned(self, sem, polarity):
        hits = sem.search("history of autoimmune disease", k=5, criterion_type=polarity)
        assert hits, f"no {polarity} hits -- HNSW filter truncation may have returned"
        assert all(h.criterion_type == polarity for h in hits)

    def test_filtered_search_is_not_truncated_by_hnsw(self, sem, embedded_db):
        """Regression: this returned 0 rows before hnsw.iterative_scan was set.

        Asserts against ground truth from an exact (non-index) scan, so it
        fails if the index scan starts dropping rows again.
        """
        (available,) = embedded_db.execute(
            "SELECT count(*) FROM eligibility_chunks"
            " WHERE criterion_type = 'inclusion' AND embedding IS NOT NULL"
        ).fetchone()
        assert available > 100, "fixture data too small for this test to mean anything"

        hits = sem.search("history of autoimmune disease", k=10,
                          criterion_type="inclusion")
        assert len(hits) == 10, (
            f"asked for 10 inclusion hits, got {len(hits)}, "
            f"though {available} inclusion chunks are indexed"
        )


class TestCandidateSetRestriction:
    """The mechanism behind the hybrid route: SQL filter first, then search."""

    def test_hits_stay_inside_the_candidate_set(self, sem, embedded_db):
        ids = [r[0] for r in embedded_db.execute(
            "SELECT nct_id FROM v_trials WHERE phases @> ARRAY['PHASE2'] LIMIT 200"
        ).fetchall()]
        hits = sem.search_trials("autoimmune disease", k=10, nct_ids=ids)
        assert hits
        assert {h.nct_id for h in hits} <= set(ids)

    def test_restriction_is_not_truncated_by_hnsw(self, sem, embedded_db):
        """A narrow candidate set is the worst case for HNSW post-filtering."""
        ids = [r[0] for r in embedded_db.execute(
            "SELECT nct_id FROM v_trials WHERE phases @> ARRAY['PHASE2']"
            " AND overall_status = 'RECRUITING'"
        ).fetchall()]
        hits = sem.search_trials("history of autoimmune disease", k=10, nct_ids=ids)
        assert len(hits) >= 8, (
            f"only {len(hits)} hits from {len(ids)} candidate trials; "
            "HNSW post-filter truncation is likely back"
        )

    def test_empty_candidate_set_returns_nothing(self, sem):
        """A structured filter that matched nothing must NOT fall back to
        searching the whole corpus -- that would answer about the wrong trials."""
        assert sem.search_trials("anything at all", k=5, nct_ids=[]) == []
        assert sem.search("anything at all", k=5, nct_ids=[]) == []

    def test_none_means_unrestricted(self, sem):
        assert sem.search("pregnancy", k=5, nct_ids=None)
