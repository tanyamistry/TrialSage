"""Tests for the citation guardrail.

A fabricated NCT ID is the most dangerous output this system can produce: it
looks authoritative, it is the exact kind of thing a reader will not verify,
and in a healthcare context it can send someone chasing a trial that does not
exist. These tests treat the model as an adversary that will cite whatever it
likes, and check that the guardrail neutralises it regardless.
"""

import pytest

from trialsage.synth.citations import (
    append_warnings,
    audit_citations,
    extract_nct_ids,
)

CONTEXT = {"NCT01234567", "NCT07654321"}


class TestExtraction:
    def test_finds_ids(self):
        assert extract_nct_ids("See NCT01234567 and NCT07654321.") == CONTEXT

    def test_ignores_malformed(self):
        # Too short, too long, or missing the prefix.
        assert extract_nct_ids("NCT123 NCT123456789012 12345678 XCT01234567") == set()

    def test_handles_empty(self):
        assert extract_nct_ids("") == set()
        assert extract_nct_ids(None) == set()


class TestFabricatedCitations:
    def test_detects_id_not_in_context(self):
        audit = audit_citations(
            "Trial NCT01234567 excludes autoimmune disease, as does NCT09999999.",
            CONTEXT)
        assert audit.fabricated == {"NCT09999999"}
        assert audit.cited == {"NCT01234567"}
        assert audit.has_fabrications
        assert not audit.ok

    def test_flag_mode_marks_it_visibly(self):
        """Flagging beats silent deletion: removing the ID would leave the
        sentence intact and still looking sourced."""
        audit = audit_citations("As shown in NCT09999999, the drug works.",
                                CONTEXT, mode="flag")
        assert "[unverified: NCT09999999]" in audit.text
        assert "NCT09999999," not in audit.text

    def test_strip_mode_removes_it(self):
        audit = audit_citations("As shown in NCT09999999, the drug works.",
                                CONTEXT, mode="strip")
        assert "NCT09999999" not in audit.text

    def test_multiple_fabrications(self):
        audit = audit_citations("NCT11111111 and NCT22222222 and NCT01234567.",
                                CONTEXT)
        assert audit.fabricated == {"NCT11111111", "NCT22222222"}
        assert audit.cited == {"NCT01234567"}

    def test_all_valid_passes_clean(self):
        audit = audit_citations(
            "NCT01234567 excludes autoimmune disease. NCT07654321 also excludes it.",
            CONTEXT)
        assert audit.fabricated == set()
        assert audit.has_fabrications is False
        assert audit.ok

    def test_empty_context_makes_every_citation_fabricated(self):
        """With nothing retrieved, any cited trial is invented by definition."""
        audit = audit_citations("Trial NCT01234567 is relevant here.", set())
        assert audit.fabricated == {"NCT01234567"}

    def test_case_insensitive_context_matching(self):
        audit = audit_citations("See NCT01234567.", {"nct01234567"})
        assert audit.fabricated == set()


class TestUncitedClaims:
    def test_flags_bare_generalisation_with_no_citations_anywhere(self):
        """An aggregate claim is only acceptable when the answer cites the
        specifics somewhere. On its own it is an unsupported generalisation."""
        audit = audit_citations(
            "Several recruiting trials exclude patients with autoimmune disease.",
            CONTEXT)
        assert audit.uncited_claims
        assert not audit.ok

    def test_cited_claim_is_accepted(self):
        audit = audit_citations(
            "NCT01234567 excludes patients who have autoimmune disease.", CONTEXT)
        assert audit.uncited_claims == []

    def test_no_match_message_is_not_a_claim(self):
        """The refusal path must not itself trip the guardrail."""
        audit = audit_citations("No matching trials found.", set())
        assert audit.uncited_claims == []
        assert audit.ok

    def test_meta_commentary_is_not_a_claim(self):
        audit = audit_citations(
            "Based on the retrieved context, I cannot answer that question.",
            CONTEXT)
        assert audit.uncited_claims == []

    @pytest.mark.parametrize("summary", [
        "None of the recruiting phase 2 oncology trials in Massachusetts allow "
        "patients with a history of autoimmune disease.",
        "All of the retrieved trials exclude patients with autoimmune disease.",
        "There are 7 trials that exclude patients with this condition.",
        "Several trials exclude patients with prior immunotherapy.",
        # Lead-in forms observed in the Phase 4 sweep, which the first version
        # of the exemption missed and wrongly flagged.
        "The eligibility criteria for prior stem cell transplant are mentioned in several trials.",
        "The following clinical trials exclude patients with active hepatitis B:",
        "The retrieved criteria indicate that patients with severe renal impairment are excluded.",
        "Based on the provided eligibility criteria, several trials exclude prior transplant.",
        "The majority of the retrieved clinical trials require an ECOG performance status of 0 or 1.",
    ])
    def test_aggregate_summaries_do_not_need_their_own_citation(self, summary):
        """A summary line is supported by the cited bullets beneath it.

        Flagging it would be a false positive, and a guardrail that cries wolf
        on correct answers is one people learn to ignore.
        """
        audit = audit_citations(
            f"{summary}\n- NCT01234567 excludes them.\n- NCT07654321 excludes them.",
            CONTEXT)
        assert audit.uncited_claims == [], audit.uncited_claims

    def test_specific_claim_without_citation_is_still_flagged(self):
        """The loosening must not blunt the actual check."""
        audit = audit_citations(
            "The trial in Boston enrolls patients with autoimmune disease and "
            "is currently recruiting participants.", CONTEXT)
        assert audit.uncited_claims

    def test_can_be_disabled_for_aggregates(self):
        """A count answers with a number and names no trial, so requiring a
        citation on it would be nonsense."""
        audit = audit_citations("There are 75 phase 3 diabetes trials.", set(),
                                require_citations=False)
        assert audit.uncited_claims == []
        assert audit.ok


class TestReporting:
    def test_unused_context_is_tracked(self):
        audit = audit_citations("Only NCT01234567 is relevant.", CONTEXT)
        assert audit.unused_context == {"NCT07654321"}

    def test_summary_mentions_fabrications(self):
        audit = audit_citations("NCT09999999 says so.", CONTEXT)
        assert "FABRICATED" in audit.summary()

    def test_warnings_appended_when_not_ok(self):
        audit = audit_citations("NCT09999999 says so.", CONTEXT)
        out = append_warnings(audit)
        assert "Citation guardrail" in out
        assert "NCT09999999" in out

    def test_no_warnings_when_clean(self):
        audit = audit_citations("NCT01234567 excludes autoimmune disease.", CONTEXT)
        assert append_warnings(audit) == audit.text
        assert "Citation guardrail" not in append_warnings(audit)
