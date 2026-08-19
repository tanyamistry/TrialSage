"""Tests for mechanically separating the two halves of a hybrid question.

Background: the router is asked to split a hybrid question into a structured
filter and an eligibility concept. It extracts the concept reliably, but
frequently returns the *whole question* as the structured half. The SQL agent
then tries to express the medical concept as a column filter --
`conditions @> ARRAY['HIV']` -- which matches nothing.

The user-visible symptom is the worst kind: a confident "No matching trials
found" for a question with thousands of real answers. In the Phase 4 sweep
this affected 3 of 16 hybrid questions.

Since the concept is already known, removing it is deterministic and does not
require the model to cooperate a second time.
"""

import pytest

from trialsage.retrieval.hybrid import strip_semantic


class TestStripsConcept:
    @pytest.mark.parametrize("structured,semantic,expected", [
        ("Which phase 3 oncology trials exclude patients with HIV?",
         "patients with HIV", "Which phase 3 oncology trials"),
        ("Which phase 1 trials mention prior radiotherapy?",
         "prior radiotherapy", "Which phase 1 trials"),
        ("Which recruiting phase 3 trials exclude patients with active hepatitis B?",
         "active hepatitis B", "Which recruiting phase 3 trials"),
        ("Which phase 2 trials allow patients with brain metastases?",
         "brain metastases", "Which phase 2 trials"),
    ])
    def test_concept_and_dangling_verb_removed(self, structured, semantic, expected):
        assert strip_semantic(structured, semantic) == expected

    def test_structured_filters_are_preserved(self):
        """Everything the SQL agent legitimately needs must survive."""
        out = strip_semantic(
            "Which recruiting phase 2 oncology trials in Massachusetts allow "
            "patients with a history of autoimmune disease?",
            "history of autoimmune disease")
        for token in ("recruiting", "phase 2", "oncology", "Massachusetts"):
            assert token in out, f"lost structured filter {token!r} from {out!r}"

    def test_already_split_input_is_left_alone(self):
        structured = "recruiting phase 2 oncology trials in Massachusetts"
        assert strip_semantic(structured, "history of autoimmune disease") == structured


class TestSafety:
    def test_no_semantic_query_is_a_no_op(self):
        text = "Which phase 3 trials are recruiting?"
        assert strip_semantic(text, None) == text
        assert strip_semantic(text, "") == text

    def test_never_returns_empty(self):
        """An empty filter description makes the SQL agent invent one from
        nothing, which is how we got hallucinated WHERE clauses. Falling back
        to the original text is strictly safer."""
        out = strip_semantic("patients with HIV", "patients with HIV")
        assert out.strip()

    def test_case_insensitive_match(self):
        out = strip_semantic("Which PHASE 3 trials exclude Patients With HIV?",
                             "patients with hiv")
        assert "HIV" not in out
        assert "PHASE 3" in out

    def test_concept_absent_from_structured_half(self):
        """Router did the split properly; nothing to remove."""
        assert strip_semantic("phase 1 trials", "prior radiotherapy") == "phase 1 trials"
