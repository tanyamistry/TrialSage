"""Tests for eligibility splitting.

The headline requirement: a criterion must carry the polarity of the section it
came from. "History of autoimmune disease" as an INCLUSION criterion and as an
EXCLUSION criterion are opposite facts, and mixing them up produces confidently
wrong medical answers.

Fixtures below are shaped after real ClinicalTrials.gov records, including the
escaped-markdown quirk and the ~9% of trials with missing or partial headers.
"""

import pytest

from trialsage.ingest.eligibility import split_eligibility, unescape

# Verbatim shape of a real record: '* ' bullets and backslash-escaped markdown.
REAL_WORLD = r"""Inclusion Criteria:

* Patients must be \> 365 days and \< 18 years at the time of enrollment
* Newly-diagnosed Ph+ or ABL-class Ph-like B-ALL \[COG\]

Exclusion Criteria:

* Prior treatment with a tyrosine kinase inhibitor
* Known history of autoimmune disease requiring systemic therapy
"""


def _texts(criteria, kind):
    return [c.text for c in criteria if c.criterion_type == kind]


class TestUnescape:
    def test_removes_markdown_backslashes(self):
        assert unescape(r"\> 365 days and \< 18 years") == "> 365 days and < 18 years"
        assert unescape(r"\[COG\]") == "[COG]"
        assert unescape(r"5 \* 10\^9") == "5 * 10^9"

    def test_leaves_clean_text_untouched(self):
        assert unescape("ECOG performance status 0-1") == "ECOG performance status 0-1"


class TestSectionTagging:
    def test_splits_both_sections(self):
        criteria = split_eligibility(REAL_WORLD)
        assert len(_texts(criteria, "inclusion")) == 2
        assert len(_texts(criteria, "exclusion")) == 2

    def test_polarity_is_correct(self):
        """The core guarantee: autoimmune disease here is an EXCLUSION."""
        criteria = split_eligibility(REAL_WORLD)
        autoimmune = [c for c in criteria if "autoimmune" in c.text.lower()]
        assert len(autoimmune) == 1
        assert autoimmune[0].criterion_type == "exclusion"

    def test_headers_are_not_emitted_as_criteria(self):
        criteria = split_eligibility(REAL_WORLD)
        assert not any("inclusion criteria" in c.text.lower() for c in criteria)
        assert not any("exclusion criteria" in c.text.lower() for c in criteria)

    def test_markdown_is_unescaped_in_output(self):
        criteria = split_eligibility(REAL_WORLD)
        assert any("> 365 days" in c.text for c in criteria)
        assert not any("\\" in c.text for c in criteria)

    def test_chunk_index_restarts_per_type(self):
        criteria = split_eligibility(REAL_WORLD)
        inc = [c.chunk_index for c in criteria if c.criterion_type == "inclusion"]
        exc = [c.chunk_index for c in criteria if c.criterion_type == "exclusion"]
        assert inc == [0, 1]
        assert exc == [0, 1]


class TestHeaderVariants:
    @pytest.mark.parametrize(
        "header",
        [
            "Inclusion Criteria:",
            "INCLUSION CRITERIA:",
            "inclusion criteria",
            "**Inclusion Criteria:**",
            "Key Inclusion Criteria:",
            "Inclusion:",
            "1. Inclusion Criteria:",
        ],
    )
    def test_recognises_header_spellings(self, header):
        text = f"{header}\n\n* Age 18 years or older at screening\n* Able to give consent"
        criteria = split_eligibility(text)
        assert len(criteria) == 2
        assert all(c.criterion_type == "inclusion" for c in criteria)

    def test_inline_headers_without_newlines(self):
        """~5% of records run the headers inline rather than on their own line."""
        text = ("Inclusion Criteria: Adults aged 18 years or older with type 2 diabetes. "
                "Exclusion Criteria: Pregnant or breastfeeding women at screening.")
        criteria = split_eligibility(text)
        kinds = {c.criterion_type for c in criteria}
        assert kinds == {"inclusion", "exclusion"}
        assert any("diabetes" in t for t in _texts(criteria, "inclusion"))
        assert any("Pregnant" in t for t in _texts(criteria, "exclusion"))


class TestBulletStyles:
    def test_star_bullets(self):
        text = "Inclusion Criteria:\n* Age 18 years or older\n* Confirmed diagnosis of disease"
        assert len(split_eligibility(text)) == 2

    def test_numbered_bullets(self):
        """15% of real records number their criteria instead of bulleting them."""
        text = ("Inclusion Criteria:\n1. Age 18 years or older at screening\n"
                "2. Confirmed diagnosis of type 2 diabetes\n3. Able to provide written consent")
        criteria = split_eligibility(text)
        assert len(criteria) == 3
        assert all(c.criterion_type == "inclusion" for c in criteria)

    def test_dash_bullets(self):
        text = "Inclusion Criteria:\n- Age 18 years or older\n- Confirmed diagnosis of disease"
        assert len(split_eligibility(text)) == 2

    def test_paragraph_separated_criteria(self):
        text = ("Inclusion Criteria:\n\nAge 18 years or older at screening.\n\n"
                "Confirmed diagnosis of type 2 diabetes mellitus.")
        assert len(split_eligibility(text)) == 2

    def test_decimal_at_line_start_is_not_a_bullet(self):
        """'2.5 mg/dL' must not be mistaken for list item number 2."""
        text = ("Inclusion Criteria:\n"
                "1. Serum creatinine below the upper limit of normal\n"
                "2.5 mg/dL threshold applies to all enrolled participants")
        criteria = split_eligibility(text)
        assert any("2.5 mg/dL" in c.text for c in criteria)


class TestDegenerateInput:
    def test_no_headers_are_tagged_unspecified_not_guessed(self):
        """~5% of trials have no headers. Guessing 'inclusion' would inject
        false facts, so these are tagged explicitly instead."""
        text = "* Age 18 years or older at screening\n* Able to provide written consent"
        criteria = split_eligibility(text)
        assert criteria
        assert all(c.criterion_type == "unspecified" for c in criteria)

    def test_inclusion_only_still_parses(self):
        """3.5% of records have an inclusion header and no exclusion header."""
        text = "Inclusion Criteria:\n* Age 18 years or older\n* Confirmed diagnosis of disease"
        criteria = split_eligibility(text)
        assert criteria
        assert all(c.criterion_type == "inclusion" for c in criteria)

    def test_preamble_before_first_header_is_unspecified(self):
        text = ("This study enrolls adults at participating centres worldwide.\n\n"
                "Inclusion Criteria:\n* Age 18 years or older at screening")
        criteria = split_eligibility(text)
        assert criteria[0].criterion_type == "unspecified"
        assert criteria[-1].criterion_type == "inclusion"

    @pytest.mark.parametrize("raw", [None, "", "   ", "\n\n"])
    def test_empty_input_returns_empty_list(self, raw):
        assert split_eligibility(raw) == []

    def test_shapeless_fragments_are_dropped(self):
        """No letters, or too short to be words -- not real criteria."""
        text = "Inclusion Criteria:\n* 18+\n* --\n* Confirmed diagnosis of type 2 diabetes"
        criteria = split_eligibility(text)
        assert len(criteria) == 1
        assert "diabetes" in criteria[0].text

    @pytest.mark.parametrize(
        "short_criterion",
        ["Pregnancy", "Prisoners", "Minors", "Hypertension", "Active cancer", "Breastfeeding"],
    )
    def test_short_but_real_criteria_survive(self, short_criterion):
        """Regression guard. A 15-char minimum silently discarded ~0.5 real
        criteria per trial -- and 'Pregnancy' is one of the most common
        exclusions in all of clinical research."""
        text = f"Exclusion Criteria:\n* {short_criterion}\n* Any other serious medical condition"
        criteria = split_eligibility(text)
        assert short_criterion in [c.text for c in criteria]

    @pytest.mark.parametrize("subheader", ["Part 1:", "Cohort A:", "For all participants:"])
    def test_subheadings_are_dropped(self, subheader):
        """A chunk ending in a colon introduces the items below it; those items
        are captured separately, so the label itself carries no information."""
        text = f"Inclusion Criteria:\n* {subheader}\n* Age 18 years or older at screening"
        criteria = split_eligibility(text)
        assert subheader not in [c.text for c in criteria]
        assert len(criteria) == 1

    def test_mid_sentence_mention_does_not_split_document(self):
        """'...meets any exclusion criteria...' inside a bullet is prose, not a
        section header. Treating it as one would mistag everything after it."""
        text = ("Inclusion Criteria:\n"
                "* Age 18 years or older at the time of screening\n"
                "* Participants who meet any exclusion criteria will be withdrawn from study")
        criteria = split_eligibility(text)
        assert all(c.criterion_type == "inclusion" for c in criteria)

    def test_oversized_block_is_split_by_sentence(self):
        long_block = " ".join(f"Criterion sentence number {i} applies here." for i in range(200))
        criteria = split_eligibility(f"Inclusion Criteria:\n{long_block}", max_chars=500)
        assert len(criteria) > 1
        assert all(c.char_len <= 500 for c in criteria)


def test_whitespace_is_normalised():
    text = "Inclusion Criteria:\n* Age   18 years\n     or older at screening"
    criteria = split_eligibility(text)
    assert "  " not in criteria[0].text
