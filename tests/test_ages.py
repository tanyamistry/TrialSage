"""Tests for age normalisation.

The critical distinction under test is None (no age stated) vs 0.0 (birth).
Conflating them would make "trials open to newborns" match every trial that
simply never specified a lower bound.
"""

import pytest

from trialsage.ingest.ages import describe_age, parse_age_to_years


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("18 Years", 18.0),
        ("65 Years", 65.0),
        ("120 Years", 120.0),
        ("1 Year", 1.0),          # singular, as the API really emits it
        ("6 Months", 0.5),
        ("18 Months", 1.5),
        ("2 Weeks", 0.038),
        ("30 Days", 0.082),
        ("1 Hour", 0.0),          # rounds to 0 at 3dp but is not None
        ("18 years", 18.0),       # lowercase
        ("  40 Years  ", 40.0),   # surrounding whitespace
        ("18", 18.0),             # bare number -> assume years
    ],
)
def test_parses_known_ages(raw, expected):
    assert parse_age_to_years(raw) == pytest.approx(expected, abs=0.001)


@pytest.mark.parametrize("raw", ["N/A", "n/a", "NA", "", "   ", None, "None", "-"])
def test_absent_ages_become_none(raw):
    assert parse_age_to_years(raw) is None


def test_zero_is_a_real_value_not_absent():
    """0 Years means birth -- neonatal trials use it, so it must not be None."""
    result = parse_age_to_years("0 Years")
    assert result == 0.0
    assert result is not None


def test_none_and_zero_are_distinguishable():
    """The bug this guards against: `if not age` treats both as missing."""
    assert parse_age_to_years("0 Years") is not parse_age_to_years("N/A")


@pytest.mark.parametrize("raw", ["eighteen years", "18 Parsecs", "??", "Years"])
def test_unparseable_returns_none_rather_than_raising(raw):
    assert parse_age_to_years(raw) is None


def test_month_conversion_is_exact_twelfths():
    assert parse_age_to_years("12 Months") == 1.0
    assert parse_age_to_years("3 Months") == pytest.approx(0.25, abs=0.001)


def test_describe_age_roundtrip():
    assert describe_age(None) == "not stated"
    assert describe_age(0.0) == "birth"
    assert describe_age(18.0) == "18 years"
    assert describe_age(0.5) == "6 months"
