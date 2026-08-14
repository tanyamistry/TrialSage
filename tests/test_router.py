"""Tests for the rule-based router.

Only the deterministic half is tested here -- no LLM calls, so these run in
milliseconds and never flake. The rules matter beyond their own accuracy: they
are the fallback whenever the local model returns unparseable JSON, an invented
route, or nothing at all, so they have to be right on the obvious cases.
"""

import pytest

from trialsage.router.classify import (
    RouteDecision,
    _extract_json,
    classify_by_rules,
)


def route_of(question: str) -> str:
    return classify_by_rules(question).route


class TestStructured:
    @pytest.mark.parametrize("question", [
        "How many phase 3 diabetes trials started in 2024?",
        "How many trials are currently recruiting?",
        "What is the average enrollment of completed trials?",
        "Which sponsor runs the most trials?",
        "How many trials were withdrawn?",
        "What is the total enrollment across phase 3 trials?",
    ])
    def test_counting_and_aggregation(self, question):
        assert route_of(question) == "structured"


class TestSemantic:
    @pytest.mark.parametrize("question", [
        "Find trials whose eligibility mentions prior immunotherapy failure",
        "Which trials exclude patients with a history of autoimmune disease?",
        "Trials that allow patients who previously failed chemotherapy",
        "What are the eligibility criteria around prior transplant?",
    ])
    def test_eligibility_prose(self, question):
        assert route_of(question) == "semantic"


class TestHybrid:
    @pytest.mark.parametrize("question", [
        "Which recruiting phase 2 oncology trials in Massachusetts allow patients"
        " with a history of autoimmune disease?",
        "Are there any recruiting phase 3 trials that exclude prior immunotherapy?",
        "Which phase 1 oncology trials allow patients with prior transplant?",
    ])
    def test_both_signals_present(self, question):
        assert route_of(question) == "hybrid"

    def test_headline_hybrid_question_explains_itself(self):
        decision = classify_by_rules(
            "Which recruiting phase 2 oncology trials in Massachusetts allow "
            "patients with a history of autoimmune disease?")
        assert decision.route == "hybrid"
        assert "history of" in decision.reasoning or "allow" in decision.reasoning
        assert decision.confidence > 0.5


class TestFallbackBehaviour:
    def test_always_returns_a_decision(self):
        """The rules are the safety net; they must never raise."""
        for question in ["", "   ", "???", "hello", "a" * 500]:
            decision = classify_by_rules(question)
            assert isinstance(decision, RouteDecision)
            assert decision.route in ("structured", "semantic", "hybrid")

    def test_unclear_question_defaults_to_semantic_with_low_confidence(self):
        """Semantic is the safer default: it returns cited criteria the
        synthesizer can refuse to over-claim on, whereas a wrong SQL query
        returns a confident number that happens to be wrong."""
        decision = classify_by_rules("tell me about trials")
        assert decision.route == "semantic"
        assert decision.confidence < 0.5

    def test_source_is_always_labelled(self):
        assert classify_by_rules("how many trials?").source == "rules"

    def test_reasoning_is_never_empty(self):
        for question in ["how many trials?", "eligibility criteria", "???"]:
            assert classify_by_rules(question).reasoning.strip()


class TestJSONExtraction:
    """The LLM half returns JSON. A local 8B model returns it wrapped in
    markdown, prefixed with prose, or followed by commentary."""

    def test_plain_json(self):
        assert _extract_json('{"route": "structured"}')["route"] == "structured"

    def test_fenced_json(self):
        out = _extract_json('```json\n{"route": "semantic"}\n```')
        assert out["route"] == "semantic"

    def test_json_with_leading_prose(self):
        out = _extract_json('Here is my answer:\n{"route": "hybrid"}')
        assert out["route"] == "hybrid"

    def test_json_with_trailing_prose(self):
        out = _extract_json('{"route": "hybrid"} — hope that helps!')
        assert out["route"] == "hybrid"

    def test_nested_braces(self):
        out = _extract_json('{"route": "hybrid", "meta": {"a": 1}}')
        assert out["meta"]["a"] == 1

    @pytest.mark.parametrize("text", ["", "no json here", "{unclosed", "{'bad': quotes}"])
    def test_unparseable_returns_none(self, text):
        assert _extract_json(text) is None
