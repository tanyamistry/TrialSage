"""Tests for LLM provider/model resolution.

The role mechanism exists so the SQL agent can be pointed at a different model
from everything else. The subtle failure it has to avoid: inheriting the global
*model name* while overriding only the *provider*, which would hand an Ollama
model id like "llama3.1:8b" to Anthropic and fail at request time.

No network calls -- `_resolve` is pure configuration logic.
"""

import pytest

from trialsage.llm import _resolve, get_llm
from trialsage.llm.base import LLMError


@pytest.fixture(autouse=True)
def clear_caches():
    get_llm.cache_clear()
    yield
    get_llm.cache_clear()


@pytest.fixture
def env(monkeypatch):
    """Start from a clean slate so the developer's real .env cannot leak in."""
    for key in ("LLM_PROVIDER", "LLM_MODEL", "SQL_LLM_PROVIDER", "SQL_LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


class TestDefaults:
    def test_falls_back_to_ollama(self, env):
        assert _resolve("default") == ("ollama", "llama3.1:8b")

    def test_global_settings_are_used(self, env):
        env.setenv("LLM_PROVIDER", "ollama")
        env.setenv("LLM_MODEL", "mistral:7b")
        assert _resolve("default") == ("ollama", "mistral:7b")

    def test_provider_is_case_insensitive(self, env):
        env.setenv("LLM_PROVIDER", "  OLLAMA ")
        assert _resolve("default")[0] == "ollama"


class TestRoleOverrides:
    def test_sql_role_inherits_when_unset(self, env):
        env.setenv("LLM_PROVIDER", "ollama")
        env.setenv("LLM_MODEL", "llama3.1:8b")
        assert _resolve("sql") == _resolve("default")

    def test_blank_role_values_fall_through(self, env):
        """.env ships with SQL_LLM_PROVIDER= (empty), which must not win."""
        env.setenv("LLM_PROVIDER", "ollama")
        env.setenv("LLM_MODEL", "llama3.1:8b")
        env.setenv("SQL_LLM_PROVIDER", "")
        env.setenv("SQL_LLM_MODEL", "")
        assert _resolve("sql") == ("ollama", "llama3.1:8b")

    def test_sql_role_can_override_provider_alone(self, env):
        """The Phase 2 escape hatch: move only the SQL agent to a hosted model."""
        env.setenv("LLM_PROVIDER", "ollama")
        env.setenv("LLM_MODEL", "llama3.1:8b")
        env.setenv("SQL_LLM_PROVIDER", "anthropic")

        assert _resolve("default") == ("ollama", "llama3.1:8b")
        provider, model = _resolve("sql")
        assert provider == "anthropic"
        # Must NOT inherit the Ollama model name across a provider change.
        assert model != "llama3.1:8b"
        assert model == "claude-sonnet-5"

    def test_sql_role_can_override_both(self, env):
        env.setenv("LLM_PROVIDER", "ollama")
        env.setenv("SQL_LLM_PROVIDER", "openai")
        env.setenv("SQL_LLM_MODEL", "gpt-4o")
        assert _resolve("sql") == ("openai", "gpt-4o")

    def test_model_is_inherited_when_provider_matches(self, env):
        env.setenv("LLM_PROVIDER", "ollama")
        env.setenv("LLM_MODEL", "qwen2.5:14b")
        env.setenv("SQL_LLM_PROVIDER", "ollama")
        assert _resolve("sql") == ("ollama", "qwen2.5:14b")


class TestErrors:
    def test_unknown_provider_is_rejected_with_a_useful_message(self, env):
        env.setenv("LLM_PROVIDER", "not-a-provider")
        with pytest.raises(LLMError) as excinfo:
            _resolve("default")
        message = str(excinfo.value)
        assert "not-a-provider" in message
        assert "ollama" in message      # lists the valid options
