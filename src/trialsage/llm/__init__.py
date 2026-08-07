"""LLM access. Import `get_llm` and never construct providers directly.

Roles let one component use a different model from the rest. That exists for a
specific reason: text-to-SQL is by far the hardest thing we ask a model to do,
and if the local 8B model turns out to be unreliable at it, the fix is to point
*only* the SQL agent at a hosted model by setting SQL_LLM_PROVIDER in .env --
leaving routing and synthesis local and free.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Type

from ..config import env
from .base import LLMError, LLMProvider, LLMResponse
from .providers import AnthropicProvider, OllamaProvider, OpenAIProvider

__all__ = ["get_llm", "describe_roles", "LLMProvider", "LLMResponse", "LLMError"]

_PROVIDERS: Dict[str, Type[LLMProvider]] = {
    "ollama": OllamaProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}

# Sensible per-provider defaults, used when no model is configured for a role.
_DEFAULT_MODELS = {
    "ollama": "llama3.1:8b",
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o-mini",
}


def _resolve(role: str) -> tuple[str, str]:
    """Return ``(provider, model)`` for a role, falling back to the defaults.

    A role-specific setting only wins if it is actually set; blank values in
    .env (which is how the template ships) fall through to the global setting.
    """
    prefix = "" if role == "default" else f"{role.upper()}_"
    provider = env(f"{prefix}LLM_PROVIDER") or env("LLM_PROVIDER", "ollama")
    provider = provider.strip().lower()

    if provider not in _PROVIDERS:
        raise LLMError(
            f"Unknown LLM provider {provider!r} for role {role!r}. "
            f"Choose one of: {', '.join(sorted(_PROVIDERS))}"
        )

    model = env(f"{prefix}LLM_MODEL")
    if not model:
        # Only inherit the global model if the provider also matches, otherwise
        # we would hand an Ollama model name to Anthropic.
        global_provider = env("LLM_PROVIDER", "ollama").strip().lower()
        model = env("LLM_MODEL") if provider == global_provider else ""
    return provider, (model or _DEFAULT_MODELS[provider])


@lru_cache(maxsize=8)
def get_llm(role: str = "default") -> LLMProvider:
    """Return the configured provider for a role (``default`` or ``sql``)."""
    provider, model = _resolve(role)
    return _PROVIDERS[provider](model)


def describe_roles(roles: tuple[str, ...] = ("default", "sql")) -> str:
    """Human-readable summary of what each role will use -- handy in logs."""
    lines = []
    for role in roles:
        try:
            provider, model = _resolve(role)
            lines.append(f"  {role:<8} -> {provider}:{model}")
        except LLMError as exc:
            lines.append(f"  {role:<8} -> ERROR: {exc}")
    return "\n".join(lines)
