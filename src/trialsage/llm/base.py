"""Provider-agnostic LLM interface.

Every component that calls a model goes through this, so switching between a
local Ollama model and a hosted API is a `.env` change rather than a code
change. It also means token counts and latency are captured in one place --
which is exactly what the Phase 5 tracing requirement needs, so we collect it
from the start rather than retrofitting it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LLMResponse:
    """A completion plus the metadata we trace on."""

    text: str
    provider: str
    model: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMError(RuntimeError):
    """Raised when a provider is unreachable or returns an unusable response."""


class LLMProvider(abc.ABC):
    """Minimal surface: one non-streaming completion call."""

    name: str = "base"

    def __init__(self, model: str) -> None:
        self.model = model

    @abc.abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Return a completion. Implementations must not raise on empty output."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} model={self.model!r}>"
