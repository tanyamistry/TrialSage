"""Concrete LLM providers: Ollama (local, default), Anthropic, OpenAI.

The hosted SDKs are imported lazily inside each class so that running fully
locally never requires them to be installed or an API key to be present.
"""

from __future__ import annotations

import time
from typing import Optional

import httpx

from ..config import env
from .base import LLMError, LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    """Local models via the Ollama HTTP API.

    Uses /api/chat rather than /api/generate so system prompts are passed as a
    proper role instead of being concatenated into the user turn. Ollama
    reports `prompt_eval_count` and `eval_count`, which gives us real token
    numbers for tracing rather than estimates.
    """

    name = "ollama"

    def __init__(self, model: str, base_url: Optional[str] = None, timeout: float = 180.0) -> None:
        super().__init__(model)
        self.base_url = (base_url or env("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages,
                      "stream": False, "options": options},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed ({self.base_url}): {exc}") from exc
        elapsed = time.perf_counter() - started

        payload = response.json()
        if "message" not in payload:
            raise LLMError(f"Unexpected Ollama response: {str(payload)[:200]}")

        return LLMResponse(
            text=payload["message"].get("content", ""),
            provider=self.name,
            model=self.model,
            latency_s=elapsed,
            prompt_tokens=payload.get("prompt_eval_count", 0),
            completion_tokens=payload.get("eval_count", 0),
            raw=payload,
        )


class AnthropicProvider(LLMProvider):
    """Claude via the Anthropic API."""

    name = "anthropic"

    def __init__(self, model: str) -> None:
        super().__init__(model)
        try:
            import anthropic  # noqa: PLC0415 -- lazy so local-only runs need no SDK
        except ImportError as exc:  # pragma: no cover
            raise LLMError("anthropic package not installed: pip install anthropic") from exc
        key = env("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set in .env")
        self._client = anthropic.Anthropic(api_key=key)

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens or 2048,
            temperature=temperature,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.perf_counter() - started
        text = "".join(block.text for block in message.content if block.type == "text")
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            latency_s=elapsed,
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
        )


class OpenAIProvider(LLMProvider):
    """GPT models via the OpenAI API."""

    name = "openai"

    def __init__(self, model: str) -> None:
        super().__init__(model)
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise LLMError("openai package not installed: pip install openai") from exc
        key = env("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set in .env")
        self._client = OpenAI(api_key=key)

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        started = time.perf_counter()
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = time.perf_counter() - started
        usage = completion.usage
        return LLMResponse(
            text=completion.choices[0].message.content or "",
            provider=self.name,
            model=self.model,
            latency_s=elapsed,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
        )
