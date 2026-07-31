"""Provider-agnostic LLM access. Three clients, one protocol, no framework.

- StubLLM: deterministic, scriptable. Tests and CI run the full pipeline with it.
- OllamaClient: local models, the development default. Zero marginal cost.
- OpenAICompatClient: any OpenAI-compatible endpoint (OpenAI, Groq, vLLM, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import httpx

from .config import Config


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient(Protocol):
    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResult: ...


@dataclass
class StubLLM:
    """Returns scripted responses in order; repeats the last one when exhausted.

    Default script echoes a grounded refusal-safe answer so pipeline tests have a
    predictable shape without any model.
    """

    script: list[str] = field(default_factory=lambda: ["STUB ANSWER [1]"])
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResult:
        self.calls.append((system, user))
        idx = min(len(self.calls) - 1, len(self.script) - 1)
        return LLMResult(text=self.script[idx], model="stub")


class OllamaClient:
    def __init__(self, cfg: Config):
        self.url = cfg.ollama_url.rstrip("/")
        self.model = cfg.ollama_model

    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResult:
        resp = httpx.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return LLMResult(
            text=data["message"]["content"],
            model=self.model,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )


class OpenAICompatClient:
    def __init__(self, cfg: Config):
        self.base = cfg.openai_base.rstrip("/")
        self.model = cfg.openai_model
        self.key = cfg.openai_key

    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResult:
        resp = httpx.post(
            f"{self.base}/chat/completions",
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "model": self.model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return LLMResult(
            text=data["choices"][0]["message"]["content"],
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )


def get_llm(cfg: Config) -> LLMClient:
    if cfg.llm_provider == "stub":
        return StubLLM()
    if cfg.llm_provider == "ollama":
        return OllamaClient(cfg)
    if cfg.llm_provider == "openai":
        return OpenAICompatClient(cfg)
    raise ValueError(f"unknown LLM provider: {cfg.llm_provider!r}")
