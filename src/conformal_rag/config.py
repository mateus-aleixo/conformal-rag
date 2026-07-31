"""Central configuration. Environment variables over flags over defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("CRAG_ROOT", Path.cwd()))


@dataclass(frozen=True)
class Config:
    db_path: Path = field(default_factory=lambda: ROOT / "data" / "index.db")
    trace_dir: Path = field(default_factory=lambda: ROOT / "traces")

    # chunking
    chunk_chars: int = 1200
    chunk_overlap: int = 200

    # retrieval
    k_bm25: int = 20
    k_vec: int = 20
    k_final: int = 5
    rrf_k: int = 60  # reciprocal-rank-fusion constant

    # LLM — provider-agnostic. "stub" | "ollama" | "openai"
    llm_provider: str = field(default_factory=lambda: os.environ.get("CRAG_LLM", "stub"))
    ollama_url: str = field(
        default_factory=lambda: os.environ.get("CRAG_OLLAMA_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.environ.get("CRAG_OLLAMA_MODEL", "qwen2.5:3b-instruct")
    )
    openai_base: str = field(
        default_factory=lambda: os.environ.get("CRAG_OPENAI_BASE", "https://api.openai.com/v1")
    )
    openai_model: str = field(
        default_factory=lambda: os.environ.get("CRAG_OPENAI_MODEL", "gpt-4o-mini")
    )
    openai_key: str = field(default_factory=lambda: os.environ.get("CRAG_OPENAI_KEY", ""))

    # conformal gate
    alpha: float = 0.10  # target wrong-answer risk among answered questions
    min_group: int = 30  # Mondrian small-group fallback, as in conformal-rul

    # agent
    max_steps: int = 6
    rul_api: str = field(
        default_factory=lambda: os.environ.get(
            "CRAG_RUL_API", "https://aao1ufi805.execute-api.eu-west-1.amazonaws.com"
        )
    )


DEFAULT = Config()
