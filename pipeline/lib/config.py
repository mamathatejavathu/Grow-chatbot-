"""Configuration loader: config.yaml + .env (implementation.md §1.3, §1.4).

Rules enforced here:
  * Unknown keys are rejected, so a typo like `min_relevance_threshold` fails
    loudly instead of silently falling back to a default.
  * `.env` wins over `config.yaml` for `llm.provider`, so the demo can run
    `LLM_PROVIDER=none` without editing the committed config.
  * `llm.model` is read from the environment at access time, never stored.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_CONFIG_PATH = "config.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppCfg(_Section):
    name: str
    disclaimer: str
    max_answer_sentences: int = Field(default=3, ge=1, le=6)
    example_questions: list[str] = Field(min_length=3, max_length=3)


class SourcesCfg(_Section):
    file: str = "sources.csv"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    user_agent: str
    request_delay_seconds: float = Field(default=2.0, ge=0)


class ChunkingCfg(_Section):
    strategy: str = "auto"
    chunk_size: int = Field(default=600, ge=100)   # CHARACTERS, not tokens
    chunk_overlap: int = Field(default=80, ge=0)
    min_chunk_chars: int = Field(default=20, ge=0)

    @field_validator("strategy")
    @classmethod
    def _known_strategy(cls, v: str) -> str:
        allowed = {"auto", "field_aware", "recursive"}
        if v not in allowed:
            raise ValueError(f"chunking.strategy must be one of {sorted(allowed)}")
        return v


class EmbeddingCfg(_Section):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = Field(default=32, ge=1)
    normalize: bool = True
    cache_dir: str = "data/embeddings"


class VectorDbCfg(_Section):
    provider: str = "chromadb"
    persist_dir: str = "data/chroma"
    collection: str = "groww_mf_faq"
    distance: str = "cosine"


class RetrievalCfg(_Section):
    top_k: int = Field(default=5, ge=1, le=20)
    min_relevance_score: float = Field(default=0.35, ge=0.0, le=1.0)
    filter_by_scheme: bool = True
    dedupe_by_field: bool = True


class LlmCfg(_Section):
    provider: str = "groq"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=300, ge=1)
    timeout_seconds: int = Field(default=30, ge=1)

    @property
    def model(self) -> str:
        """Model id always comes from .env (LLM_MODEL), never config.yaml."""
        return os.getenv("LLM_MODEL", "").strip()


class SafetyCfg(_Section):
    enable_pii_guard: bool = True
    enable_advice_guard: bool = True
    max_input_chars: int = Field(default=500, ge=50, le=5000)
    refuse_message: str
    pii_message: str
    performance_message: str
    language_message: str


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: AppCfg
    sources: SourcesCfg
    chunking: ChunkingCfg
    embedding: EmbeddingCfg
    vector_db: VectorDbCfg
    retrieval: RetrievalCfg
    llm: LlmCfg
    safety: SafetyCfg


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load, validate, apply env overrides, and ensure output dirs exist."""
    load_dotenv(override=False)

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found - run this from the repository root")

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cfg = Config.model_validate(raw)

    # .env wins for the provider so `LLM_PROVIDER=none` works without edits.
    env_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if env_provider:
        cfg.llm.provider = env_provider

    for d in (cfg.sources.raw_dir, cfg.sources.processed_dir,
              cfg.embedding.cache_dir, cfg.vector_db.persist_dir,
              "artifacts/runs"):
        Path(d).mkdir(parents=True, exist_ok=True)

    return cfg
