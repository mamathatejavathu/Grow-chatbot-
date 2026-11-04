"""Keyless provider: generation is handled extractively in `query/generator.py`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoneProvider:
    name: str = "none"
    model: str = ""
    reason: str = "LLM disabled (LLM_PROVIDER=none)"

    def complete(self, system: str, user: str, *, temperature: float,
                 max_tokens: int, timeout: int) -> str:
        raise RuntimeError(f"no LLM provider: {self.reason}")
