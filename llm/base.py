"""LLM provider protocol and factory (architecture.md §8.1).

Only this package reads API keys. Query code depends on the protocol, so the
provider is swapped by editing `.env` (`LLM_PROVIDER`, `LLM_MODEL`) and never by
touching retrieval, generation, or the UI.

Shipped adapters: Groq (the PRD default) and `none` (extractive mode). A
provider that has no adapter raises a clear error; `generate()` then falls back
to the extractive answer instead of crashing the demo.
"""

from __future__ import annotations

import os
from typing import Protocol


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str, *, temperature: float,
                 max_tokens: int, timeout: int) -> str:
        ...


def get_provider(cfg, env: dict | None = None) -> LLMProvider:
    """Resolve `cfg.llm.provider` to an adapter.

    Missing key or model is **not** an exception: it returns the `none` adapter
    with a reason, which is exactly the keyless demo mode required by FR-6.8.
    """
    values = os.environ if env is None else env
    name = (getattr(cfg.llm, "provider", "none") or "none").strip().lower()

    if name in ("", "none"):
        from llm.providers.none_ import NoneProvider

        return NoneProvider()

    model = (getattr(cfg.llm, "model", "") or "").strip()

    if name == "groq":
        from llm.providers.none_ import NoneProvider

        api_key = (values.get("GROQ_API_KEY") or "").strip()
        if not api_key:
            return NoneProvider(reason="GROQ_API_KEY is not set")
        if not model:
            return NoneProvider(reason="LLM_MODEL is not set")

        from llm.providers.groq_ import GroqProvider

        return GroqProvider(api_key=api_key, model=model)

    raise ValueError(
        f"llm.provider={name!r} has no adapter - use 'groq' or 'none' "
        f"(add llm/providers/{name}.py to support it)")
