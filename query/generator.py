"""Grounded generation with a first-class extractive fallback (§6.2).

`generate()` never raises for provider problems: a missing key, a timeout, a
429, or an empty completion returns the top retrieved chunk verbatim. Retrieval
is the load-bearing part; the LLM is an eloquence layer (architecture.md §11).

The returned `Generation` carries the prompt and raw response so the service
can persist `artifacts/generation/<run_id>.json` after validation (FR-6.7).
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field

from llm.base import get_provider
from query.prompts import SYSTEM, build_user


@dataclass
class Generation:
    draft: str
    mode: str                      # llm | extractive | extractive_fallback | no_context
    provider: str
    model: str
    prompt: str = ""
    raw_response: str = ""
    latency_ms: int = 0
    error: str = ""
    fallback_reason: str = ""
    metadata: dict = field(default_factory=dict)


def _date(hit) -> str:
    fetched = (hit.metadata or {}).get("fetched_at", "")
    return fetched[:10] if fetched else "n/a"


def _extractive(hits) -> str:
    """Top chunk verbatim + its own URL and fetch date."""
    hit = hits[0]
    return (f"{hit.text}\n"
            f"Source: {hit.url}\n"
            f"Last updated from sources: {_date(hit)}")


def _extractive_result(hits, mode: str, *, provider: str = "none",
                       error: str = "", fallback_reason: str = "",
                       started: float | None = None) -> Generation:
    return Generation(
        draft=_extractive(hits),
        mode=mode,
        provider=provider,
        model="",
        latency_ms=int((time.perf_counter() - started) * 1000)
        if started is not None else 0,
        error=error,
        fallback_reason=fallback_reason,
    )


def generate(question: str, hits, cfg, provider=None) -> Generation:
    started = time.perf_counter()

    if not hits:
        return Generation(draft="", mode="no_context", provider="none", model="",
                          latency_ms=0)

    if provider is None:
        try:
            provider = get_provider(cfg)
        except Exception as exc:                   # misconfigured / no adapter
            return _extractive_result(
                hits, "extractive_fallback", error=str(exc),
                fallback_reason="provider_unavailable", started=started)

    if provider.name == "none":
        return _extractive_result(
            hits, "extractive", provider="none",
            fallback_reason=getattr(provider, "reason", ""),
            started=started)

    prompt = f"{SYSTEM}\n\n{build_user(question, hits)}"
    try:
        raw = provider.complete(
            SYSTEM,
            build_user(question, hits),
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
            timeout=cfg.llm.timeout_seconds,
        )
    except Exception as exc:
        return _extractive_result(
            hits, "extractive_fallback", provider=provider.name, error=str(exc),
            fallback_reason="provider_error", started=started)

    if not (raw or "").strip():
        return _extractive_result(
            hits, "extractive_fallback", provider=provider.name,
            fallback_reason="empty_response", started=started)

    return Generation(
        draft=raw.strip(),
        mode="llm",
        provider=provider.name,
        model=getattr(provider, "model", ""),
        prompt=prompt,
        raw_response=raw,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
