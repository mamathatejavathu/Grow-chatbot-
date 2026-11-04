"""Generator tests (implementation.md §6.2, PRD FR-6.1–6.8).

The extractive path is a feature, not error handling: no key, a 429, an empty
completion, or an unsupported provider must all produce a cited answer instead
of a crash. The `Generation` object must also carry the exact prompt so the
service can persist it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm.providers.none_ import NoneProvider
from pipeline.lib.schemas import Hit
from query.generator import generate
from query.prompts import SYSTEM

URL = "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth"

CFG = SimpleNamespace(
    llm=SimpleNamespace(temperature=0.0, max_tokens=300, timeout_seconds=30))


def _hit():
    return Hit(
        chunk_id="s03:f_expense_ratio:2",
        score=0.9,
        field="expense_ratio",
        url=URL,
        text="HDFC Flexi Cap Fund — Expense ratio: 0.75%",
        metadata={"scheme_name": "HDFC Flexi Cap Fund",
                  "fetched_at": "2026-11-01T10:03:27Z"},
    )


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, response="", error=None):
        self.response = response
        self.error = error
        self.calls = []

    def complete(self, system, user, *, temperature, max_tokens, timeout):
        self.calls.append({
            "system": system,
            "user": user,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        })
        if self.error is not None:
            raise self.error
        return self.response


def _assert_extractive(generation):
    assert generation.draft.startswith(_hit().text)
    assert f"Source: {URL}" in generation.draft
    assert "Last updated from sources: 2026-11-01" in generation.draft


def test_no_hits_means_no_context_and_no_call():
    generation = generate("anything", [], CFG, provider=FakeProvider("x"))

    assert generation.mode == "no_context"
    assert generation.draft == ""
    assert generation.provider == "none"


def test_no_provider_is_extractive_by_design():
    generation = generate("expense ratio?", [_hit()], CFG, provider=NoneProvider())

    assert generation.mode == "extractive"
    assert generation.provider == "none"
    assert generation.prompt == ""
    _assert_extractive(generation)


def test_llm_response_is_returned_with_the_exact_prompt():
    provider = FakeProvider(
        f"Expense ratio is 0.75%.\nSource: {URL}\n"
        "Last updated from sources: 2026-11-01")
    generation = generate("expense ratio?", [_hit()], CFG, provider=provider)

    assert generation.mode == "llm"
    assert generation.provider == "fake"
    assert generation.model == "fake-model"
    assert generation.raw_response.startswith("Expense ratio is 0.75%")

    assert SYSTEM in generation.prompt
    assert "CONTEXT:" in generation.prompt
    assert _hit().text in generation.prompt
    assert URL in generation.prompt
    assert "expense ratio?" in generation.prompt

    call = provider.calls[0]
    assert call["system"] == SYSTEM
    assert _hit().text in call["user"]
    assert call["temperature"] == CFG.llm.temperature
    assert call["max_tokens"] == CFG.llm.max_tokens
    assert call["timeout"] == CFG.llm.timeout_seconds


def test_provider_exception_falls_back_to_the_chunk():
    generation = generate("expense ratio?", [_hit()], CFG,
                          provider=FakeProvider(error=RuntimeError("429")))

    assert generation.mode == "extractive_fallback"
    assert generation.fallback_reason == "provider_error"
    assert "429" in generation.error
    _assert_extractive(generation)


def test_empty_completion_falls_back_to_the_chunk():
    generation = generate("expense ratio?", [_hit()], CFG,
                          provider=FakeProvider(response="   "))

    assert generation.mode == "extractive_fallback"
    assert generation.fallback_reason == "empty_response"
    _assert_extractive(generation)


def test_unavailable_provider_factory_falls_back(monkeypatch):
    def boom(cfg):
        raise ValueError("llm.provider='gemini' has no adapter")

    monkeypatch.setattr("query.generator.get_provider", boom)
    generation = generate("expense ratio?", [_hit()], CFG)

    assert generation.mode == "extractive_fallback"
    assert generation.fallback_reason == "provider_unavailable"
    assert "gemini" in generation.error
    _assert_extractive(generation)


def test_latency_is_reported():
    generation = generate("expense ratio?", [_hit()], CFG, provider=NoneProvider())
    assert generation.latency_ms >= 0
