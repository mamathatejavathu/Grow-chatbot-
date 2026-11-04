"""Provider factory tests (implementation.md §6.2, architecture.md §8).

A missing key must be a supported mode, not a startup crash: the factory returns
the `none` adapter with a reason, and `generate()` turns that into an extractive
answer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm.base import get_provider
from llm.providers.none_ import NoneProvider


def _cfg(provider="groq", model="qwen/qwen3.8-27b"):
    return SimpleNamespace(
        llm=SimpleNamespace(provider=provider, model=model))


def test_none_provider_is_selected_explicitly():
    provider = get_provider(_cfg("none"), env={})

    assert isinstance(provider, NoneProvider)
    assert provider.name == "none"


def test_groq_without_a_key_degrades_to_none():
    provider = get_provider(_cfg("groq"), env={})

    assert isinstance(provider, NoneProvider)
    assert provider.name == "none"
    assert "GROQ_API_KEY" in provider.reason


def test_groq_without_a_model_degrades_to_none():
    provider = get_provider(_cfg("groq", model=""), env={"GROQ_API_KEY": "k"})

    assert isinstance(provider, NoneProvider)
    assert "LLM_MODEL" in provider.reason


def test_groq_with_key_and_model_returns_the_adapter():
    provider = get_provider(
        _cfg("groq"), env={"GROQ_API_KEY": "test-key"})

    assert provider.name == "groq"
    assert provider.model == "qwen/qwen3.8-27b"


def test_unsupported_provider_raises_a_clear_error():
    with pytest.raises(ValueError, match="has no adapter"):
        get_provider(_cfg("gemini"), env={})


def test_none_provider_complete_explains_itself():
    with pytest.raises(RuntimeError, match="no LLM provider"):
        NoneProvider().complete(
            "system", "user", temperature=0.0, max_tokens=10, timeout=1)
