"""Streamlit UI tests (implementation.md §6.4, PRD FR-8).

`AppTest` runs the real `app.py`; only `query.service.ask` is patched, so the UI
wiring is tested without loading MiniLM. The pipeline expander, the single
clickable citation, the PII input clearing, and the index-not-ready path are all
asserted from the rendered element tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import query.service as service_module
from pipeline.lib.config import load_config

APP = str(Path(__file__).resolve().parents[1] / "app.py")
URL = "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth"
AMC_URL = "https://groww.in/mutual-funds/amc/hdfc-mutual-funds"
DATE = "2026-11-01"


def _result(mode="extractive", *, answer=None, prompt="", provider="none",
            model="", hits=None, checks=None):
    answer = answer or (
        "HDFC Flexi Cap Fund — Expense ratio: 0.75%\n"
        f"Source: {URL}\n"
        f"Last updated from sources: {DATE}")
    return {
        "run_id": "20261102T120000Z-abcd",
        "question": "What is the expense ratio of HDFC Flexi Cap?",
        "answer": answer,
        "mode": mode,
        "checks": checks or {
            "sentences": 1,
            "max_sentences": 3,
            "citation": URL,
            "citation_valid": True,
            "citation_replaced": False,
            "perf_claims": False,
            "perf_removed": 0,
            "fallback_used": False,
        },
        "hits": hits if hits is not None else [{
            "chunk_id": "s03:f_expense_ratio:2",
            "score": 0.904,
            "field": "expense_ratio",
            "url": URL,
            "text": "HDFC Flexi Cap Fund — Expense ratio: 0.75%",
            "metadata": {"scheme_name": "HDFC Flexi Cap Fund",
                         "fetched_at": "2026-11-01T10:03:27Z"},
        }],
        "filters": {"scheme_name": "HDFC Flexi Cap Fund"},
        "gate": "pass" if mode.startswith("extractive") or mode == "llm" else "skipped",
        "guard": {"pii": False, "advice": False, "performance": False,
                  "language_ok": True, "truncated": False, "redactions": []},
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "latency_ms": 12,
        "citation": URL,
        "last_updated": DATE,
    }


@pytest.fixture
def app(monkeypatch):
    calls = []

    def factory(result):
        def fake_ask(question, cfg, **kwargs):
            calls.append(question)
            return result(question) if callable(result) else result

        monkeypatch.setattr(service_module, "ask", fake_ask)
        return AppTest.from_file(APP, default_timeout=60).run(), calls

    return factory


def test_page_renders_title_disclaimer_examples_and_input(app):
    cfg = load_config()
    at, calls = app(_result())

    assert not at.exception
    assert calls == []
    assert at.title[0].value == cfg.app.name
    assert at.info[0].value == cfg.app.disclaimer
    assert [b.label for b in list(at.button)[:-1]] == cfg.app.example_questions
    assert at.button[-1].label == "Ask"
    assert at.text_input[0].label == "Your question"
    assert at.text_input[0].value == ""
    assert at.selectbox[0].value == "Any scheme / AMC overview"
    assert len(at.caption) == 0


def test_example_chip_asks_and_renders_one_clickable_citation(app):
    cfg = load_config()
    at, calls = app(_result())

    at.button[0].click().run()

    assert not at.exception
    assert calls == [cfg.app.example_questions[0]]
    assert at.text_input[0].value == cfg.app.example_questions[0]

    markdown = [m.value for m in at.markdown]
    assert any("Expense ratio: 0.75%" in value for value in markdown)
    assert any(f"**Source:** [{URL}]({URL})" in value for value in markdown)
    assert at.caption[0].value == f"Last updated from sources: {DATE}"

    assert len(at.expander) == 1
    assert at.expander[0].label == "Show pipeline"
    assert len(at.expander[0].dataframe) == 1
    assert len(at.expander[0].json) == 2


def test_form_submit_asks_the_typed_question(app):
    at, calls = app(_result())

    at.text_input[0].set_value("What is the exit load of HDFC Mid Cap Fund?")
    at.button[-1].click().run()

    assert not at.exception
    assert calls == ["What is the exit load of HDFC Mid Cap Fund?"]


def test_pii_refusal_clears_the_input_box(app):
    result = _result(
        "pii_refusal",
        answer=("Please don't share personal identifiers.\n"
                f"Source: {AMC_URL}\n"
                f"Last updated from sources: {DATE}"),
        provider="", model="", hits=[],
        checks={"refused": True, "reason": "pii_refusal",
                "citation": AMC_URL, "citation_valid": True, "sentences": 0},
    )
    at, calls = app(result)

    at.text_input[0].set_value("My PAN is ABCDE1234F")
    at.button[-1].click().run()

    assert not at.exception
    assert calls == ["My PAN is ABCDE1234F"]
    assert at.text_input[0].value == ""
    assert at.session_state["q_input"] == ""
    assert any("Please don't share personal identifiers." in m.value
               for m in at.markdown)
    assert any(AMC_URL in m.value for m in at.markdown)


def test_empty_submit_is_rejected_without_calling_the_service(app):
    at, calls = app(_result())

    at.button[-1].click().run()

    assert not at.exception
    assert calls == []
    assert at.error[0].value == service_module.EMPTY_MESSAGE


def test_missing_index_shows_a_build_instruction(app, monkeypatch):
    def fail(question, cfg, **kwargs):
        raise RuntimeError(
            "data/embeddings/manifest.json missing - run "
            "`python -m pipeline.run --stage all` first")

    monkeypatch.setattr(service_module, "ask", fail)
    at = AppTest.from_file(APP, default_timeout=60).run()

    at.button[0].click().run()

    assert not at.exception
    assert len(at.error) == 1
    assert "index is not ready" in at.error[0].value.lower()
    assert "--stage all" in at.error[0].value


def test_scheme_selector_prefills_the_question(app):
    at, calls = app(_result())

    at.selectbox[0].set_value("HDFC Mid Cap Fund").run()

    assert not at.exception
    assert at.text_input[0].value == "HDFC Mid Cap Fund"
    assert calls == []


def test_pipeline_expander_shows_the_llm_prompt_model_and_latency(app):
    prompt = "SYSTEM RULES\n\nCONTEXT:\n[1] expense ratio 0.75%"
    at, calls = app(_result(
        "llm", prompt=prompt, provider="groq",
        model="qwen/qwen3.8-27b"))

    at.button[0].click().run()

    expander = at.expander[0]
    assert len(expander.code) == 1
    assert expander.code[0].value == prompt
    summary = "\n".join(m.value for m in expander.markdown)
    assert "groq" in summary
    assert "qwen/qwen3.8-27b" in summary
    assert "12 ms" in summary
