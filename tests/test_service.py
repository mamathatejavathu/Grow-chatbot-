"""Answer-assembly tests (implementation.md §6.3, PRD FR-6.4–7.6).

A fake retriever keeps these tests fast; the real corpus is covered by
`test_e2e.py`. What is pinned here is the orchestration contract:

* guard before retrieval,
* only redacted text in artifacts,
* refusal links always come from the corpus (the `s01` AMC page),
* generation + validation + three linked artifacts for a passing query.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm.providers.none_ import NoneProvider
from pipeline.lib.config import load_config
from pipeline.lib.schemas import Hit
from query import service as service_module
from query.service import ask

URL = "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth"
AMC_URL = "https://groww.in/mutual-funds/amc/hdfc-mutual-funds"
DATE = "2026-11-01"


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture(autouse=True)
def artifact_root(monkeypatch, tmp_path):
    """Redirect every `artifacts/...` write in this module to `tmp_path`."""
    real_write_json = service_module.write_json

    def write(path, obj):
        target = Path(path)
        if target.parts and target.parts[0] == "artifacts":
            target = tmp_path / target
        real_write_json(target, obj)

    monkeypatch.setattr(service_module, "write_json", write)
    return tmp_path


def _hit():
    return Hit(
        chunk_id="s03:f_expense_ratio:2",
        score=0.904,
        field="expense_ratio",
        url=URL,
        text="HDFC Flexi Cap Fund — Expense ratio: 0.75%",
        metadata={"scheme_name": "HDFC Flexi Cap Fund",
                  "fetched_at": "2026-11-01T10:03:27Z"},
    )


class FakeRetriever:
    def __init__(self, hits=(), gate="pass", filters=None):
        self.hits = list(hits)
        self.gate = gate
        self.filters = filters or {"scheme_name": "HDFC Flexi Cap Fund"}
        self.last_trace_path = "artifacts/retrieval/fake.json"
        self.calls = []

    def retrieve(self, question, *, run_id=None, guard=None, **kwargs):
        self.calls.append(
            {"question": question, "run_id": run_id, "guard": guard})
        return list(self.hits), dict(self.filters), self.gate


class BoomProvider:
    name = "groq"
    model = "qwen/qwen3.8-27b"

    def complete(self, *args, **kwargs):
        raise RuntimeError("429 rate limited")


def _artifact(root, *parts):
    return json.loads(root.joinpath(*parts).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# factual path
# --------------------------------------------------------------------------- #

def test_factual_answer_is_cited_and_validated(cfg):
    retriever = FakeRetriever([_hit()])

    result = ask("What is the expense ratio of HDFC Flexi Cap?", cfg,
                 retriever=retriever, provider=NoneProvider())

    assert result["mode"] == "extractive"
    assert result["gate"] == "pass"
    assert result["answer"].count("Source:") == 1
    assert f"Source: {URL}" in result["answer"]
    assert f"Last updated from sources: {DATE}" in result["answer"]
    assert result["citation"] == URL
    assert result["last_updated"] == DATE
    assert result["checks"]["citation_valid"] is True
    assert result["checks"]["sentences"] <= 3
    assert result["checks"]["perf_claims"] is False

    assert len(retriever.calls) == 1
    assert retriever.calls[0]["question"] == (
        "What is the expense ratio of HDFC Flexi Cap?")
    assert retriever.calls[0]["guard"]["pii"] is False


def test_passing_query_writes_three_linked_artifacts(cfg, artifact_root):
    result = ask("What is the expense ratio of HDFC Flexi Cap?", cfg,
                 retriever=FakeRetriever([_hit()]), provider=NoneProvider())
    run_id = result["run_id"]

    run = _artifact(artifact_root, "artifacts", "runs", f"{run_id}.json")
    generation = _artifact(
        artifact_root, "artifacts", "generation", f"{run_id}.json")

    assert run["stage"] == "ask"
    assert run["question"] == result["question"]
    assert run["gate"] == "pass"
    assert run["mode"] == "extractive"
    assert run["answer"] == result["answer"]
    assert run["draft"] == _hit().text + (
        f"\nSource: {URL}\nLast updated from sources: {DATE}")
    assert run["checks"] == result["checks"]
    assert run["hits"] == [result["hits"][0]]
    assert run["retrieval_trace"] == "artifacts/retrieval/fake.json"
    assert run["generation_trace"] == f"artifacts/generation/{run_id}.json"

    assert generation["stage"] == "generate"
    assert generation["provider"] == "none"
    assert generation["prompt"] == ""
    assert generation["raw_response"].startswith(_hit().text)
    assert generation["mode"] == "extractive"
    assert generation["validation"] == result["checks"]
    assert "LLM disabled" in generation["fallback_reason"]


def test_provider_failure_falls_back_but_keeps_the_contract(cfg, artifact_root):
    result = ask("What is the expense ratio of HDFC Flexi Cap?", cfg,
                 retriever=FakeRetriever([_hit()]), provider=BoomProvider())

    assert result["mode"] == "extractive_fallback"
    assert result["answer"].count("Source:") == 1
    assert f"Source: {URL}" in result["answer"]
    assert result["checks"]["citation_valid"] is True

    generation = _artifact(
        artifact_root, "artifacts", "generation", f"{result['run_id']}.json")
    assert generation["mode"] == "extractive_fallback"
    assert "429" in generation["error"]
    assert generation["fallback_reason"] == "provider_error"


# --------------------------------------------------------------------------- #
# refusals — no retrieval, one official link
# --------------------------------------------------------------------------- #

def test_advice_refusal_skips_retrieval(cfg, artifact_root):
    retriever = FakeRetriever([_hit()])

    result = ask("Should I buy HDFC Mid Cap?", cfg,
                 retriever=retriever, provider=NoneProvider())

    assert result["mode"] == "advice_refusal"
    assert result["gate"] == "skipped"
    assert retriever.calls == []
    assert cfg.safety.refuse_message in result["answer"]
    assert f"Source: {AMC_URL}" in result["answer"]
    assert result["answer"].count("Source:") == 1
    assert result["last_updated"] == DATE

    run = _artifact(
        artifact_root, "artifacts", "runs", f"{result['run_id']}.json")
    assert run["mode"] == "advice_refusal"
    assert run["checks"]["refused"] is True
    assert run["generation_trace"] is None


def test_performance_refusal_uses_the_corpus_link(cfg):
    result = ask("What returns did this fund give?", cfg,
                 retriever=FakeRetriever([_hit()]), provider=NoneProvider())

    assert result["mode"] == "performance_refusal"
    assert cfg.safety.performance_message in result["answer"]
    assert f"Source: {AMC_URL}" in result["answer"]


def test_language_refusal(cfg):
    result = ask("कृपया बताएं", cfg,
                 retriever=FakeRetriever([_hit()]), provider=NoneProvider())

    assert result["mode"] == "language_refusal"
    assert cfg.safety.language_message in result["answer"]
    assert f"Source: {AMC_URL}" in result["answer"]


def test_empty_input_is_not_persisted(cfg, artifact_root):
    retriever = FakeRetriever([_hit()])

    result = ask("   ", cfg, retriever=retriever, provider=NoneProvider())

    assert result["mode"] == "empty_input"
    assert result["answer"].startswith(service_module.EMPTY_MESSAGE)
    assert retriever.calls == []
    assert not (artifact_root / "artifacts").exists()


# --------------------------------------------------------------------------- #
# PII — the important one
# --------------------------------------------------------------------------- #

def test_pii_is_refused_and_the_raw_value_is_never_stored(cfg, artifact_root):
    raw = "ABCDE1234F"
    retriever = FakeRetriever([_hit()])

    result = ask(f"My PAN is {raw}", cfg,
                 retriever=retriever, provider=NoneProvider())

    assert result["mode"] == "pii_refusal"
    assert retriever.calls == []
    assert raw not in result["question"]
    assert raw not in result["answer"]
    assert result["guard"]["redactions"] == ["pan"]
    assert "[REDACTED_PAN]" in result["question"]

    for path in artifact_root.rglob("*.json"):
        assert raw not in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# relevance gate
# --------------------------------------------------------------------------- #

def test_gate_failure_returns_no_context_and_the_amc_link(cfg, artifact_root):
    retriever = FakeRetriever([_hit()], gate="fail")

    result = ask("What is the weather?", cfg,
                 retriever=retriever, provider=NoneProvider())

    assert result["mode"] == "no_context"
    assert result["gate"] == "fail"
    assert service_module.NO_CONTEXT_MESSAGE in result["answer"]
    assert f"Source: {AMC_URL}" in result["answer"]
    assert result["checks"]["best_score"] == pytest.approx(0.904)
    assert len(result["hits"]) == 1
    assert not (artifact_root / "artifacts" / "generation").exists()


def test_empty_gate_result_also_uses_the_amc_link(cfg):
    result = ask("What is the weather?", cfg,
                 retriever=FakeRetriever([], gate="fail"),
                 provider=NoneProvider())

    assert result["mode"] == "no_context"
    assert result["hits"] == []
    assert f"Source: {AMC_URL}" in result["answer"]
