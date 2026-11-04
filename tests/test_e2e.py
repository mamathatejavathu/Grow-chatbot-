"""End-to-end tests against the real corpus (implementation.md §7.3).

No network and no API key: `LLM_PROVIDER=none` makes generation extractive. The
real MiniLM model and the real Chroma store are used, so this is the test that
proves the five-URL index still answers the PRD questions with the right link.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm.providers.none_ import NoneProvider
from pipeline.lib.config import load_config
from query import service as service_module
from query.service import ask

AMC_URL = "https://groww.in/mutual-funds/amc/hdfc-mutual-funds"
DATE = "2026-11-01"


@pytest.fixture(scope="module")
def artifact_root(tmp_path_factory):
    """Keep real-corpus traces out of the repository for this whole module."""
    root = tmp_path_factory.mktemp("e2e-artifacts")

    import query.retriever as retriever_module

    originals = {
        retriever_module: retriever_module.write_json,
        service_module: service_module.write_json,
    }

    def redirect(real_write):
        def write(path, obj):
            target = Path(path)
            if target.parts and target.parts[0] == "artifacts":
                target = root / target
            real_write(target, obj)

        return write

    for module, real_write in originals.items():
        module.write_json = redirect(real_write)

    yield root

    for module, real_write in originals.items():
        module.write_json = real_write


@pytest.fixture(scope="module")
def cfg():
    cfg = load_config()
    cfg.llm.provider = "none"                 # no key, no network
    return cfg


@pytest.fixture(scope="module")
def retriever(cfg, artifact_root):
    from query.retriever import Retriever

    service_module.clear_cache()
    instance = Retriever(cfg)
    yield instance
    instance.close()
    service_module.clear_cache()


@pytest.mark.parametrize("question,expected_url", [
    ("expense ratio HDFC Flexi Cap", "hdfc-equity-fund-direct-growth"),
    ("ELSS lock-in", "hdfc-elss-tax-saver"),
    ("exit load HDFC Mid Cap", "hdfc-mid-cap-fund-direct-growth"),
])
def test_answer_has_the_correct_source(question, expected_url, cfg, retriever):
    result = ask(question, cfg, retriever=retriever, provider=NoneProvider())

    assert result["gate"] == "pass"
    assert result["mode"] == "extractive"
    assert expected_url in result["answer"]
    assert result["answer"].count("Source:") == 1
    assert result["checks"]["sentences"] <= 3
    assert result["checks"]["perf_claims"] is False
    assert result["last_updated"] == DATE


def test_advice_uses_the_amc_page(cfg, retriever):
    result = ask("Should I buy HDFC Mid Cap?", cfg,
                 retriever=retriever, provider=NoneProvider())

    assert result["mode"] == "advice_refusal"
    assert f"Source: {AMC_URL}" in result["answer"]
    assert result["answer"].count("Source:") == 1


def test_low_relevance_question_does_not_guess(cfg, retriever):
    result = ask("What is the weather?", cfg,
                 retriever=retriever, provider=NoneProvider())

    assert result["mode"] == "no_context"
    assert result["gate"] == "fail"
    assert "I don't have that information in my sources." in result["answer"]
    assert f"Source: {AMC_URL}" in result["answer"]


def test_every_passing_answer_has_three_linked_traces(cfg, retriever,
                                                     artifact_root):
    result = ask("What is the riskometer level of HDFC Mid Cap Fund?", cfg,
                 retriever=retriever, provider=NoneProvider())
    run_id = result["run_id"]

    run = json.loads((artifact_root / "artifacts" / "runs" /
                      f"{run_id}.json").read_text(encoding="utf-8"))
    gen = json.loads((artifact_root / "artifacts" / "generation" /
                      f"{run_id}.json").read_text(encoding="utf-8"))
    retrieval = json.loads((artifact_root / "artifacts" / "retrieval" /
                            f"{run_id}.json").read_text(encoding="utf-8"))

    assert run["generation_trace"] == f"artifacts/generation/{run_id}.json"
    assert run["retrieval_trace"] == f"artifacts/retrieval/{run_id}.json"
    assert run["question"] == result["question"]
    assert run["hits"] == result["hits"]
    assert gen["validation"] == result["checks"]
    assert retrieval["relevance_gate"] == "pass"
    assert retrieval["filters"] == {"scheme_name": "HDFC Mid Cap Fund"}
    assert retrieval["best_score"] >= cfg.retrieval.min_relevance_score


def test_real_pii_is_not_written_anywhere(cfg, retriever, artifact_root):
    raw = "ZZZZZ9999Z"
    result = ask(f"My PAN is {raw}", cfg,
                 retriever=retriever, provider=NoneProvider())

    assert result["mode"] == "pii_refusal"
    assert raw not in result["question"]
    for path in artifact_root.rglob("*.json"):
        assert raw not in path.read_text(encoding="utf-8")
