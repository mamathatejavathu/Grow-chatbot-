"""Answer assembly: guard → retrieve → generate → validate → artifacts (§6.3).

One `run_id` ties together:

    artifacts/retrieval/<run_id>.json     (written by the retriever)
    artifacts/generation/<run_id>.json    (prompt, raw response, validation)
    artifacts/runs/<run_id>.json          (combined ask trace)

Only `GuardResult.clean` — never the raw question — is passed to retrieval,
prompts, or artifacts (PRD FR-7.5). Refusals use the `s01` AMC page as their one
link because the corpus contains no separate help/education page.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from pipeline.lib.io import read_jsonl, write_json
from pipeline.lib.runid import new_run_id
from query import guard, validator
from query.generator import generate

AMC_DOC_ID = "s01"
AMC_FALLBACK_URL = "https://groww.in/mutual-funds/amc/hdfc-mutual-funds"
NO_CONTEXT_MESSAGE = "I don't have that information in my sources."
EMPTY_MESSAGE = "Please enter a question."

_RETRIEVERS: dict[tuple, object] = {}


def get_retriever(cfg):
    """Process-wide retriever cache; loading MiniLM on every keystroke is 20 s."""
    key = (
        cfg.vector_db.persist_dir,
        cfg.vector_db.collection,
        cfg.embedding.model,
        bool(cfg.embedding.normalize),
    )
    if key not in _RETRIEVERS:
        from query.retriever import Retriever

        _RETRIEVERS[key] = Retriever(cfg)
    return _RETRIEVERS[key]


def clear_cache() -> None:
    _RETRIEVERS.clear()


@lru_cache(maxsize=8)
def _amc_meta(processed_dir: str, sources_file: str) -> tuple[str, str]:
    """URL + fetch date of the AMC overview page, for refusal citations."""
    for chunk in read_jsonl(Path(processed_dir) / "chunks.jsonl"):
        if chunk.get("doc_id") == AMC_DOC_ID:
            fetched = (chunk.get("fetched_at") or "")[:10]
            return chunk["url"], fetched or "n/a"

    try:
        with open(sources_file, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("id") == AMC_DOC_ID:
                    return row["url"], "n/a"
    except OSError:
        pass
    return AMC_FALLBACK_URL, "n/a"


def _answer_text(message: str, url: str, date: str) -> str:
    return (f"{message}\n"
            f"Source: {url}\n"
            f"Last updated from sources: {date}")


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _refusal(cfg, clean: str, run_id: str, mode: str, message: str,
             guard_info: dict, started: float, write_artifacts: bool, *,
             hits=(), filters: dict | None = None, gate: str = "skipped",
             best_score: float | None = None, write: bool = True) -> dict:
    url, date = _amc_meta(cfg.sources.processed_dir, cfg.sources.file)
    answer = _answer_text(message, url, date)
    checks = {
        "refused": True,
        "reason": mode,
        "sentences": 0,
        "citation": url,
        "citation_valid": True,
        "citation_replaced": False,
        "perf_claims": False,
        "fallback_used": False,
    }
    if best_score is not None:
        checks["best_score"] = round(best_score, 6)

    result = {
        "run_id": run_id,
        "question": clean,
        "answer": answer,
        "mode": mode,
        "checks": checks,
        "hits": [asdict(hit) for hit in hits],
        "filters": filters or {},
        "gate": gate,
        "guard": guard_info,
        "provider": "",
        "model": "",
        "prompt": "",
        "latency_ms": _elapsed_ms(started),
        "citation": url,
        "last_updated": date,
    }

    if write and write_artifacts:
        write_json(f"artifacts/runs/{run_id}.json", {
            "stage": "ask",
            "run_id": run_id,
            "question": clean,
            "guard": guard_info,
            "filters": result["filters"],
            "gate": gate,
            "hits": result["hits"],
            "draft": "",
            "answer": answer,
            "mode": mode,
            "provider": "",
            "model": "",
            "checks": checks,
            "latency_ms": result["latency_ms"],
            "retrieval_trace": None,
            "generation_trace": None,
        })
    return result


def ask(question: str, cfg, *, retriever=None, provider=None,
        write_artifacts: bool = True) -> dict:
    """Answer one question and return everything the UI/trace needs."""
    started = time.perf_counter()
    run_id = new_run_id()

    g = guard.check(question or "", cfg)
    guard_info = g.to_json()

    # PII first: nothing below this line may see or persist the raw value.
    if g.pii:
        return _refusal(cfg, g.clean, run_id, "pii_refusal",
                        cfg.safety.pii_message, guard_info, started,
                        write_artifacts)
    if not g.language_ok:
        return _refusal(cfg, g.clean, run_id, "language_refusal",
                        cfg.safety.language_message, guard_info, started,
                        write_artifacts)
    if not (question or "").strip():
        return _refusal(cfg, "", run_id, "empty_input", EMPTY_MESSAGE,
                        guard_info, started, write_artifacts, write=False)
    if g.advice:
        return _refusal(cfg, g.clean, run_id, "advice_refusal",
                        cfg.safety.refuse_message, guard_info, started,
                        write_artifacts)
    if g.performance:
        return _refusal(cfg, g.clean, run_id, "performance_refusal",
                        cfg.safety.performance_message, guard_info, started,
                        write_artifacts)

    retriever = retriever or get_retriever(cfg)
    hits, filters, gate = retriever.retrieve(
        g.clean, run_id=run_id, guard=guard_info)

    if gate != "pass" or not hits:
        return _refusal(
            cfg, g.clean, run_id, "no_context", NO_CONTEXT_MESSAGE,
            guard_info, started, write_artifacts, hits=hits, filters=filters,
            gate=gate,
            best_score=hits[0].score if hits else None)

    generation = generate(g.clean, hits, cfg, provider=provider)
    answer, checks = validator.validate(generation.draft, hits, cfg)
    latency_ms = _elapsed_ms(started)

    retrieval_trace = getattr(retriever, "last_trace_path", None)
    generation_trace = f"artifacts/generation/{run_id}.json"
    result = {
        "run_id": run_id,
        "question": g.clean,
        "answer": answer,
        "mode": generation.mode,
        "checks": checks,
        "hits": [asdict(hit) for hit in hits],
        "filters": filters,
        "gate": gate,
        "guard": guard_info,
        "provider": generation.provider,
        "model": generation.model,
        "prompt": generation.prompt,
        "latency_ms": latency_ms,
        "citation": checks["citation"],
        "last_updated": checks["citation"] and _date_for(checks["citation"],
                                                         hits),
    }

    if write_artifacts:
        write_json(generation_trace, {
            "stage": "generate",
            "run_id": run_id,
            "provider": generation.provider,
            "model": generation.model,
            "temperature": cfg.llm.temperature,
            "prompt": generation.prompt,
            "raw_response": generation.raw_response or generation.draft,
            "mode": generation.mode,
            "fallback_reason": generation.fallback_reason,
            "error": generation.error,
            "latency_ms": generation.latency_ms,
            "validation": checks,
        })
        write_json(f"artifacts/runs/{run_id}.json", {
            "stage": "ask",
            "run_id": run_id,
            "question": g.clean,
            "guard": guard_info,
            "filters": filters,
            "gate": gate,
            "hits": result["hits"],
            "draft": generation.draft,
            "answer": answer,
            "mode": generation.mode,
            "provider": generation.provider,
            "model": generation.model,
            "checks": checks,
            "latency_ms": latency_ms,
            "retrieval_trace": retrieval_trace,
            "generation_trace": generation_trace,
        })

    return result


def _date_for(citation: str, hits) -> str:
    for hit in hits:
        if hit.url == citation:
            fetched = (hit.metadata or {}).get("fetched_at", "")
            return fetched[:10] if fetched else "n/a"
    return "n/a"
