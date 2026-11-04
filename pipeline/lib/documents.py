"""Data Loading hand-off: read + validate `documents.jsonl` (implementation.md §3.1).

This module touches only disk. It exists so "documents loaded" is its own
inspectable stage output, and so a broken scrape surfaces here as a load
failure rather than as silently bad chunks.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.lib.io import read_jsonl

REQUIRED = ("doc_id", "url", "title", "source_type", "fetched_at",
            "content_hash", "text")

# `scheme_name` is the one field that may be empty: the AMC overview is not a
# scheme page. It must still be present (so the key exists in every record),
# and a `source_type == "scheme"` record must fill it.
OPTIONAL_EMPTY = ("scheme_name",)

MIN_TEXT_CHARS = 300


def load_documents(cfg) -> tuple[list[dict], dict]:
    """Read documents.jsonl, validate the contract, return (docs, report)."""
    path = Path(cfg.sources.processed_dir) / "documents.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run `python -m pipeline.run --stage load` first")

    docs: list[dict] = []
    report: dict = {
        "stage": "load_documents",
        "path": str(path),
        "read": 0,
        "valid": 0,
        "rejected": [],
        "warnings": [],
    }

    for i, d in enumerate(read_jsonl(path)):
        report["read"] += 1
        missing = [k for k in REQUIRED if not d.get(k)]
        missing += [k for k in OPTIONAL_EMPTY if k not in d]
        if d.get("source_type") == "scheme" and not d.get("scheme_name"):
            missing.append("scheme_name (required for source_type=scheme)")
        if missing:
            report["rejected"].append(
                {"index": i, "doc_id": d.get("doc_id"), "missing": missing})
            continue
        if len(d["text"]) < MIN_TEXT_CHARS:
            report["warnings"].append(
                f"{d['doc_id']}: short text ({len(d['text'])} chars)")
        docs.append(d)
        report["valid"] += 1

    if not docs:
        raise ValueError("no valid documents - every record failed validation")

    report["doc_ids"] = [d["doc_id"] for d in docs]
    return docs, report
