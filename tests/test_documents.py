"""Tests for the Data Loading hand-off (implementation.md §3.1)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.lib.documents import MIN_TEXT_CHARS, load_documents


def _cfg(tmp_path: Path):
    return SimpleNamespace(
        sources=SimpleNamespace(processed_dir=str(tmp_path)))


def _write(tmp_path: Path, docs: list[dict]) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs),
        encoding="utf-8")


def _doc(**overrides) -> dict:
    doc = {
        "doc_id": "s02",
        "url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        "title": "HDFC Large Cap Fund",
        "scheme_name": "HDFC Large Cap Fund",
        "source_type": "scheme",
        "fetched_at": "2026-11-01T10:15:21Z",
        "content_hash": "sha256:abc",
        "text": "x" * (MIN_TEXT_CHARS + 10),
    }
    doc.update(overrides)
    return doc


def test_missing_file_is_a_loud_failure(tmp_path):
    with pytest.raises(FileNotFoundError, match="--stage load"):
        load_documents(_cfg(tmp_path))


def test_amc_document_may_have_empty_scheme_name(tmp_path):
    _write(tmp_path, [_doc(doc_id="s01", source_type="amc", scheme_name="")])
    docs, report = load_documents(_cfg(tmp_path))
    assert report["read"] == report["valid"] == 1
    assert report["rejected"] == []
    assert docs[0]["doc_id"] == "s01"


def test_scheme_document_must_fill_scheme_name(tmp_path):
    _write(tmp_path, [_doc(scheme_name="")])
    with pytest.raises(ValueError, match="no valid documents"):
        load_documents(_cfg(tmp_path))


def test_scheme_name_key_must_exist(tmp_path):
    doc = _doc()
    del doc["scheme_name"]
    _write(tmp_path, [doc])
    with pytest.raises(ValueError, match="no valid documents"):
        load_documents(_cfg(tmp_path))


def test_missing_required_field_is_rejected_not_fatal(tmp_path):
    good = _doc()
    bad = _doc(doc_id="s03", url="")
    _write(tmp_path, [good, bad])
    docs, report = load_documents(_cfg(tmp_path))
    assert [d["doc_id"] for d in docs] == ["s02"]
    assert report["rejected"] == [
        {"index": 1, "doc_id": "s03", "missing": ["url"]}]


def test_short_text_is_a_warning(tmp_path):
    _write(tmp_path, [_doc(text="too short")])
    docs, report = load_documents(_cfg(tmp_path))
    assert report["valid"] == 1
    assert report["warnings"] == ["s02: short text (9 chars)"]


def test_every_record_failing_raises(tmp_path):
    _write(tmp_path, [_doc(text="")])
    with pytest.raises(ValueError, match="every record failed validation"):
        load_documents(_cfg(tmp_path))
