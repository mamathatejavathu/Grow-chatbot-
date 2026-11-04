"""Tests for the chunking stage (implementation.md §3).

Two layers:

* pure-function tests on synthetic Groww-shaped text (no disk, no network);
* a corpus test over the real `data/processed/chunks.jsonl`, which asserts the
  phase's pass criteria (closed vocabulary, no bare labels, s04 lock-in, ...).

The corpus test skips when the stage has not been run yet, so `pytest` still
works in a fresh checkout.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.lib.fields import (
    VOCABULARY,
    header_facts,
    looks_like_definition,
    normalize_label,
    valid_value,
)
from pipeline.lib.schemas import Chunk
from pipeline.stages.chunker import (
    best_facts,
    choose,
    line_chunks,
    parse_pairs,
    residual_lines,
)

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"


def _cfg(**overrides):
    defaults = dict(strategy="auto", chunk_size=600, chunk_overlap=80,
                    min_chunk_chars=20)
    defaults.update(overrides)
    return SimpleNamespace(chunking=SimpleNamespace(**defaults))


# --------------------------------------------------------------------------- #
# labels and values
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    ("Expense ratio", "expense ratio"),
    ("  MIN. FOR SIP:  ", "min. for sip"),
    ("Fund size (AUM)", "fund size (aum)"),
])
def test_normalize_label(raw, expected):
    assert normalize_label(raw) == expected


def test_looks_like_definition_only_for_sentences():
    sentence = ("A fee payable to a mutual fund house for managing your mutual "
                "fund investments. It is the total percentage of assets.")
    assert looks_like_definition(sentence)
    assert not looks_like_definition("Nil")
    assert not looks_like_definition("NIFTY 500 Total Return Index")
    assert not looks_like_definition("Exit load of 1% if redeemed within 1 year")


@pytest.mark.parametrize("value", [
    "", "--", "01 Jan 2013", "2013-01-01", "Expense ratio", "View details",
    "Fund management",
])
def test_valid_value_rejects_noise(value):
    assert not valid_value(value)


@pytest.mark.parametrize("value", [
    "1.02%", "Nil", "₹37,293.95 Cr", "NIFTY 500 Total Return Index",
    "Exit load of 1% if redeemed within 1 year",
])
def test_valid_value_accepts_facts(value):
    assert valid_value(value)


# --------------------------------------------------------------------------- #
# fact extraction
# --------------------------------------------------------------------------- #

def test_parse_pairs_never_skips_a_line():
    """The value must be the *immediately* next line."""
    lines = ["Expense ratio", "", "1.02%", "Exit load", "Nil"]
    facts = parse_pairs(lines)
    assert [f.field for f in facts] == ["exit_load"]
    assert facts[0].value == "Nil"


def test_glossary_definition_is_not_the_fact():
    definition = ("A fee payable to a mutual fund house for managing your "
                  "mutual fund investments. It is charged yearly.")
    lines = ["Exit load", definition, "Exit load", "Nil"]
    facts = best_facts(parse_pairs(lines))
    by_field = {f.field: f.value for f in facts}
    assert by_field["exit_load"] == "Nil"
    assert by_field["definition"] == definition


def test_aum_priority_prefers_scheme_level():
    lines = [
        "Total AUM", "₹9,34,075.66 Cr",
        "Fund size (AUM)", "₹37,293.95 Cr",
    ]
    facts = best_facts(parse_pairs(lines))
    assert [(f.field, f.value) for f in facts] == [
        ("aum", "₹37,293.95 Cr")]


def test_header_facts_reads_riskometer_and_lock_in():
    lines = ["HDFC ELSS Tax Saver Fund", "Equity", "Very High Risk",
             "3Y Lock-in", "NAV: 30 Oct '26"]
    facts = {f.field: f.value for f in header_facts(lines)}
    assert facts["riskometer"] == "Very High Risk"
    assert facts["lock_in"] == "3 years"


def test_header_facts_reads_fund_managers():
    lines = ["Fund management", "AK", "Amar Kalkundrikar", "Dec 2025",
             "- Present", "Education"]
    facts = {f.field: f.value for f in header_facts(lines)}
    assert facts["fund_manager"] == "Amar Kalkundrikar"


# --------------------------------------------------------------------------- #
# residual text
# --------------------------------------------------------------------------- #

def test_residual_skips_about_and_fund_management_blocks():
    lines = [
        "Expense ratio", "1.02%",
        "Fund management", "AK", "Amar Kalkundrikar", "Dec 2025", "- Present",
        "Education", "Bio text.",
        "About", "Repeats the AMC AUM.", ";", "Investment Objective",
        "The scheme seeks capital appreciation.",
    ]
    residual = residual_lines(lines, {0, 1})
    assert residual == ["Investment Objective",
                        "The scheme seeks capital appreciation."]


def test_line_chunks_respect_budget_and_never_split_a_line():
    lines = [f"line {i:02d} " + "x" * 30 for i in range(20)]
    chunks = line_chunks(lines, size=100, overlap=0, min_chars=1)
    assert all(len(c) <= 100 for c in chunks)
    emitted = [line for chunk in chunks for line in chunk.splitlines()]
    assert emitted == lines          # every line once, none split


def test_line_chunks_overlap_on_line_boundaries():
    lines = [f"line {i:02d} " + "x" * 20 for i in range(10)]
    chunks = line_chunks(lines, size=100, overlap=50, min_chars=1)
    assert len(chunks) > 1
    first = chunks[0].splitlines()
    second = chunks[1].splitlines()
    assert first[-1] in second          # at least one line of overlap


# --------------------------------------------------------------------------- #
# strategy selection
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("source_type,strategy", [
    ("scheme", "field_aware"),
    ("amc", "recursive"),
    ("education", "recursive"),
])
def test_choose_matches_source_type(source_type, strategy):
    doc = {"source_type": source_type}
    assert choose(doc, _cfg())[0] == strategy


def test_choose_respects_forced_config():
    assert choose({"source_type": "scheme"}, _cfg(strategy="recursive"))[0] \
        == "recursive"


# --------------------------------------------------------------------------- #
# the real corpus
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def chunks():
    if not CHUNKS.exists():
        pytest.skip("chunks.jsonl missing - run `python -m pipeline.run "
                    "--stage chunk`")
    return [json.loads(line) for line in
            CHUNKS.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_every_chunk_matches_the_schema(chunks):
    for row in chunks:
        chunk = Chunk(**row)                      # raises on a drifted field
        metadata = chunk.metadata()
        assert all(isinstance(v, (str, int, float, bool))
                   for v in metadata.values())
        assert metadata["field"] in VOCABULARY


def test_no_bare_label_chunks(chunks):
    bare = [c["chunk_id"] for c in chunks
            if len(c["text"].splitlines()) == 1 and c["text"].endswith(":")]
    assert bare == []


def test_every_scheme_has_field_chunks(chunks):
    for doc_id in ("s02", "s03", "s04", "s05"):
        fields = {c["field"] for c in chunks
                  if c["doc_id"] == doc_id and c["field"] != "general"}
        assert {"expense_ratio", "exit_load", "min_sip", "min_investment",
                "riskometer", "benchmark", "fund_manager", "aum"} <= fields


def test_elss_exit_load_is_nil_not_the_glossary(chunks):
    texts = [c["text"] for c in chunks
             if c["doc_id"] == "s04" and c["field"] == "exit_load"]
    assert len(texts) == 1
    assert texts[0].endswith("Exit load: Nil")


def test_elss_lock_in_is_three_years(chunks):
    texts = [c["text"] for c in chunks
             if c["doc_id"] == "s04" and c["field"] == "lock_in"]
    assert texts == ["HDFC ELSS Tax Saver Fund — Lock-in: 3 years"]


def test_residual_never_repeats_amc_aum_on_scheme_pages(chunks):
    for doc_id in ("s02", "s03", "s04", "s05"):
        residual = " ".join(c["text"] for c in chunks
                            if c["doc_id"] == doc_id
                            and c["field"] == "general")
        assert "Total AUM" not in residual
        assert "Asset Under Management" not in residual
        assert "Also manages" not in residual
