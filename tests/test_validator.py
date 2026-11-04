"""Validator tests (implementation.md §6.3, PRD FR-6.4/6.5/7.6).

The validator can only remove content and rebuild the citation. These tests pin
the output contract: ≤3 sentences, exactly one `Source:` line whose URL comes
from the retrieved hits, and the date of that cited hit.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline.lib.schemas import Hit
from query.validator import validate

URL_A = "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth"
URL_B = "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth"


def _hit(chunk_id="s03:f_expense_ratio:2", url=URL_A,
         text="HDFC Flexi Cap Fund — Expense ratio: 0.75%",
         fetched_at="2026-11-01T10:03:27Z", field="expense_ratio"):
    return Hit(
        chunk_id=chunk_id,
        score=0.9,
        field=field,
        url=url,
        text=text,
        metadata={"scheme_name": "HDFC Flexi Cap Fund",
                  "fetched_at": fetched_at},
    )


HITS = [_hit()]
TWO_HITS = [
    _hit(),
    _hit(chunk_id="s04:f_lock_in:0", url=URL_B,
         text="HDFC ELSS Tax Saver Fund — Lock-in: 3 years",
         fetched_at="2026-10-15T09:00:00Z", field="lock_in"),
]
CFG = SimpleNamespace(app=SimpleNamespace(max_answer_sentences=3))


def test_truncates_to_three_sentences():
    final, checks = validate("A. B. C. D. E.", HITS, CFG)

    assert checks["sentences"] == 3
    assert final.splitlines()[0] == "A. B. C."
    assert final.count("Source:") == 1


def test_replaces_a_foreign_url_with_the_retrieved_source():
    final, checks = validate(
        "Expense ratio is 0.75%. Source: https://evil.example/x", HITS, CFG)

    assert "evil.example" not in final
    assert checks["citation_valid"] is False
    assert checks["citation_replaced"] is True
    assert checks["citation"] == URL_A
    assert f"Source: {URL_A}" in final


def test_keeps_the_first_valid_url_and_drops_every_other_link():
    final, checks = validate(
        f"Expense ratio is 0.75%. Source: https://evil.example/x {URL_A}",
        HITS, CFG)

    assert checks["citation"] == URL_A
    assert final.count("Source:") == 1
    assert "evil.example" not in final


def test_performance_sentence_is_removed_and_flagged():
    final, checks = validate("It returned 12% annual returns.", HITS, CFG)

    assert checks["perf_claims"] is True
    assert checks["perf_removed"] == 1
    assert "12%" not in final
    assert checks["fallback_used"] is True
    assert "Expense ratio" in final


def test_performance_word_without_a_number_is_also_removed():
    final, checks = validate("The fund performance was strong.", HITS, CFG)

    assert checks["perf_removed"] == 1
    assert "performance" not in final.lower()


def test_date_comes_from_the_cited_hit_not_the_top_hit():
    draft = f"HDFC ELSS Tax Saver Fund — Lock-in: 3 years. Source: {URL_B}"
    final, checks = validate(draft, TWO_HITS, CFG)

    assert checks["citation"] == URL_B
    assert "Last updated from sources: 2026-10-15" in final


def test_empty_draft_falls_back_to_the_top_chunk():
    final, checks = validate("", HITS, CFG)

    assert checks["fallback_used"] is True
    assert "Expense ratio: 0.75%" in final
    assert checks["citation"] == URL_A


def test_empty_draft_with_a_performance_only_chunk_is_neutral():
    perf_hit = _hit(text="The fund gave 12% annual returns.",
                    field="general")
    final, checks = validate("", [perf_hit], CFG)

    assert "12%" not in final
    assert "I don't have that information in my sources." in final
    assert final.count("Source:") == 1


def test_contract_lines_never_leak_into_the_body():
    draft = (f"HDFC Flexi Cap Fund — Expense ratio: 0.75%\n"
             f"Source: {URL_A}\n"
             f"Last updated from sources: 2026-11-01")
    final, _ = validate(draft, HITS, CFG)

    body = final.split("Source:")[0]
    assert "Last updated" not in body
    assert "http" not in body


def test_no_hits_never_emits_an_empty_source_line():
    final, checks = validate("Some fact.", [], CFG)

    assert "Source:" not in final
    assert "Last updated from sources: n/a" in final
    assert checks["citation"] == ""
