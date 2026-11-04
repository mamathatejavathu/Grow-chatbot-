"""Output-contract enforcement (§6.3, PRD FR-6.4/6.5/7.6).

The LLM's citation is a hint, never truth. This module rebuilds the citation
from retrieved metadata, so a foreign URL cannot survive and every answer ends
with exactly one clickable official source plus the source date.

It can only remove content, never add facts. If filtering leaves nothing, the
top retrieved chunk is used verbatim (or a neutral refusal when even that
contains a performance claim).
"""

from __future__ import annotations

import re

URL_RE = re.compile(r"https?://[^\s<>\"')]+")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

# Percentage followed by a performance word ("12% annual return", "8% CAGR").
PERF_RE = re.compile(
    r"\d+(?:\.\d+)?\s*%\s*(?:p\.?\s*a\.?|annual(?:ised|ized)?|returns?|cagr)",
    re.I)
# Any sentence that talks about performance, even without a number.
PERF_WORD_RE = re.compile(
    r"\b(?:returns?|cagr|annual(?:ised|ized)?|performance|profit|gains?)\b",
    re.I)
SOURCE_LABEL_RE = re.compile(r"(?i)\b(?:source|reference|link)s?\s*:\s*")
UPDATED_RE = re.compile(r"(?i)last updated from sources\s*:.*")


def _clean_body(text: str) -> str:
    # The date line must go before the "source(s):" label rule, otherwise that
    # rule eats the "sources:" in "Last updated from sources:" and leaves the
    # date stranded in the answer body.
    text = UPDATED_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = SOURCE_LABEL_RE.sub(" ", text)
    text = text.replace("**", " ").replace("__", " ").replace("`", " ")
    return " ".join(text.split())


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_RE.split(text) if s.strip()]


def _has_perf(text: str) -> bool:
    return bool(PERF_RE.search(text) or PERF_WORD_RE.search(text))


def _date_for(citation: str, hits) -> str:
    for hit in hits:
        if hit.url == citation:
            fetched = (hit.metadata or {}).get("fetched_at", "")
            return fetched[:10] if fetched else "n/a"
    if hits:
        fetched = (hits[0].metadata or {}).get("fetched_at", "")
        return fetched[:10] if fetched else "n/a"
    return "n/a"


def validate(draft: str, hits, cfg) -> tuple[str, dict]:
    """Return `(final_answer, checks)` with exactly one metadata-derived source."""
    max_sentences = cfg.app.max_answer_sentences
    allowed = {hit.url for hit in hits}

    found = [url.rstrip(".,;:!?)]}") for url in URL_RE.findall(draft or "")]
    valid = [url for url in found if url in allowed]
    citation_valid = bool(valid)
    citation_replaced = not citation_valid
    citation = valid[0] if valid else (hits[0].url if hits else "")

    body = _clean_body(draft or "")
    perf_claims = _has_perf(body)

    sentences = []
    perf_removed = 0
    for sentence in _sentences(body):
        if _has_perf(sentence):
            perf_removed += 1
            continue
        sentences.append(sentence)
    sentences = sentences[:max_sentences]

    fallback_used = False
    if not sentences and hits:
        fallback = _clean_body(hits[0].text)
        if fallback and not _has_perf(fallback):
            sentences = _sentences(fallback)[:max_sentences]
            fallback_used = bool(sentences)
    if not sentences:
        sentences = ["I don't have that information in my sources."]

    answer_body = " ".join(sentences)
    date = _date_for(citation, hits)

    if hits:
        final = (f"{answer_body}\n"
                 f"Source: {citation}\n"
                 f"Last updated from sources: {date}")
    else:
        # The service only validates after a passing retrieval, but keeping this
        # branch honest means a caller can never emit an empty Source line.
        final = f"{answer_body}\nLast updated from sources: n/a"

    checks = {
        "sentences": len(sentences),
        "max_sentences": max_sentences,
        "citation": citation if hits else "",
        "citation_valid": citation_valid,
        "citation_replaced": citation_replaced,
        "perf_claims": perf_claims,
        "perf_removed": perf_removed,
        "fallback_used": fallback_used,
    }
    return final, checks
