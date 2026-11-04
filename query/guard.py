"""Pre-retrieval guards (implementation.md §7.1, PRD FR-7.1–7.6).

Order is part of the contract: **truncate → PII redact → classify**. No raw
identifier ever leaves this module — `GuardResult.clean` is the only text the
service, retriever, prompts, and artifacts are allowed to see.

The advice and performance classifiers are deterministic rules. They are
deliberately conservative: over-refusing a factual question is safer than
letting an opinion or a return figure through (architecture.md §10).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Specific patterns run before the broad "account" catch-all, otherwise a PAN
# or phone number would be labelled as an account number.
PII = {
    "pan": r"\b[A-Z]{5}[\s-]?[0-9]{4}[\s-]?[A-Z]\b",
    # Aadhaar is only labelled when it keeps its 4-4-4 grouping; a bare
    # 12-digit run is caught by the account rule below (still redacted, just a
    # less specific label).
    "aadhaar": r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b",
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
    "phone": r"\b(?:\+?91[\s-]?)?[6-9]\d{9}\b",
    "otp": r"\b(?:otp|one[\s-]?time[\s-]?password)\b\D{0,10}\d{4,6}\b",
    "account": r"\b\d{9,18}\b",
}

ADVICE = [
    r"\bshould i\b",
    r"\bis it good\b",
    r"\bbest\b",
    r"\bworth it\b",
    r"\brecommend",
    r"\bcompare\b",
    r"\bwill it give\b",
    r"\bwill (?:this|the) (?:fund )?give\b",
    r"\bgood returns?\b",
    r"\bbuy\b",
    r"\bsell\b",
    r"\bhold\b",
    r"\bportfolio\b",
    r"\binvest in\b",
    r"\bis (?:this|it) safe\b",
    r"\brisky\b",
]

PERFORMANCE = [
    r"\breturns?\b",
    r"\bcagr\b",
    r"\bperformance\b",
    r"\bprofit\b",
    r"\bgains?\b",
    r"\bhow much (?:will|would|could|has|did|can)\b",
    r"\b1[\s-]?year\b.*\breturn",
]

NON_LATIN = re.compile(
    r"[\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\u0900-\u097F"
    r"\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]")
LATIN = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class GuardResult:
    pii: bool
    advice: bool
    performance: bool
    language_ok: bool
    truncated: bool
    clean: str
    redactions: tuple[str, ...] = ()

    def to_json(self) -> dict:
        """Trace-friendly view. Values are labels, never the matched text."""
        return {
            "pii": self.pii,
            "advice": self.advice,
            "performance": self.performance,
            "language_ok": self.language_ok,
            "truncated": self.truncated,
            "redactions": list(self.redactions),
        }


def _language_ok(text: str) -> bool:
    """English-only heuristic: reject non-Latin scripts, allow Hinglish.

    A real language detector would be another dependency; the demo's scope is
    English questions, and rejecting unknown scripts is the part that matters
    for not silently mis-answering.
    """
    if NON_LATIN.search(text):
        return False
    if not any(ch.isalpha() for ch in text):
        return True
    return bool(LATIN.search(text))


def check(question: str, cfg) -> GuardResult:
    limit = int(getattr(cfg.safety, "max_input_chars", 500))
    truncated = len(question) > limit
    raw = question[:limit]

    language_ok = _language_ok(raw)
    clean = raw
    redactions: list[str] = []

    if cfg.safety.enable_pii_guard:
        for name, pattern in PII.items():
            clean, count = re.subn(
                pattern, f"[REDACTED_{name.upper()}]", clean, flags=re.I)
            if count:
                redactions.append(name)

    advice = bool(
        cfg.safety.enable_advice_guard
        and any(re.search(pattern, clean, re.I) for pattern in ADVICE))
    performance = any(re.search(pattern, clean, re.I)
                      for pattern in PERFORMANCE)

    return GuardResult(
        pii=bool(redactions),
        advice=advice,
        performance=performance,
        language_ok=language_ok,
        truncated=truncated,
        clean=clean,
        redactions=tuple(redactions),
    )
