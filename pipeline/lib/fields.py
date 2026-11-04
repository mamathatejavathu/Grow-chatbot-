"""Closed field vocabulary and label helpers (implementation.md §3.2).

The chunker only emits fields listed in `VOCABULARY`; everything else becomes
`general` prose or a `definition`. A closed list is what makes metadata
filtering and dedupe reliable at retrieval time.

Groww pages put the same label in two places: the real fact (e.g.
``Expense ratio`` -> ``1.02%``) and a glossary entry (``Expense ratio`` -> a
long sentence). `looks_like_definition()` plus `LABEL_PRIORITY` resolve that
without hard-coding line numbers.
"""

from __future__ import annotations

import re
from typing import NamedTuple

VOCABULARY = frozenset({
    "expense_ratio",
    "exit_load",
    "min_sip",
    "min_investment",
    "lock_in",
    "riskometer",
    "benchmark",
    "fund_manager",
    "aum",
    "scheme_count",
    "definition",
    "general",
})

# Label -> field. Keys are compared after `normalize_label()`.
LABELS = {
    "expense ratio": "expense_ratio",
    "total expense ratio": "expense_ratio",
    "exit load": "exit_load",
    "minimum sip": "min_sip",
    "min. sip": "min_sip",
    "min. for sip": "min_sip",
    "minimum investment": "min_investment",
    "minimum lumpsum investment": "min_investment",
    "min. for 1st investment": "min_investment",
    "min. for first investment": "min_investment",
    "min. for 2nd investment": "min_investment",
    "min. for second investment": "min_investment",
    "lock-in": "lock_in",
    "lock in": "lock_in",
    "lock-in period": "lock_in",
    "lock in period": "lock_in",
    "riskometer": "riskometer",
    "risk level": "riskometer",
    "benchmark": "benchmark",
    "fund benchmark": "benchmark",
    "fund manager": "fund_manager",
    "aum": "aum",
    "fund size (aum)": "aum",
    "total aum": "aum",
    "no of schemes": "scheme_count",
}

# Terms whose *definition* (rather than a value) may follow the label.
GLOSSARY_TERMS = frozenset({
    "expense ratio",
    "tax",
    "exit load",
    "stamp duty",
    "annualised returns",
    "absolute returns",
})

# Higher wins when one field is found more than once. Ties go to the last
# occurrence, which is what makes the final `Exit load` value win over the
# glossary definition that appears earlier on the page.
LABEL_PRIORITY = {
    "fund size (aum)": 3,       # scheme-level AUM
    "aum": 2,
    "total aum": 1,             # AMC-level AUM, appears on every scheme page
    "min. for 1st investment": 3,
    "min. for first investment": 3,
    "minimum investment": 2,
    "minimum lumpsum investment": 2,
    "min. for 2nd investment": 1,
    "min. for second investment": 1,
}

# A date is not the value of `Exit load`; Groww shows the load *history*
# dates before the current value.
DATE_RE = re.compile(
    r"^(?:"
    r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}"          # 01 Jan 2013
    r"|[A-Za-z]{3,9}\s+\d{4}"                   # Jul 2022
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"           # 01/01/2013
    r"|\d{4}-\d{2}-\d{2}"                       # 2013-01-01
    r")$"
)

RISK_RE = re.compile(
    r"\b(very high|high|moderately high|moderate|low to moderate|low)"
    r"\s+risk\b",
    re.IGNORECASE,
)
RISK_LEVELS = {
    "very high": "Very High",
    "high": "High",
    "moderately high": "Moderately High",
    "moderate": "Moderate",
    "low to moderate": "Low to Moderate",
    "low": "Low",
}

# "3Y Lock-in" and "Lock-in of 3 years".
LOCKIN_RE = re.compile(
    r"\b(\d+)\s*(Y|Years?|Months?)\s+lock[-\s]?in\b", re.IGNORECASE)
LOCKIN_RE_ALT = re.compile(
    r"\block[-\s]?in(?:\s+period)?\s*(?:of\s*)?(\d+)\s*(Y|Years?|Months?)\b",
    re.IGNORECASE,
)

# Fund-management block: initials, name, month year, "- Present".
MANAGER_INITIALS_RE = re.compile(r"^[A-Z]{1,3}$")
MANAGER_NAME_RE = re.compile(r"^[A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){1,3}$")
MANAGER_DATE_RE = re.compile(
    r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}$",
    re.IGNORECASE,
)

# Values that are page furniture rather than facts.
STOP_VALUES = frozenset({
    "view details",
    "education",
    "experience",
    "also manages these schemes",
    "fund management",
    "understand terms",
    "returns and rankings",
    "minimum investments",
    "investment objective",
    "check past data",
    "compare similar funds",
    "return calculator",
    "see all",
    "about",
})


class Fact(NamedTuple):
    """One candidate fact before de-duplication."""

    field: str
    label: str
    value: str
    line: int
    priority: int = 1


def normalize_label(line: str) -> str:
    """Lower-case, drop a trailing colon, and collapse whitespace."""
    text = line.strip().lower().rstrip(":").strip()
    return re.sub(r"\s+", " ", text)


def looks_like_definition(value: str) -> bool:
    """True for the long, sentence-like glossary entries.

    Groww's real values are short ("1.02%", "Nil", "NIFTY 500 ..."), while a
    glossary entry is a full sentence ending in a period.
    """
    text = value.strip()
    return len(text) >= 70 and len(text.split()) >= 10 and text.endswith(".")


def valid_value(value: str) -> bool:
    """Reject dates, page furniture, and another label."""
    text = value.strip()
    if not text or text == "--":
        return False
    if normalize_label(text) in LABELS or normalize_label(text) in GLOSSARY_TERMS:
        return False
    if DATE_RE.match(text):
        return False
    if normalize_label(text) in STOP_VALUES:
        return False
    return True


def priority_for(label: str) -> int:
    return LABEL_PRIORITY.get(label, 1)


def _lock_in_value(match: re.Match) -> str:
    count = int(match.group(1))
    unit = match.group(2).lower()
    period = "month" if unit.startswith("m") else "year"
    if count != 1:
        period += "s"
    return f"{count} {period}"


def header_facts(lines: list[str]) -> list[Fact]:
    """Facts that are not a label/value pair.

    The scheme header carries the riskometer and the ELSS lock-in; the fund
    management block carries the manager names.
    """
    facts: list[Fact] = []

    for i, line in enumerate(lines[:20]):
        match = RISK_RE.search(line)
        if match and not any(f.field == "riskometer" for f in facts):
            level = RISK_LEVELS[match.group(1).lower()]
            facts.append(Fact("riskometer", "Riskometer", f"{level} Risk", i, 2))

        match = LOCKIN_RE.search(line) or LOCKIN_RE_ALT.search(line)
        if match and not any(f.field == "lock_in" for f in facts):
            facts.append(Fact("lock_in", "Lock-in", _lock_in_value(match), i, 2))

    managers: list[str] = []
    first_line = 0
    for i in range(len(lines) - 2):
        if not (
            MANAGER_NAME_RE.match(lines[i])
            and MANAGER_DATE_RE.match(lines[i + 1])
            and lines[i + 2].strip() == "- Present"
            and i > 0
            and MANAGER_INITIALS_RE.match(lines[i - 1])
        ):
            continue
        if not managers:
            first_line = i
        if lines[i] not in managers:
            managers.append(lines[i])
    if managers:
        facts.append(Fact("fund_manager", "Fund manager",
                          ", ".join(managers), first_line, 1))

    return facts
