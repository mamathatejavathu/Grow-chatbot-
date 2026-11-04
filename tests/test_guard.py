"""Guard tests (implementation.md §7.3, PRD FR-7.1–7.6).

The order is part of the contract: truncate → redact → classify. These tests
assert both the decision and the redacted text, because a correct flag with a
leaked identifier in `clean` is still a privacy failure.
"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from pipeline.lib.config import load_config
from query.guard import check


@pytest.fixture
def cfg():
    return load_config()


# --------------------------------------------------------------------------- #
# PII — all six patterns
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,label,raw", [
    ("My PAN is ABCDE1234F", "pan", "ABCDE1234F"),
    ("aadhaar 1234 5678 9012", "aadhaar", "1234 5678 9012"),
    ("OTP is 482913", "otp", "482913"),
    ("mail me at a@b.com", "email", "a@b.com"),
    ("call 9876543210", "phone", "9876543210"),
    ("my account 123456789012", "account", "123456789012"),
])
def test_pii_is_detected_redacted_and_not_leaked(cfg, text, label, raw):
    result = check(text, cfg)

    assert result.pii is True
    assert label in result.redactions
    assert f"[REDACTED_{label.upper()}]" in result.clean
    assert raw not in result.clean

    trace = json.dumps(result.to_json())
    assert raw not in trace
    assert label in trace


def test_lowercase_pan_is_still_pii(cfg):
    result = check("pan abcde1234f", cfg)
    assert result.pii is True
    assert "abcde1234f" not in result.clean.lower()


def test_redaction_runs_before_advice_classification(cfg):
    result = check("My PAN is ABCDE1234F, should I buy this fund?", cfg)
    assert result.pii is True
    assert result.advice is True
    assert "ABCDE1234F" not in result.clean


# --------------------------------------------------------------------------- #
# advice
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text", [
    "Should I buy HDFC Flexi Cap?",
    "Which fund is best?",
    "Compare these two funds",
    "Is this fund worth it?",
    "Will it give good returns?",
    "Recommend a fund for me",
    "Is this fund risky?",
])
def test_advice_is_refused(cfg, text):
    assert check(text, cfg).advice is True


@pytest.mark.parametrize("text", [
    "What is the expense ratio of HDFC Mid Cap?",
    "ELSS lock-in period?",
    "How do I download my capital gains statement?",
    "What is the exit load?",
    "Minimum SIP amount?",
    "What is the riskometer level of HDFC Flexi Cap?",
])
def test_factual_questions_are_not_advice(cfg, text):
    result = check(text, cfg)
    assert result.pii is False
    assert result.advice is False


# --------------------------------------------------------------------------- #
# performance
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text", [
    "What returns did this fund give?",
    "What is the 1-year return?",
    "CAGR of this fund?",
    "How much will I earn?",
    "Show me the performance",
])
def test_performance_questions_are_classified(cfg, text):
    assert check(text, cfg).performance is True


def test_plain_fact_is_not_performance(cfg):
    result = check("What is the exit load of HDFC ELSS Tax Saver?", cfg)
    assert result.performance is False


# --------------------------------------------------------------------------- #
# input limits and language
# --------------------------------------------------------------------------- #

def test_long_input_is_truncated_before_any_processing(cfg):
    result = check("a" * (cfg.safety.max_input_chars + 400), cfg)

    assert result.truncated is True
    assert len(result.clean) == cfg.safety.max_input_chars


@pytest.mark.parametrize("text", [
    "कृपया बताएं",
    "基金的费用比率是多少",
    "какой у фонда выход",
])
def test_non_latin_scripts_are_rejected(cfg, text):
    assert check(text, cfg).language_ok is False


def test_hinglish_in_latin_script_is_accepted(cfg):
    assert check("HDFC flexi cap ka expense ratio kya hai", cfg).language_ok is True


def test_empty_input_is_harmless(cfg):
    result = check("", cfg)
    assert result.pii is False
    assert result.advice is False
    assert result.language_ok is True
    assert result.clean == ""


# --------------------------------------------------------------------------- #
# switches
# --------------------------------------------------------------------------- #

def test_pii_guard_can_be_disabled(cfg):
    local = deepcopy(cfg)
    local.safety.enable_pii_guard = False

    result = check("My PAN is ABCDE1234F", local)
    assert result.pii is False
    assert "ABCDE1234F" in result.clean


def test_advice_guard_can_be_disabled(cfg):
    local = deepcopy(cfg)
    local.safety.enable_advice_guard = False

    assert check("Should I buy this fund?", local).advice is False
