"""
Unit tests for confidence-based stopping in the diagnostic flow.

`is_terminal_turn` decides when the interview closes — the model stops asking
follow-ups and presents its assessment. The contract under test:

    * emergency redirect always closes (priority 1),
    * reaching the gatekeeper's confidence threshold closes early (priority 2 —
      the primary, confidence-based signal that replaced the fixed turn pattern),
    * the diagnostic turn cap is only a hard backstop (priority 3),
    * a low-confidence, mid-conversation turn stays open.

`parse_diagnostic_confidence` normalises the analyzer's raw value into 0–100.
"""

from __future__ import annotations

import pytest

from graphrag.config.settings import settings
from graphrag.domain.messages import is_terminal_turn, parse_diagnostic_confidence


# ---------------------------------------------------------------------------
# parse_diagnostic_confidence — robust 0–100 coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (85, 85.0),
        (0, 0.0),
        (100, 100.0),
        (150, 100.0),          # clamped up
        (-10, 0.0),            # clamped down
        (0.85, 85.0),          # probability form scaled to 0–100
        ("85", 85.0),
        ("85%", 85.0),
        (" 72 % ", 72.0),
    ],
)
def test_parse_confidence_valid(raw, expected):
    assert parse_diagnostic_confidence(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "high", "n/a", True, False, [], {}])
def test_parse_confidence_missing_or_unparseable_is_none(raw):
    assert parse_diagnostic_confidence(raw) is None


# ---------------------------------------------------------------------------
# is_terminal_turn — confidence-based stopping
# ---------------------------------------------------------------------------


def test_high_confidence_stops_even_on_first_turn():
    # Above threshold at turn 0 — stop asking, present the assessment.
    assert is_terminal_turn(turn_count=0, analysis={"diagnostic_confidence": 85}) is True


def test_threshold_boundary_is_inclusive():
    threshold = settings.DIAGNOSTIC_CONFIDENCE_THRESHOLD
    assert is_terminal_turn(turn_count=0, analysis={"diagnostic_confidence": threshold}) is True
    assert is_terminal_turn(turn_count=0, analysis={"diagnostic_confidence": threshold - 1}) is False


def test_low_confidence_keeps_interview_open():
    assert is_terminal_turn(turn_count=2, analysis={"diagnostic_confidence": 40}) is False


def test_turn_cap_is_a_backstop_for_low_confidence():
    # Confidence never crossed the bar, but the cap forces closure.
    analysis = {"diagnostic_confidence": 30}
    assert is_terminal_turn(turn_count=settings.MAX_DIAGNOSTIC_TURNS, analysis=analysis) is True


def test_emergency_redirect_always_terminal():
    analysis = {"final_action": "emergency_redirect", "diagnostic_confidence": 10}
    assert is_terminal_turn(turn_count=0, analysis=analysis) is True


def test_missing_confidence_falls_back_to_turn_cap():
    # No confidence signal (e.g. trivial-skip turn) → only the cap can close it.
    assert is_terminal_turn(turn_count=0, analysis={}) is False
    assert is_terminal_turn(turn_count=99, analysis={}) is True
    assert is_terminal_turn(turn_count=0, analysis=None) is False


def test_explicit_threshold_override():
    analysis = {"diagnostic_confidence": 70}
    assert is_terminal_turn(turn_count=0, analysis=analysis, confidence_threshold=65) is True
    assert is_terminal_turn(turn_count=0, analysis=analysis, confidence_threshold=90) is False


def test_probability_form_confidence_crosses_threshold():
    # Analyzer emitted 0.9 instead of 90 — normalisation still stops the interview.
    assert is_terminal_turn(turn_count=0, analysis={"diagnostic_confidence": 0.9}) is True
