"""Clarification — at-most-one question per turn, safety-critical only."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ClarificationReason(str, Enum):
    MISSING_DURATION = "missing_duration"
    MISSING_LOCATION = "missing_location"
    MISSING_SEVERITY = "missing_severity"
    AMBIGUOUS_MEDICATION = "ambiguous_medication"
    CONTRADICTION = "contradiction"
    AMBIGUOUS_CHRONOLOGY = "ambiguous_chronology"


class ClarificationQuestion(BaseModel):
    reason: ClarificationReason
    question: str = Field(min_length=4, max_length=240)
    safety_critical: bool = False


class ClarificationRequest(BaseModel):
    """Input: an utterance (and optionally a candidate episode) to evaluate."""
    user_id: str
    utterance: str
    candidate: dict | None = None  # serialized EpisodeCandidate, optional


class ClarificationResponse(BaseModel):
    needs_clarification: bool
    # Always 0 or 1 items by contract; never more.
    questions: list[ClarificationQuestion] = Field(default_factory=list, max_length=1)
