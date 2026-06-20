"""
Decay scoring — pure math, no I/O.

Chronic conditions and allergies never decay. Everything else uses an
exponential decay with a configurable half-life (default 14 days).

The decay score multiplies into the composite rank score, so a value of 1.0
means "treat as if it happened today" and 0.0 means "do not surface".
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from episodic.config import EpisodicConfig
from episodic.schemas.episode import (
    ClinicalPriority,
    Episode,
    EpisodeCategory,
)


_CHRONIC_DURATION_HINTS = (
    "ongoing", "lifelong", "long-term", "chronic", "permanent", "indefinite",
)


def is_chronic(ep: Episode) -> bool:
    """True if the episode represents a long-running clinical state."""
    if ep.category == EpisodeCategory.ALLERGY:
        return True
    # Diagnosed conditions persist by default — clinical facts like "T2DM" or
    # "HTN" don't decay unless explicitly marked resolved (which today we'd
    # represent as a new episode with category=symptom or a status change).
    if ep.category == EpisodeCategory.CONDITION:
        return True
    # Long-running medications (lisinopril daily, metformin BID, etc.) carry
    # a "ongoing"/"lifelong"/"long-term" duration hint. Discontinued meds use
    # phrasing like "drug stopped" / "discontinued" which won't match.
    if ep.category == EpisodeCategory.MEDICATION:
        dur = (ep.temporal_data.duration or "").lower()
        if any(hint in dur for hint in _CHRONIC_DURATION_HINTS):
            return True
    return False


def compute_decay_score(ep: Episode, *, now: datetime | None = None) -> float:
    """
    Exponential decay with half-life from config. Returns [0, 1].

    Chronic conditions and critical-priority allergies → 1.0 forever.
    Everything else: 0.5^(age_days / half_life_days), clamped at [0.05, 1.0].
    """
    now = now or datetime.now(tz=timezone.utc)
    if is_chronic(ep):
        return 1.0
    if ep.clinical_priority == ClinicalPriority.CRITICAL:
        # Critical events decay slower — double the half-life.
        half_life = EpisodicConfig.DECAY_HALF_LIFE_DAYS * 2
    else:
        half_life = EpisodicConfig.DECAY_HALF_LIFE_DAYS

    age_days = max((now - ep.timestamp).total_seconds() / 86400.0, 0.0)
    if half_life <= 0:
        return 1.0
    raw = math.pow(0.5, age_days / half_life)
    # Floor so a relevant-but-old episode can still surface if everything else is empty.
    return max(0.05, min(1.0, raw))
