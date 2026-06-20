"""Contradiction DTOs."""

from __future__ import annotations

import uuid
from enum import Enum

from pydantic import BaseModel, Field


class ContradictionSeverity(str, Enum):
    INFO = "info"          # benign update (e.g. "headache resolved")
    WARNING = "warning"    # potentially clinically relevant
    CRITICAL = "critical"  # safety-critical (denied vs reported allergy, etc.)


class Contradiction(BaseModel):
    prior_episode_id: uuid.UUID
    prior_summary: str
    current_claim: str
    reason: str = Field(min_length=4, max_length=400)
    severity: ContradictionSeverity = ContradictionSeverity.WARNING


class ContradictionReport(BaseModel):
    user_id: str
    has_contradictions: bool
    contradictions: list[Contradiction] = Field(default_factory=list)
    confidence_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    triggers_clarification: bool = False
