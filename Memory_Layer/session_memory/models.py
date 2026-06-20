"""
models.py
─────────
Pydantic v2 data models for the session memory layer.

Defines the canonical data structures that flow through the memory system:
  - Message       : a single user/assistant exchange turn
  - StructuredState : extracted medical context from the conversation
  - SessionMemory   : the complete in-memory snapshot of one session
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Role(str, Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


class RiskLevel(str, Enum):
    NONE     = "none"
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Message — one conversational turn
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """A single turn in the conversation (user query or assistant reply)."""

    id:        str      = Field(default_factory=lambda: uuid4().hex)
    role:      Role
    content:   str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    # Enrichment attached by the gatekeeper / query analyzer
    intent:     str | None = None
    risk_level: RiskLevel  = RiskLevel.NONE
    query_type: str | None = None   # e.g. "symptom_query", "guideline", …

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# StructuredState — extracted medical context for the whole session
# ---------------------------------------------------------------------------

class StructuredState(BaseModel):
    """
    Running medical context distilled from the conversation so far.
    Updated incrementally as the session progresses.

    All fields are machine-readable primitives — no prose descriptions.
    """

    # ── Core medical entities ────────────────────────────────────────────
    symptoms:           list[str]       = Field(default_factory=list)
    conditions:         list[str]       = Field(default_factory=list)
    chronic_conditions: list[str]       = Field(default_factory=list)
    drugs:              list[str]       = Field(default_factory=list)
    allergies:          list[str]       = Field(default_factory=list)
    severity:           list[str]       = Field(default_factory=list)
    duration:           list[str]       = Field(default_factory=list)
    demographics:       dict[str, Any]  = Field(default_factory=dict)  # {age, sex, weight, …}

    # ── Conversation intent & goal ───────────────────────────────────────
    active_task:        str | None      = None   # e.g. "symptom_query", "guideline"
    current_goal:       str | None      = None   # e.g. "understand treatment options"
    last_intent:        str | None      = None   # most recent query intent

    # ── Clinical context ─────────────────────────────────────────────────
    care_setting:       str | None      = None   # "home", "hospital", "urgent_care"
    retrieval_strategy: str | None      = None   # "symptom_focused", "drug_interaction", …
    risk_level:         RiskLevel       = RiskLevel.NONE

    # ── Accumulated named entities across turns ──────────────────────────
    discussed_entities: list[str]       = Field(default_factory=list)
    previous_concerns:  list[str]       = Field(default_factory=list)
    follow_up_references: list[str]     = Field(default_factory=list)

    # ── Patient-expressed preferences ───────────────────────────────────
    preferences:        dict[str, Any]  = Field(default_factory=dict)
    # e.g. {"avoid_surgery": True, "prefers_natural": True, "language": "simple"}

    # ── Extension point for future fields ───────────────────────────────
    extra:              dict[str, Any]  = Field(default_factory=dict)

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# SessionMemory — the full session snapshot stored in Redis
# ---------------------------------------------------------------------------

MAX_RECENT_TURNS: int = 10   # hard cap on turns kept in memory


class SessionMemory(BaseModel):
    """
    Complete memory snapshot for one user session.

    Stored under Redis key: session:{session_id}
    """

    session_id:   str           = Field(default_factory=lambda: uuid4().hex)
    created_at:   datetime      = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    updated_at:   datetime      = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    # The rolling conversation window (capped at MAX_RECENT_TURNS)
    recent_turns: list[Message]     = Field(default_factory=list)

    # A rolling prose summary of older turns (populated by the summarizer)
    summary:      str               = ""

    # Structured medical state extracted / updated each turn
    state:        StructuredState   = Field(default_factory=StructuredState)

    model_config = {"use_enum_values": True}

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def add_turn(self, message: Message) -> None:
        """Append a message and enforce the recent-turn cap."""
        self.recent_turns.append(message)
        self._trim_turns()
        self.updated_at = datetime.now(tz=timezone.utc)

    def _trim_turns(self) -> None:
        """Keep only the most recent MAX_RECENT_TURNS turns."""
        if len(self.recent_turns) > MAX_RECENT_TURNS:
            self.recent_turns = self.recent_turns[-MAX_RECENT_TURNS:]

    @property
    def turn_count(self) -> int:
        return len(self.recent_turns)

    @property
    def last_user_message(self) -> Message | None:
        for turn in reversed(self.recent_turns):
            if turn.role == Role.USER:
                return turn
        return None

    @property
    def last_assistant_message(self) -> Message | None:
        for turn in reversed(self.recent_turns):
            if turn.role == Role.ASSISTANT:
                return turn
        return None
