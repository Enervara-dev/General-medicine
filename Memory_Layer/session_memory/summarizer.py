"""
summarizer.py
─────────────
Heuristic conversation summarizer for the Enervera session memory layer.

Responsibilities:
  - Compress older conversation turns into a concise rolling text summary.
  - Preserve clinically and conversationally important signals:
      symptoms, conditions, drugs, demographics, active goals, risk flags.
  - Move turns older than a configurable threshold into the summary blob.
  - Keep the most recent N turns completely untouched in recent_turns.
  - NEVER re-summarize the existing summary — only raw Message turns are
    processed, preventing runaway compression artifacts.

Design notes:
  - Fully deterministic — NO LLM calls in this version.
  - Template-driven: produces clean, compact prose suitable for prompt injection.
  - Designed so the template can be swapped for an LLM call later without
    changing the public API (should_summarize / summarize_session).

Public API:
    should_summarize(session)             → bool
    build_turn_summary(turns)             → str
    summarize_session(session, keep_last) → SessionMemory  (new copy)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from .models import Message, Role, SessionMemory, StructuredState

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# When recent_turns exceed this, trigger summarization
SUMMARIZE_THRESHOLD: int = 8

# How many of the most-recent turns to always keep verbatim
KEEP_LAST_N: int = 4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_timestamp(dt: datetime) -> str:
    """Return a compact UTC timestamp string for summary headers."""
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _role_label(role: str) -> str:
    return {"user": "Patient", "assistant": "Assistant", "system": "System"}.get(
        role, role.capitalize()
    )


def _state_context_lines(state: StructuredState) -> list[str]:
    """
    Convert the StructuredState into compact, prose-style context lines
    for embedding into the summary block.
    Only includes fields that have actual values.
    """
    lines: list[str] = []

    if state.demographics:
        parts = []
        if "age" in state.demographics:
            parts.append(f"age {state.demographics['age']}")
        if "sex" in state.demographics:
            parts.append(state.demographics["sex"])
        if parts:
            lines.append(f"Patient: {', '.join(parts)}.")

    if state.symptoms:
        lines.append(f"Reported symptoms: {', '.join(state.symptoms)}.")

    if state.conditions:
        lines.append(f"Known conditions: {', '.join(state.conditions)}.")

    if state.chronic_conditions:
        lines.append(f"Chronic conditions: {', '.join(state.chronic_conditions)}.")

    if state.drugs:
        lines.append(f"Medications mentioned: {', '.join(state.drugs)}.")

    if state.allergies:
        lines.append(f"Allergies mentioned: {', '.join(state.allergies)}.")

    if state.severity:
        lines.append(f"Severity descriptors: {', '.join(state.severity)}.")

    if state.duration:
        lines.append(f"Duration: {', '.join(state.duration)}.")

    if state.previous_concerns:
        lines.append(f"Previous concerns: {', '.join(state.previous_concerns)}.")

    if state.active_task:
        lines.append(f"Active task: {state.active_task}.")

    if state.current_goal:
        lines.append(f"Patient goal: {state.current_goal}.")

    if state.care_setting:
        lines.append(f"Care setting: {state.care_setting}.")

    if state.risk_level and str(state.risk_level) not in ("none", "RiskLevel.NONE"):
        lines.append(f"Risk level: {state.risk_level}.")

    prefs = [k for k, v in state.preferences.items() if v]
    if prefs:
        lines.append(f"Preferences: {', '.join(prefs)}.")

    return lines


def _turns_to_dialogue(turns: Sequence[Message]) -> str:
    """
    Convert a list of Message objects into a compact readable dialogue block.
    User messages are labelled 'Patient:', assistant messages 'Assistant:'.
    Long assistant responses are truncated to 300 chars to keep summary lean.
    """
    lines: list[str] = []
    for msg in turns:
        label   = _role_label(msg.role)
        content = msg.content.strip()

        # Trim very long assistant replies — we preserve intent, not verbatim text
        if msg.role == Role.ASSISTANT and len(content) > 300:
            content = content[:297] + "…"

        lines.append(f"{label}: {content}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core summarization logic
# ---------------------------------------------------------------------------

def build_turn_summary(
    turns: Sequence[Message],
    state: StructuredState | None = None,
    existing_summary: str = "",
) -> str:
    """
    Build a concise rolling text summary from a batch of older Message turns.

    The summary is structured in three sections:
      1. [Medical Context] — extracted state fields (if state provided)
      2. [Prior Conversation] — compact dialogue from the given turns
      3. Prepended to any existing summary (NEVER re-summarized)

    Args:
        turns:            The older turns to compress. Caller ensures these
                          are the turns being moved OUT of recent_turns.
        state:            Current StructuredState for clinical context header.
        existing_summary: Any prior summary text — prepended as-is, never
                          re-processed.

    Returns:
        A single string summary block.
    """
    if not turns:
        return existing_summary

    sections: list[str] = []

    # ── Section 1: Medical context header (from StructuredState) ───────────
    if state:
        ctx_lines = _state_context_lines(state)
        if ctx_lines:
            sections.append("[Medical Context]\n" + "\n".join(ctx_lines))

    # ── Section 2: Compressed dialogue from the older turns ────────────────
    dialogue = _turns_to_dialogue(turns)
    ts_start = _format_timestamp(turns[0].timestamp)
    ts_end   = _format_timestamp(turns[-1].timestamp)
    sections.append(
        f"[Prior Conversation — {ts_start} to {ts_end}]\n{dialogue}"
    )

    new_block = "\n\n".join(sections)

    # ── Append to prior summary WITHOUT re-processing it ───────────────────
    if existing_summary.strip():
        return existing_summary.strip() + "\n\n---\n\n" + new_block
    return new_block


def should_summarize(
    session: SessionMemory,
    threshold: int = SUMMARIZE_THRESHOLD,
) -> bool:
    """
    Return True when the session has enough turns to warrant compression.

    Args:
        session:   The current SessionMemory.
        threshold: Minimum number of recent_turns before summarization fires.

    Returns:
        bool
    """
    return len(session.recent_turns) >= threshold


def summarize_session(
    session: SessionMemory,
    keep_last: int = KEEP_LAST_N,
) -> SessionMemory:
    """
    Compress older turns into the summary field and return a new SessionMemory.

    Algorithm:
      1. Split recent_turns into [older_turns | kept_turns].
         older_turns  = recent_turns[:-keep_last]
         kept_turns   = recent_turns[-keep_last:]
      2. Build a new summary block from older_turns + existing state context.
      3. Prepend the new block to any existing session.summary (never re-processed).
      4. Return a new SessionMemory with:
           - recent_turns  = kept_turns  (the newest N turns, verbatim)
           - summary       = combined summary block
           - state         = unchanged

    The original session is NOT mutated.

    Args:
        session:   The SessionMemory to compress.
        keep_last: Number of most-recent turns to keep verbatim.

    Returns:
        A new SessionMemory with compressed older turns.
    """
    turns = session.recent_turns

    # Nothing to compress if turns <= keep_last
    if len(turns) <= keep_last:
        return session.model_copy(deep=True)

    older_turns = list(turns[:-keep_last])
    kept_turns  = list(turns[-keep_last:])

    new_summary = build_turn_summary(
        turns=older_turns,
        state=session.state,
        existing_summary=session.summary,
    )

    updated = session.model_copy(deep=True)
    updated.recent_turns = kept_turns
    updated.summary      = new_summary
    updated.updated_at   = datetime.now(tz=timezone.utc)

    return updated


def maybe_summarize(
    session: SessionMemory,
    threshold: int = SUMMARIZE_THRESHOLD,
    keep_last:  int = KEEP_LAST_N,
) -> SessionMemory:
    """
    Convenience wrapper: summarize only if threshold is reached.

    Call this once per turn, after adding the new message and extracting state:

        session = maybe_summarize(session)
        await mgr.save_session(session)

    Args:
        session:   Current SessionMemory.
        threshold: Turn count that triggers compression.
        keep_last: Verbatim turns to keep after compression.

    Returns:
        Either the original session (unchanged) or a newly compressed copy.
    """
    if should_summarize(session, threshold):
        return summarize_session(session, keep_last)
    return session
