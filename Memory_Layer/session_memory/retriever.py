"""
retriever.py
────────────
In-session memory retrieval helpers for the Enervera Medical RAG system.

Responsibilities:
  - Provide clean, structured views of a SessionMemory without touching Redis
    or any vector database.
  - Assemble a "working memory" payload that downstream components (prompt
    builder, RAG pipeline) can consume directly.
  - Keep token usage minimal via configurable truncation.

Design notes:
  - Pure functions — no async, no I/O, no LLM calls.
  - All functions operate on an in-memory SessionMemory object already loaded
    from Redis by SessionManager.
  - Working memory format is a typed dataclass so callers get IDE support and
    the shape can be extended without breaking existing consumers.
  - Semantic / vector retrieval is deliberately excluded here — this layer
    handles only session-scoped, recency-based retrieval.

Public API:
    get_recent_turns(session, n)   → list[Message]
    get_summary(session)           → str
    get_structured_state(session)  → StructuredState
    get_working_memory(session, …) → WorkingMemory
    format_working_memory(wm)      → str          (plain-text for prompt injection)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Message, Role, SessionMemory, StructuredState

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_RECENT_N:           int = 6     # turns to include in working memory
MAX_SUMMARY_CHARS:          int = 1200  # hard cap on summary text in working memory
MAX_TURN_CONTENT_CHARS:     int = 400   # per-turn content truncation


# ---------------------------------------------------------------------------
# WorkingMemory — typed payload handed to downstream components
# ---------------------------------------------------------------------------

@dataclass
class WorkingMemory:
    """
    Assembled session context ready for use in prompt construction or
    RAG pipeline augmentation.

    Fields are intentionally simple / serialisable primitives so this
    dataclass can be passed across module boundaries without coupling.
    """
    session_id:         str
    summary:            str                 # compressed older context (may be "")
    recent_turns:       list[Message]       # verbatim recent exchange window
    state:              StructuredState     # latest extracted medical state

    # Convenience aggregates (pre-computed for callers)
    active_task:        str | None          # mirrors state.active_task
    risk_level:         str                 # mirrors state.risk_level (str form)
    discussed_entities: list[str]           # mirrors state.discussed_entities
    preferences:        dict                # mirrors state.preferences

    # Metadata
    turn_count:         int = 0             # total recent turns in window
    has_summary:        bool = False        # whether a prior summary exists


# ---------------------------------------------------------------------------
# get_recent_turns
# ---------------------------------------------------------------------------

def get_recent_turns(
    session: SessionMemory,
    n: int = DEFAULT_RECENT_N,
    roles: set[str] | None = None,
) -> list[Message]:
    """
    Return the N most recent turns from the session, optionally filtered
    by role.

    Args:
        session: The loaded SessionMemory.
        n:       Maximum number of turns to return.
        roles:   Optional set of role strings to include, e.g. {"user"}.
                 If None, all roles are returned.

    Returns:
        List of Message objects, oldest-first, at most n items.
    """
    turns = session.recent_turns

    if roles:
        turns = [t for t in turns if t.role in roles]

    return list(turns[-n:])


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------

def get_summary(
    session: SessionMemory,
    max_chars: int = MAX_SUMMARY_CHARS,
) -> str:
    """
    Return the session's rolling summary, optionally truncated to avoid
    excessive token use in downstream prompts.

    Truncation always cuts at a paragraph boundary where possible to
    keep the text coherent.

    Args:
        session:   The loaded SessionMemory.
        max_chars: Hard character cap; 0 means no cap.

    Returns:
        Summary string (may be empty if no summarization has occurred yet).
    """
    text = session.summary.strip()

    if not text or max_chars == 0:
        return text

    if len(text) <= max_chars:
        return text

    # Try to cut at a paragraph break within the limit
    truncated = text[:max_chars]
    last_break = truncated.rfind("\n\n")
    if last_break > max_chars * 0.5:
        return truncated[:last_break].strip() + "\n\n[…summary truncated…]"

    return truncated.strip() + " […summary truncated…]"


# ---------------------------------------------------------------------------
# get_structured_state
# ---------------------------------------------------------------------------

def get_structured_state(session: SessionMemory) -> StructuredState:
    """
    Return the session's current StructuredState.

    Thin wrapper provided for API consistency — callers should use this
    rather than accessing session.state directly so future middleware
    (caching, validation) can be added without changing call sites.

    Args:
        session: The loaded SessionMemory.

    Returns:
        The StructuredState object.
    """
    return session.state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int) -> str:
    """Truncate text to `limit` chars, appending '…' if cut."""
    if len(text) <= limit:
        return text
    return text[:limit - 1].rstrip() + "…"


def _role_label(role: str) -> str:
    return {"user": "Patient", "assistant": "Assistant", "system": "System"}.get(
        role, role.capitalize()
    )


def _format_turn(msg: Message, max_chars: int = MAX_TURN_CONTENT_CHARS) -> str:
    """Format a single message as 'Role: content' with optional truncation."""
    label   = _role_label(msg.role)
    content = _truncate(msg.content.strip(), max_chars)
    return f"{label}: {content}"


def _state_to_dict(state: StructuredState) -> dict:
    """
    Convert StructuredState to a minimal, non-empty dict for display /
    downstream use. Skips empty collections and None values.
    """
    out: dict = {}
    for key, val in state.model_dump(mode="python").items():
        if val is None:
            continue
        if isinstance(val, list) and not val:
            continue
        if isinstance(val, dict) and not val:
            continue
        if str(val) in ("none", "RiskLevel.NONE", ""):
            continue
        out[key] = val
    return out


# ---------------------------------------------------------------------------
# get_working_memory  — primary public function
# ---------------------------------------------------------------------------

def get_working_memory(
    session:      SessionMemory,
    recent_n:     int = DEFAULT_RECENT_N,
    max_summary:  int = MAX_SUMMARY_CHARS,
) -> WorkingMemory:
    """
    Assemble a complete WorkingMemory payload from the current session.

    This is the primary entry point for any downstream component that needs
    session context (prompt builder, RAG pipeline augmentor, etc.).

    Args:
        session:     The loaded SessionMemory.
        recent_n:    How many recent turns to include verbatim.
        max_summary: Character cap on the summary section.

    Returns:
        A populated WorkingMemory dataclass.
    """
    recent  = get_recent_turns(session, n=recent_n)
    summary = get_summary(session, max_chars=max_summary)
    state   = get_structured_state(session)

    # Normalise risk_level to plain string for easy comparison downstream
    risk_str = (
        state.risk_level.value
        if hasattr(state.risk_level, "value")
        else str(state.risk_level)
    )

    return WorkingMemory(
        session_id         = session.session_id,
        summary            = summary,
        recent_turns       = recent,
        state              = state,
        active_task        = state.active_task,
        risk_level         = risk_str,
        discussed_entities = list(state.discussed_entities),
        preferences        = dict(state.preferences),
        turn_count         = len(recent),
        has_summary        = bool(summary),
    )


# ---------------------------------------------------------------------------
# format_working_memory  — plain-text renderer for prompt injection
# ---------------------------------------------------------------------------

def format_working_memory(
    wm:              WorkingMemory,
    include_summary: bool = True,
    include_state:   bool = True,
    include_turns:   bool = True,
) -> str:
    """
    Render a WorkingMemory object as a compact, human-readable plain-text
    block suitable for injection into an LLM prompt.

    Sections are included only when they have content, keeping the output
    token-lean.

    Args:
        wm:              The WorkingMemory to render.
        include_summary: Whether to include the prior summary section.
        include_state:   Whether to include the structured state section.
        include_turns:   Whether to include the recent turns dialogue.

    Returns:
        A multi-section plain-text string.

    Example output:
        === PRIOR CONTEXT ===
        Patient: age 42, female.
        Reported symptoms: fever, chills, sore_throat.
        ...

        === RECENT CONVERSATION ===
        Patient: I have a fever and sore throat for 3 days…
        Assistant: Based on your symptoms, this could be…
    """
    sections: list[str] = []

    # ── Section 1: Prior Summary ─────────────────────────────────────────
    if include_summary and wm.has_summary and wm.summary:
        sections.append("=== PRIOR CONTEXT ===\n" + wm.summary)

    # ── Section 2: Structured State ──────────────────────────────────────
    if include_state:
        state_dict = _state_to_dict(wm.state)
        if state_dict:
            state_lines: list[str] = []

            # Demographics
            demo = state_dict.get("demographics", {})
            if demo:
                parts = []
                if "age" in demo:
                    parts.append(f"age {demo['age']}")
                if "sex" in demo:
                    parts.append(demo["sex"])
                if parts:
                    state_lines.append(f"Patient: {', '.join(parts)}.")

            # Clinical
            for key, label in [
                ("symptoms",   "Symptoms"),
                ("conditions", "Conditions"),
                ("chronic_conditions", "Chronic conditions"),
                ("drugs",      "Medications"),
                ("allergies",  "Allergies"),
                ("severity",   "Severity"),
                ("duration",   "Duration"),
                ("previous_concerns", "Previous concerns"),
                ("follow_up_references", "Follow-up references"),
            ]:
                val = state_dict.get(key)
                if val:
                    state_lines.append(f"{label}: {', '.join(val)}.")

            # Intent / goal
            if state_dict.get("active_task"):
                state_lines.append(f"Active task: {state_dict['active_task']}.")
            if state_dict.get("current_goal"):
                state_lines.append(f"Goal: {state_dict['current_goal']}.")
            if state_dict.get("care_setting"):
                state_lines.append(f"Care setting: {state_dict['care_setting']}.")
            if wm.risk_level not in ("none", "RiskLevel.NONE", ""):
                state_lines.append(f"Risk level: {wm.risk_level}.")

            # Preferences
            prefs = [k for k, v in (state_dict.get("preferences") or {}).items() if v]
            if prefs:
                state_lines.append(f"Preferences: {', '.join(prefs)}.")

            if state_lines:
                sections.append("=== SESSION STATE ===\n" + "\n".join(state_lines))

    # ── Section 3: Recent Conversation ──────────────────────────────────
    if include_turns and wm.recent_turns:
        dialogue = "\n".join(_format_turn(t) for t in wm.recent_turns)
        sections.append("=== RECENT CONVERSATION ===\n" + dialogue)

    return "\n\n".join(sections)
