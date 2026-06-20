"""
context_builder.py
──────────────────
Final context assembly layer for the Enervera Medical RAG session memory system.

Responsibilities:
  - Assemble all memory layers into a clean, ordered prompt payload.
  - Apply token-aware trimming so the assembled context never exceeds budget.
  - Produce a structured ContextPayload that the RAG pipeline can consume
    directly, with placeholder slots for RAG-retrieved documents and tool
    outputs that will be injected later.

Assembly order (matches LLM attention priority, most persistent → most recent):
  1. SYSTEM PROMPT       ← role, tone, safety rules
  2. STRUCTURED STATE    ← extracted medical context from session
  3. ROLLING SUMMARY     ← compressed older turns
  4. RECENT TURNS        ← verbatim last N exchanges
  5. RAG CONTEXT         ← placeholder (injected by RAG pipeline later)
  6. CURRENT QUERY       ← the live user question

Design notes:
  - Pure functions — no async, no I/O, no LLM calls.
  - Token estimation is character-based (÷ 4) — fast and good-enough for
    budget gating without requiring a tokenizer dependency.
  - ContextPayload is a typed dataclass with explicit RAG/tool slots so
    future integration requires zero API changes to this module.
  - Every section is independently toggleable via build_final_prompt() flags.

Public API:
    build_memory_context(wm)                           → str
    build_conversation_context(wm, max_tokens)         → str
    assemble_context_payload(wm, query, ...)           → ContextPayload
    build_final_prompt(payload)                        → FinalPrompt

System-prompt construction has moved to
`app.services.orchestration.prompt_layers.compose_system_prompt` — both the
FastAPI orchestrator and the legacy CLI now call it. The
`ContextPayload.system_context` field is kept for backward compat but is
always empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import Message, Role, StructuredState
from .retriever import WorkingMemory

# ---------------------------------------------------------------------------
# Token budget configuration
# ---------------------------------------------------------------------------

# Approximate character-to-token ratio (conservative, model-agnostic)
_CHARS_PER_TOKEN: int = 4

# Default total token budget for the entire assembled context
DEFAULT_TOKEN_BUDGET: int = 3_500

# Per-section soft caps (in tokens)
# System prompt budget bumped from 400 → 1150 to accommodate the layered
# clinician prompt (ranked differential, mechanism reasoning, escalation
# policy, RAG grounding, questioning strategy).
SYSTEM_PROMPT_MAX_TOKENS:  int = 1150
STATE_SECTION_MAX_TOKENS:  int = 250
SUMMARY_SECTION_MAX_TOKENS:int = 400
TURNS_SECTION_MAX_TOKENS:  int = 600
RAG_SECTION_MAX_TOKENS:    int = 800
QUERY_MAX_TOKENS:          int = 150

# Per-turn content character cap (before token conversion)
MAX_TURN_CHARS: int = 350


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Fast character-based token estimation (÷ 4)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _trim_to_tokens(text: str, max_tokens: int, suffix: str = " […trimmed]") -> str:
    """Hard-trim text to `max_tokens`, appending suffix if cut."""
    limit = max_tokens * _CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    cut = text[: limit - len(suffix)].rstrip()
    # Prefer cutting at a sentence or paragraph boundary
    for sep in ("\n\n", "\n", ". ", " "):
        idx = cut.rfind(sep)
        if idx > limit * 0.6:
            return cut[:idx].rstrip() + suffix
    return cut + suffix


def _role_label(role: str) -> str:
    return {"user": "Patient", "assistant": "Assistant", "system": "System"}.get(
        role, role.capitalize()
    )


def _format_turn(msg: Message) -> str:
    label   = _role_label(msg.role)
    content = msg.content.strip()
    if len(content) > MAX_TURN_CHARS:
        content = content[:MAX_TURN_CHARS - 1].rstrip() + "…"
    return f"{label}: {content}"


def _state_lines(state: StructuredState) -> list[str]:
    """Render StructuredState as compact, non-empty bullet lines."""
    lines: list[str] = []

    demo = state.demographics or {}
    name = demo.get("name")
    parts = []
    if "age" in demo:
        parts.append(f"age {demo['age']}")
    if "sex" in demo:
        parts.append(demo["sex"])
    if name and parts:
        lines.append(f"Patient name: {name} ({', '.join(parts)}).")
    elif name:
        lines.append(f"Patient name: {name}.")
    elif parts:
        lines.append(f"Patient: {', '.join(parts)}.")

    pairs = [
        ("symptoms",   "Symptoms"),
        ("conditions", "Conditions"),
        ("chronic_conditions", "Chronic conditions"),
        ("drugs",      "Medications"),
        ("allergies",  "Allergies"),
        ("severity",   "Severity"),
        ("duration",   "Duration"),
        ("previous_concerns", "Previous concerns"),
        ("follow_up_references", "Follow-up references"),
    ]
    for attr, label in pairs:
        val = getattr(state, attr, [])
        if val:
            lines.append(f"{label}: {', '.join(val)}.")

    if state.active_task:
        lines.append(f"Active task: {state.active_task}.")
    if state.current_goal:
        lines.append(f"Goal: {state.current_goal}.")
    if state.care_setting:
        lines.append(f"Care setting: {state.care_setting}.")

    risk = str(getattr(state, "risk_level", "") or "")
    if risk and risk not in ("none", "RiskLevel.NONE", ""):
        lines.append(f"Risk level: {risk}.")

    prefs = [k for k, v in (state.preferences or {}).items() if v]
    if prefs:
        lines.append(f"Preferences: {', '.join(prefs)}.")

    return lines


# ---------------------------------------------------------------------------
# ContextPayload — structured intermediate payload
# ---------------------------------------------------------------------------

@dataclass
class ContextPayload:
    """
    Structured intermediate payload produced by assemble_context_payload().

    Each section is a separate string so downstream components can
    independently inject, replace, or reorder sections before calling
    build_final_prompt().

    RAG integration points (injected externally, after memory assembly):
      rag_context     ← vector + graph retrieved documents
      tool_outputs    ← tool call results (future)
      citations       ← source attribution strings (future)
      reranker_scores ← per-chunk relevance metadata (future)
    """

    # Memory sections (built by this module)
    system_context:       str
    memory_context:       str          # structured state block
    summary_context:      str          # rolling summary block
    conversation_context: str          # recent turns block

    # Live query
    user_query:           str

    # RAG slots (injected externally — empty by default)
    rag_context:          str                 = ""
    tool_outputs:         list[str]           = field(default_factory=list)
    citations:            list[str]           = field(default_factory=list)
    reranker_scores:      dict[str, float]    = field(default_factory=dict)

    # Metadata
    session_id:           str                 = ""
    active_task:          str | None          = None
    risk_level:           str                 = "none"
    estimated_tokens:     int                 = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for logging / handoff to the RAG pipeline."""
        return {
            "memory_context":  self._memory_block(),
            "rag_context":     self.rag_context,
            "user_query":      self.user_query,
            "session_id":      self.session_id,
            "active_task":     self.active_task,
            "risk_level":      self.risk_level,
            "estimated_tokens": self.estimated_tokens,
            "has_rag":         bool(self.rag_context),
            "has_tools":       bool(self.tool_outputs),
            "citations":       self.citations,
        }

    def _memory_block(self) -> str:
        """Combined memory sections as a single string."""
        parts = []
        if self.memory_context:
            parts.append(self.memory_context)
        if self.summary_context:
            parts.append(self.summary_context)
        if self.conversation_context:
            parts.append(self.conversation_context)
        return "\n\n".join(parts)


@dataclass
class FinalPrompt:
    """
    Fully assembled prompt ready to be sent to the LLM.

    Provides both a messages list (for chat APIs) and a single-string
    form (for completion APIs), so callers choose the format they need.
    """
    messages:      list[dict[str, str]]   # OpenAI-style messages list
    system_text:   str                    # system message content
    user_text:     str                    # user message content (full assembled context)
    total_tokens:  int                    # estimated token count
    payload:       ContextPayload | None  = None  # source payload (for debugging)


# ---------------------------------------------------------------------------
# System-prompt construction has moved.
#
# The earlier `build_system_context()` / `_RISK_TONE` lived here but were not
# read by either the FastAPI orchestrator (which composes its own prompt) or
# the legacy CLI pipeline (which only consumes `memory_context` and
# `conversation_context` from the payload). The actual system prompt is now
# layered in `app.services.orchestration.prompt_layers.compose_system_prompt`.
#
# `ContextPayload.system_context` remains as an empty-string placeholder so
# any external caller that destructures the field doesn't crash.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# build_memory_context
# ---------------------------------------------------------------------------

def build_memory_context(wm: WorkingMemory) -> str:
    """
    Build the structured state section from a WorkingMemory object.

    Includes only non-empty fields — produces nothing if the session has
    no extracted state yet (first turn).

    Args:
        wm: The assembled WorkingMemory from the retriever layer.

    Returns:
        A formatted state block string, or "" if state is empty.
    """
    lines = _state_lines(wm.state)
    if not lines:
        return ""

    body = "\n".join(lines)
    block = f"[SESSION STATE]\n{body}"
    return _trim_to_tokens(block, STATE_SECTION_MAX_TOKENS)


# ---------------------------------------------------------------------------
# build_conversation_context
# ---------------------------------------------------------------------------

def build_conversation_context(
    wm:         WorkingMemory,
    max_tokens: int = TURNS_SECTION_MAX_TOKENS,
) -> str:
    """
    Build the recent conversation section.

    Constructs the dialogue block from wm.recent_turns oldest-first,
    then trims to the token budget if necessary (cutting oldest turns first
    to keep the most recent context intact).

    Args:
        wm:         WorkingMemory with recent_turns already loaded.
        max_tokens: Soft token cap for the entire turns section.

    Returns:
        Formatted dialogue block string, or "" if no turns exist.
    """
    if not wm.recent_turns:
        return ""

    # Build individual turn lines oldest → newest
    turn_lines = [_format_turn(t) for t in wm.recent_turns]

    # Greedily include turns from newest → oldest until budget is consumed
    selected: list[str] = []
    budget = max_tokens * _CHARS_PER_TOKEN

    for line in reversed(turn_lines):
        if len(line) + 1 > budget:  # +1 for newline
            break
        selected.insert(0, line)
        budget -= len(line) + 1

    if not selected:
        # Even the most recent turn exceeds budget — include it truncated
        selected = [_trim_to_tokens(turn_lines[-1], max_tokens)]

    dialogue = "\n".join(selected)

    omitted = len(turn_lines) - len(selected)
    header  = (
        f"[RECENT CONVERSATION — {len(selected)} turns"
        + (f", {omitted} older omitted" if omitted else "")
        + "]"
    )
    return f"{header}\n{dialogue}"


# ---------------------------------------------------------------------------
# assemble_context_payload
# ---------------------------------------------------------------------------

def assemble_context_payload(
    wm:           WorkingMemory,
    user_query:   str,
    query_type:   str  = "unknown",
    goal:         str  = "provide a medical answer",
    rag_context:  str  = "",
    token_budget: int  = DEFAULT_TOKEN_BUDGET,
) -> ContextPayload:
    """
    Assemble all memory layers into a structured ContextPayload.

    This is the primary orchestration function. It:
      1. Builds each section independently.
      2. Enforces per-section token caps.
      3. Validates the total against `token_budget`, trimming the summary
         first (least critical) if over budget.
      4. Returns a ContextPayload with placeholder RAG slots ready for
         the pipeline to fill.

    Args:
        wm:           WorkingMemory from get_working_memory().
        user_query:   The raw current user question (not yet rewritten).
        query_type:   Active task type for system prompt adaptation.
        goal:         Human-readable retrieval goal.
        rag_context:  Pre-retrieved RAG documents (may be empty at call time).
        token_budget: Maximum total tokens for the assembled context.

    Returns:
        A fully populated ContextPayload.
    """
    risk = wm.risk_level

    # ── Build each section ─────────────────────────────────────────────────
    # `system_context` is intentionally empty — the layered composer in
    # app.services.orchestration.prompt_layers builds the real system prompt
    # at the orchestrator level. The field stays in ContextPayload for
    # backward compat with any external caller that destructures it.
    system_ctx  = ""
    memory_ctx  = build_memory_context(wm)
    conv_ctx    = build_conversation_context(wm)

    # Rolling summary — trim harder than other sections when over budget
    raw_summary = wm.summary.strip() if wm.has_summary else ""
    summary_ctx = ""
    if raw_summary:
        summary_ctx = _trim_to_tokens(
            f"[PRIOR CONTEXT]\n{raw_summary}",
            SUMMARY_SECTION_MAX_TOKENS,
        )

    # RAG section header (content injected externally)
    rag_ctx = ""
    if rag_context.strip():
        rag_ctx = _trim_to_tokens(
            f"[CLINICAL KNOWLEDGE]\n{rag_context.strip()}",
            RAG_SECTION_MAX_TOKENS,
        )

    query_ctx = _trim_to_tokens(user_query.strip(), QUERY_MAX_TOKENS)

    # ── Token accounting ──────────────────────────────────────────────────
    total = sum(
        _estimate_tokens(s)
        for s in [system_ctx, memory_ctx, summary_ctx, conv_ctx, rag_ctx, query_ctx]
        if s
    )

    # If over budget, trim summary progressively
    if total > token_budget and summary_ctx:
        excess_tokens = total - token_budget
        new_summary_tokens = max(50, SUMMARY_SECTION_MAX_TOKENS - excess_tokens)
        summary_ctx = _trim_to_tokens(
            f"[PRIOR CONTEXT]\n{raw_summary}",
            new_summary_tokens,
        )
        total = sum(
            _estimate_tokens(s)
            for s in [system_ctx, memory_ctx, summary_ctx, conv_ctx, rag_ctx, query_ctx]
            if s
        )

    return ContextPayload(
        system_context       = system_ctx,
        memory_context       = memory_ctx,
        summary_context      = summary_ctx,
        conversation_context = conv_ctx,
        user_query           = query_ctx,
        rag_context          = rag_ctx,
        session_id           = wm.session_id,
        active_task          = wm.active_task,
        risk_level           = risk,
        estimated_tokens     = total,
    )


# ---------------------------------------------------------------------------
# build_final_prompt
# ---------------------------------------------------------------------------

def build_final_prompt(
    payload:              ContextPayload,
    include_state:        bool = True,
    include_summary:      bool = True,
    include_turns:        bool = True,
    include_rag:          bool = True,
) -> FinalPrompt:
    """
    Render a ContextPayload into a FinalPrompt ready for LLM submission.

    Produces both:
      - `messages`: an OpenAI-style chat messages list
      - `user_text`: a single concatenated user block (for completion APIs)

    Sections are included/excluded via flags so the caller can ablate
    components during testing without rebuilding the payload.

    Args:
        payload:         The assembled ContextPayload.
        include_state:   Include the structured state block.
        include_summary: Include the rolling summary block.
        include_turns:   Include the recent conversation block.
        include_rag:     Include the RAG/clinical knowledge block.

    Returns:
        FinalPrompt with messages list, user_text, and token estimate.
    """
    # ── Build ordered user block ───────────────────────────────────────────
    user_parts: list[str] = []

    if include_state and payload.memory_context:
        user_parts.append(payload.memory_context)

    if include_summary and payload.summary_context:
        user_parts.append(payload.summary_context)

    if include_turns and payload.conversation_context:
        user_parts.append(payload.conversation_context)

    # Tool outputs (future — injected externally)
    for tool_out in payload.tool_outputs:
        user_parts.append(f"[TOOL OUTPUT]\n{tool_out.strip()}")

    # Citations block (future)
    if payload.citations:
        cit_block = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(payload.citations))
        user_parts.append(f"[SOURCES]\n{cit_block}")

    # The live question appears before retrieved clinical knowledge to keep
    # final assembly order aligned with the memory-aware GraphRAG contract.
    user_parts.append(f"[USER QUESTION]\n{payload.user_query}")

    if include_rag and payload.rag_context:
        user_parts.append(payload.rag_context)

    user_text = "\n\n".join(user_parts)

    messages: list[dict[str, str]] = [
        {"role": "system",  "content": payload.system_context},
        {"role": "user",    "content": user_text},
    ]

    total_tokens = _estimate_tokens(payload.system_context) + _estimate_tokens(user_text)

    return FinalPrompt(
        messages     = messages,
        system_text  = payload.system_context,
        user_text    = user_text,
        total_tokens = total_tokens,
        payload      = payload,
    )
