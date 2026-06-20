"""
Async session memory primitives — the FastAPI path's drop-in replacement
for [graphrag.memory.session_adapter.SessionMemoryAdapter], minus the
asyncio.run() bridges that crash inside an event loop.

Each function here calls Memory_Layer.SessionManager directly. The
SessionManager is already async-native; the sync facade only existed
because the CLI pipeline was sync.

Layout:
    load_session(mgr, session_id)      → SessionBundle
    build_retrieval_query(query, wm)   → str (pure)
    assemble_memory_payload(...)       → ContextPayload (reuse existing builder)
    save_after_turn(mgr, ...)          → SessionMemory (with updated state + turns)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from Memory_Layer.session_memory import (
    Message,
    RiskLevel,
    Role,
    SessionManager,
    SessionMemory,
    assemble_context_payload,
    extract_state,
    get_working_memory,
    maybe_summarize,
)
from Memory_Layer.session_memory.context_builder import ContextPayload
from Memory_Layer.session_memory.retriever import WorkingMemory


@dataclass
class SessionBundle:
    session: SessionMemory
    working_memory: WorkingMemory


async def load_session(mgr: SessionManager, session_id: str) -> SessionBundle:
    """Load the session from Redis (or in-memory fallback) and project working memory."""
    session = await mgr.load_session(session_id)
    if session is None:
        session = await mgr.create_session(session_id=session_id)
    return SessionBundle(session=session, working_memory=get_working_memory(session))


def build_retrieval_query(query_text: str, wm: WorkingMemory) -> str:
    """
    Mirror of SessionMemoryAdapter.build_retrieval_query. Pure function — no I/O.

    Composes the original question + structured clinical context + rolling
    summary + last few user statements into a single string that the vector
    retriever embeds.
    """
    parts = [f"Current question: {query_text.strip()}"]

    state = wm.state
    state_terms: list[str] = []
    for values in (
        state.symptoms,
        state.drugs,
        state.allergies,
        state.conditions,
        state.chronic_conditions,
        state.duration,
        state.previous_concerns,
        state.discussed_entities,
    ):
        state_terms.extend(values)

    seen: set[str] = set()
    compact: list[str] = []
    for term in state_terms:
        clean = str(term).strip()
        if clean and clean not in seen:
            seen.add(clean)
            compact.append(clean)

    if compact:
        parts.append("Patient clinical context: " + ", ".join(compact[:30]))

    if wm.summary:
        parts.append("Rolling summary: " + wm.summary[:600])

    if wm.recent_turns:
        recent_user_turns = [
            t.content.strip()
            for t in wm.recent_turns[-4:]
            if t.role == Role.USER and t.content.strip()
        ]
        if recent_user_turns:
            parts.append("Recent patient statements: " + " | ".join(recent_user_turns))

    return "\n".join(parts)


def assemble_memory_payload(
    *,
    wm: WorkingMemory,
    user_query: str,
    query_type: str,
    goal: str,
    vector_context: str,
    graph_context: str,
) -> ContextPayload:
    """Compose the LLM's context block. Pure wrapper around the existing builder."""
    rag_context = _combine_rag_context(vector_context, graph_context)
    return assemble_context_payload(
        wm=wm,
        user_query=user_query,
        query_type=query_type,
        goal=goal,
        rag_context=rag_context,
    )


async def save_after_turn(
    mgr: SessionManager,
    *,
    session: SessionMemory,
    user_query: str,
    assistant_answer: str,
    analysis: dict[str, Any],
    query_type: str | None,
) -> SessionMemory:
    """Update structured state, append turns, maybe summarize, then persist."""
    user_msg = Message(
        role=Role.USER,
        content=user_query,
        intent=analysis.get("intent"),
        query_type=query_type or analysis.get("intent"),
        risk_level=_risk_from_analysis(analysis),
    )
    session.state = extract_state(session, user_msg)
    session.add_turn(user_msg)

    if assistant_answer:
        session.add_turn(Message(role=Role.ASSISTANT, content=assistant_answer))

    session = maybe_summarize(session)
    await mgr.save_session(session)
    return session


def _combine_rag_context(vector_context: str, graph_context: str) -> str:
    parts: list[str] = []
    if vector_context:
        parts.append("=== AVAILABLE CLINICAL SUMMARIES ===\n" + vector_context.strip())
    if graph_context:
        parts.append("=== EXPERT CLINICAL GRAPH RELATIONS ===\n" + graph_context.strip())
    return "\n\n".join(parts)


def _risk_from_analysis(analysis: dict[str, Any]) -> RiskLevel:
    raw = (analysis or {}).get("risk_level") or "none"
    try:
        return RiskLevel(str(raw).lower())
    except ValueError:
        return RiskLevel.NONE
