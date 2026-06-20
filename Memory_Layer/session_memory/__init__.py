"""
session_memory
──────────────
Redis-backed session memory layer for the Enervera Medical RAG system.

Public surface:
    Models       : Message, Role, RiskLevel, SessionMemory, StructuredState
    Session Mgr  : SessionManager, session_context
    State Extract: extract_entities, extract_state, merge_state, update_preferences
    Summarizer   : maybe_summarize, summarize_session, should_summarize
    Retriever    : get_working_memory, format_working_memory, WorkingMemory
    Context Bld  : assemble_context_payload, build_memory_context, build_conversation_context, ContextPayload, FinalPrompt
                   (system-prompt construction now lives in app.services.orchestration.prompt_layers)
"""

from .models import (
    MAX_RECENT_TURNS,
    Message,
    Role,
    RiskLevel,
    SessionMemory,
    StructuredState,
)
from .session_manager import (
    SESSION_TTL_SEC,
    SessionManager,
    session_context,
)
from .state_extractor import (
    RawEntities,
    extract_entities,
    extract_state,
    merge_state,
    update_preferences,
)
from .summarizer import (
    KEEP_LAST_N,
    SUMMARIZE_THRESHOLD,
    build_turn_summary,
    maybe_summarize,
    should_summarize,
    summarize_session,
)
from .retriever import (
    WorkingMemory,
    format_working_memory,
    get_recent_turns,
    get_structured_state,
    get_summary,
    get_working_memory,
)
from .context_builder import (
    ContextPayload,
    FinalPrompt,
    assemble_context_payload,
    build_conversation_context,
    build_final_prompt,
    build_memory_context,
)

__all__ = [
    # Models
    "Message", "Role", "RiskLevel", "SessionMemory",
    "StructuredState", "MAX_RECENT_TURNS",
    # Session manager
    "SessionManager", "session_context", "SESSION_TTL_SEC",
    # State extractor
    "RawEntities", "extract_entities", "extract_state",
    "merge_state", "update_preferences",
    # Summarizer
    "should_summarize", "summarize_session", "maybe_summarize",
    "build_turn_summary", "SUMMARIZE_THRESHOLD", "KEEP_LAST_N",
    # Retriever
    "WorkingMemory", "get_recent_turns", "get_summary",
    "get_structured_state", "get_working_memory", "format_working_memory",
    # Context builder
    "ContextPayload", "FinalPrompt",
    "build_memory_context", "build_conversation_context",
    "assemble_context_payload", "build_final_prompt",
]
