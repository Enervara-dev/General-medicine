from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from Memory_Layer.session_memory import (
    Message,
    RiskLevel,
    Role,
    SessionManager,
    SessionMemory,
    assemble_context_payload,
    build_final_prompt,
    extract_state,
    get_working_memory,
    maybe_summarize,
)
from Memory_Layer.session_memory.context_builder import ContextPayload, FinalPrompt
from Memory_Layer.session_memory.retriever import WorkingMemory

logger = logging.getLogger(__name__)


@dataclass
class MemoryAwareSession:
    session: SessionMemory
    working_memory: WorkingMemory


class SessionMemoryAdapter:
    """
    Synchronous facade around the async Redis-backed memory layer.

    GraphRAG stays focused on retrieval and generation; this adapter owns
    loading, state updates, summarization, and prompt assembly.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        self.redis_url = redis_url

    def load(self, session_id: str) -> MemoryAwareSession:
        return self._run(self._load(session_id))

    def build_retrieval_query(self, query_text: str, wm: WorkingMemory) -> str:
        parts = [f"Current question: {query_text.strip()}"]

        state_terms = []
        state = wm.state
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
        compact_terms = []
        for term in state_terms:
            clean = str(term).strip()
            if clean and clean not in seen:
                seen.add(clean)
                compact_terms.append(clean)

        if compact_terms:
            parts.append("Patient clinical context: " + ", ".join(compact_terms[:30]))

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

    def assemble_payload(
        self,
        wm: WorkingMemory,
        user_query: str,
        query_type: str,
        goal: str,
        vector_context: str,
        graph_context: str,
    ) -> ContextPayload:
        rag_context = self._combine_rag_context(vector_context, graph_context)
        return assemble_context_payload(
            wm=wm,
            user_query=user_query,
            query_type=query_type,
            goal=goal,
            rag_context=rag_context,
        )

    def build_final_prompt(
        self,
        wm: WorkingMemory,
        user_query: str,
        query_type: str,
        goal: str,
        vector_context: str,
        graph_context: str,
    ) -> FinalPrompt:
        payload = self.assemble_payload(
            wm=wm,
            user_query=user_query,
            query_type=query_type,
            goal=goal,
            vector_context=vector_context,
            graph_context=graph_context,
        )
        return build_final_prompt(payload)

    def update_after_interaction(
        self,
        session: SessionMemory,
        user_query: str,
        assistant_answer: str,
        analysis: dict[str, Any] | None = None,
        query_type: str | None = None,
    ) -> SessionMemory:
        return self._run(
            self._update_after_interaction(
                session=session,
                user_query=user_query,
                assistant_answer=assistant_answer,
                analysis=analysis or {},
                query_type=query_type,
            )
        )

    async def _load(self, session_id: str) -> MemoryAwareSession:
        async with self._manager() as mgr:
            session = await mgr.load_session(session_id)
            if session is None:
                session = await mgr.create_session(session_id=session_id)

        return MemoryAwareSession(
            session=session,
            working_memory=get_working_memory(session),
        )

    async def _update_after_interaction(
        self,
        session: SessionMemory,
        user_query: str,
        assistant_answer: str,
        analysis: dict[str, Any],
        query_type: str | None,
    ) -> SessionMemory:
        user_msg = Message(
            role=Role.USER,
            content=user_query,
            intent=analysis.get("intent"),
            query_type=query_type or analysis.get("intent"),
            risk_level=self._risk_from_analysis(analysis),
        )
        session.state = extract_state(session, user_msg)
        session.add_turn(user_msg)

        if assistant_answer:
            session.add_turn(Message(role=Role.ASSISTANT, content=assistant_answer))

        session = maybe_summarize(session)

        async with self._manager() as mgr:
            await mgr.save_session(session)

        return session

    def _manager(self) -> SessionManager:
        if self.redis_url:
            return SessionManager(redis_url=self.redis_url)
        return SessionManager()

    @staticmethod
    def _combine_rag_context(vector_context: str, graph_context: str) -> str:
        parts = []
        if vector_context:
            parts.append("=== AVAILABLE CLINICAL SUMMARIES ===\n" + vector_context.strip())
        if graph_context:
            parts.append("=== EXPERT CLINICAL GRAPH RELATIONS ===\n" + graph_context.strip())
        return "\n\n".join(parts)

    @staticmethod
    def _risk_from_analysis(analysis: dict[str, Any]) -> RiskLevel:
        raw = (analysis or {}).get("risk_level") or "none"
        try:
            return RiskLevel(str(raw).lower())
        except ValueError:
            return RiskLevel.NONE

    @staticmethod
    def _run(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        raise RuntimeError(
            "GraphRAGPipeline.run() is synchronous and cannot run inside an active "
            "event loop. Use the async memory layer directly from async servers."
        )
