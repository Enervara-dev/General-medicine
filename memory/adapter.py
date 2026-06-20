"""
LongitudinalMemoryAdapter — the integration surface the pipeline talks to.

Replaces graphrag/memory/session_adapter.py. Exposes both async-native and
sync-compatibility methods so the existing CLI pipeline keeps working during
migration.

Key contract:
    load(patient_id, request_id) → MemoryContext
        Used by Stage -2 of GraphRAGPipeline to fetch memory before retrieval.

    record_turn(patient_id, session_id, user_utterance, assistant_answer)
        Persists raw turns + triggers the update_after_turn orchestrator.

    build_prompt_block(ctx) → str
        Renders MemoryContext into the prompt-ready string fragment the
        pipeline injects before the medical RAG context.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from sqlalchemy import select

from memory.cache.redis_cache import RedisStateCache
from memory.db.session import get_session
from memory.models.conversation_event import ConversationEvent, ConvRole
from memory.models.patient import Patient
from memory.pipelines.update_after_turn import update_after_turn
from memory.schemas.retrieval import MemoryContext
from memory.services.retrieval import EmbedderProtocol, RetrievalService

logger = logging.getLogger(__name__)


class LongitudinalMemoryAdapter:
    def __init__(
        self,
        *,
        retrieval: RetrievalService | None = None,
        cache: RedisStateCache | None = None,
        embed_query: EmbedderProtocol | None = None,
    ) -> None:
        self._cache = cache or RedisStateCache()
        self._retrieval = retrieval or RetrievalService(cache=self._cache)
        self._embed_query = embed_query  # async (str) -> list[float], or None

    # ------------------------------------------------------------------
    # Async API (FastAPI-ready)
    # ------------------------------------------------------------------

    async def aload(
        self,
        *,
        patient_id: uuid.UUID,
        request_id: uuid.UUID,
        query_text: str,
    ) -> MemoryContext:
        async with get_session() as session:
            ctx = await self._retrieval.build_context(
                session,
                patient_id=patient_id,
                request_id=request_id,
                query_text=query_text,
                embed_query=self._embed_query,
            )
        return ctx

    async def arecord_assistant_turn(
        self,
        *,
        patient_id: uuid.UUID,
        session_id: str,
        request_id: uuid.UUID,
        answer: str,
        analysis_payload: dict[str, Any] | None = None,
    ) -> None:
        async with get_session() as session:
            session.add(ConversationEvent(
                patient_id=patient_id,
                session_id=session_id,
                role=ConvRole.ASSISTANT,
                content=answer,
                analysis_payload=analysis_payload,
                request_id=request_id,
            ))

    async def arecord_user_turn(
        self,
        *,
        patient_id: uuid.UUID,
        session_id: str,
        user_utterance: str,
        request_id: uuid.UUID,
    ) -> None:
        """Trigger the full extraction + consolidation flow for a user turn."""
        await update_after_turn(
            patient_id=patient_id,
            session_id=session_id,
            user_utterance=user_utterance,
            request_id=request_id,
        )

    async def aensure_patient(
        self,
        *,
        patient_id: uuid.UUID,
        external_id: str | None = None,
    ) -> None:
        """
        Upsert a Patient row. CLI clients supply an externally-managed UUID;
        the row must exist before any conversation_event / clinical_fact rows
        can reference it (FK constraint).
        """
        async with get_session() as session:
            row = await session.get(Patient, patient_id)
            if row is None:
                session.add(Patient(id=patient_id, external_id=external_id))

    async def aload_recent_turns(
        self,
        *,
        session_id: str,
        limit: int = 6,
    ) -> list[ConversationEvent]:
        """
        Pull the last N raw conversation events for the current session.

        conversation_event is the audit log; usually it's not prompted. But the
        LLM still needs the last few turns of the *same* conversation to handle
        in-session follow-ups ("is it serious?") that don't translate into
        structured facts. This is the narrow channel through which raw turns
        reach the prompt.
        """
        if limit <= 0:
            return []
        async with get_session() as session:
            stmt = (
                select(ConversationEvent)
                .where(ConversationEvent.session_id == session_id)
                .order_by(ConversationEvent.created_at.desc())
                .limit(limit)
            )
            rows = (await session.execute(stmt)).scalars().all()
        # Return oldest-first so the prompt reads chronologically.
        return list(reversed(rows))

    async def aclose(self) -> None:
        await self._cache.close()

    # ------------------------------------------------------------------
    # Sync compatibility shims (CLI / legacy callers only)
    # ------------------------------------------------------------------

    def load(
        self,
        *,
        patient_id: uuid.UUID,
        request_id: uuid.UUID,
        query_text: str,
    ) -> MemoryContext:
        return _run(self.aload(
            patient_id=patient_id, request_id=request_id, query_text=query_text
        ))

    def record_user_turn(
        self,
        *,
        patient_id: uuid.UUID,
        session_id: str,
        user_utterance: str,
        request_id: uuid.UUID,
    ) -> None:
        _run(self.arecord_user_turn(
            patient_id=patient_id,
            session_id=session_id,
            user_utterance=user_utterance,
            request_id=request_id,
        ))

    def record_assistant_turn(
        self,
        *,
        patient_id: uuid.UUID,
        session_id: str,
        request_id: uuid.UUID,
        answer: str,
        analysis_payload: dict[str, Any] | None = None,
    ) -> None:
        _run(self.arecord_assistant_turn(
            patient_id=patient_id,
            session_id=session_id,
            request_id=request_id,
            answer=answer,
            analysis_payload=analysis_payload,
        ))

    def load_recent_turns(
        self,
        *,
        session_id: str,
        limit: int = 6,
    ) -> list[ConversationEvent]:
        return _run(self.aload_recent_turns(session_id=session_id, limit=limit))

    def ensure_patient(
        self,
        *,
        patient_id: uuid.UUID,
        external_id: str | None = None,
    ) -> None:
        _run(self.aensure_patient(patient_id=patient_id, external_id=external_id))

    # ------------------------------------------------------------------
    # Prompt assembly (the only thing that becomes prompt text)
    # ------------------------------------------------------------------

    @staticmethod
    def build_prompt_block(ctx: MemoryContext) -> str:
        """Render MemoryContext into the prompt fragment Stage 4 will inject."""
        s = ctx.snapshot

        sections: list[str] = []

        if s.summary_text:
            sections.append(f"=== CURRENT PATIENT STATE ===\n{s.summary_text}")

        active_lines: list[str] = []
        if s.allergies:
            active_lines.append("Allergies: " + ", ".join(_fmt_name(a) for a in s.allergies))
        if s.medications:
            active_lines.append("Medications:")
            active_lines.extend(f"  - {_fmt_med(m)}" for m in s.medications)
        if s.symptoms:
            active_lines.append("Active symptoms:")
            active_lines.extend(f"  - {_fmt_symptom(sy)}" for sy in s.symptoms)
        if s.conditions:
            active_lines.append("Conditions: " + ", ".join(_fmt_name(c) for c in s.conditions))
        if s.unresolved_followups:
            active_lines.append("Unresolved: " + ", ".join(
                _fmt_name(f) for f in s.unresolved_followups
            ))
        if active_lines:
            sections.append("=== ACTIVE CLINICAL FACTS ===\n" + "\n".join(active_lines))

        if ctx.recent_episodes:
            sections.append("=== RECENT EPISODES ===\n" + "\n".join(
                f"- {e.occurred_at.date()}: {e.title} — {e.description[:140]}"
                for e in ctx.recent_episodes
            ))

        if ctx.semantic_recalls:
            sections.append("=== RELEVANT PRIOR DISCUSSION ===\n" + "\n".join(
                f"- {sn.content[:200]}" for sn in ctx.semantic_recalls
            ))

        return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fmt_name(entry: dict) -> str:
    return entry.get("name", "<unknown>")


def _fmt_med(entry: dict) -> str:
    name = entry.get("name", "<unknown>")
    val = entry.get("value", {}) or {}
    bits = [name]
    if val.get("dose"):
        unit = val.get("unit", "")
        bits.append(f"{val['dose']}{unit}")
    if val.get("frequency"):
        bits.append(val["frequency"])
    if val.get("route"):
        bits.append(val["route"])
    return " ".join(bits)


def _fmt_symptom(entry: dict) -> str:
    name = entry.get("name", "<unknown>")
    val = entry.get("value", {}) or {}
    bits = [name]
    if val.get("severity"):
        bits.append(f"({val['severity']})")
    if entry.get("since"):
        bits.append(f"since {entry['since'][:10]}")
    return " ".join(bits)


def _run(coro):
    """Sync bridge for CLI callers. Refuses to run inside a live event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "LongitudinalMemoryAdapter sync methods cannot be called from inside "
        "a running event loop. Use the async API (aload/arecord_*) instead."
    )
