"""
RetrievalService — build the MemoryContext handed to the pipeline.

Rules:
- Structured retrieval runs FIRST. Allergies always included. Active meds
  always included. Symptoms ranked by importance * decay_score.
- Recent episodes (last 30 days) bounded to top-3 by importance * recency.
- pgvector recall is INVOKED ONLY when structured retrieval cannot cover
  the query — heuristic: query mentions concepts not present in the
  structured active fact set.
- Every retrieval is logged to retrieval_log for provenance.
- Hot snapshot pulled from Redis cache when available; cache invalidated
  on writes via RedisStateCache.invalidate().
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from memory.cache.redis_cache import RedisStateCache
from memory.models.clinical_fact import ClinicalFact, FactStatus, FactType
from memory.models.episodic_memory import EpisodicMemory
from memory.models.patient_state import PatientState
from memory.models.retrieval_log import RetrievalLog
from memory.schemas.fact import ClinicalFactDTO
from memory.schemas.retrieval import (
    EpisodicMemoryDTO,
    MemoryContext,
    RetrievalDecision,
    SemanticMemoryDTO,
)
from memory.schemas.state import PatientStateSnapshot

logger = logging.getLogger(__name__)


# Tunable thresholds — promote to RuntimeConfig as deployment matures.
_RECENT_EPISODE_DAYS = 30
_MAX_EPISODES = 3
_MAX_SEMANTIC = 2
_SEMANTIC_SIMILARITY_FLOOR = 0.75
_SEMANTIC_DECAY_FLOOR = 0.3

_CHARS_PER_TOKEN = 4  # rough — replace with tiktoken in Wave 4 of full plan


class RetrievalService:
    """Assemble a MemoryContext for one user turn."""

    def __init__(self, cache: RedisStateCache | None = None) -> None:
        self._cache = cache or RedisStateCache()

    async def build_context(
        self,
        session: AsyncSession,
        *,
        patient_id: uuid.UUID,
        request_id: uuid.UUID,
        query_text: str,
        embed_query: "EmbedderProtocol | None" = None,
    ) -> MemoryContext:
        """
        Compose the memory context for `query_text`.

        `embed_query`, if provided, is an async callable returning the query's
        embedding vector. Only invoked when structured retrieval cannot cover
        the query.
        """
        # 1) Hot cache → snapshot
        snapshot = await self._cache.get(patient_id)
        used_cache = snapshot is not None
        if snapshot is None:
            snapshot = await self._load_snapshot(session, patient_id)
            await self._cache.set(snapshot)

        # 2) Structured: active facts (allergies + meds always, symptoms ranked)
        active_facts = await self._load_active_facts(session, patient_id)
        active_dtos = [ClinicalFactDTO.model_validate(f) for f in active_facts]

        # 3) Recent episodes
        episodes = await self._load_recent_episodes(session, patient_id)
        episode_dtos = [EpisodicMemoryDTO.model_validate(e) for e in episodes]

        # 4) Vector recall — only if structured did not satisfy
        semantic_dtos: list[SemanticMemoryDTO] = []
        vector_invoked = False
        if embed_query is not None and self._should_invoke_vector(query_text, active_dtos):
            vector_invoked = True
            try:
                semantic_dtos = await self._semantic_recall(
                    session, patient_id, query_text, embed_query
                )
            except Exception as exc:
                logger.warning("Semantic recall failed; degrading gracefully: %s", exc)

        decision = RetrievalDecision(
            strategy=self._classify_strategy(episode_dtos, semantic_dtos, used_cache),
            used_cache=used_cache,
            vector_recall_invoked=vector_invoked,
            fact_count=len(active_dtos),
            episode_count=len(episode_dtos),
            semantic_count=len(semantic_dtos),
        )

        ctx = MemoryContext(
            patient_id=patient_id,
            request_id=request_id,
            snapshot=snapshot,
            active_facts=active_dtos,
            recent_episodes=episode_dtos,
            semantic_recalls=semantic_dtos,
            decision=decision,
            estimated_tokens=self._estimate_tokens(snapshot, active_dtos, episode_dtos, semantic_dtos),
        )

        await self._log_retrieval(session, ctx, query_text)
        return ctx

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @staticmethod
    async def _load_snapshot(
        session: AsyncSession, patient_id: uuid.UUID
    ) -> PatientStateSnapshot:
        row = await session.get(PatientState, patient_id)
        if row is None:
            # First-touch: return an empty snapshot.
            return PatientStateSnapshot(
                patient_id=patient_id,
                version=0,
                last_consolidated_at=datetime.now(tz=timezone.utc),
            )
        return PatientStateSnapshot(
            patient_id=patient_id,
            version=row.version,
            last_consolidated_at=row.last_consolidated_at,
            summary_text=row.summary_text,
            risk_level=row.risk_level,
            **(row.snapshot or {}),
        )

    @staticmethod
    async def _load_active_facts(
        session: AsyncSession, patient_id: uuid.UUID
    ) -> Sequence[ClinicalFact]:
        # Allergies and meds always included; symptoms ordered by importance × decay.
        stmt = (
            select(ClinicalFact)
            .where(
                ClinicalFact.patient_id == patient_id,
                ClinicalFact.status == FactStatus.ACTIVE,
            )
            .order_by(
                (ClinicalFact.importance * ClinicalFact.decay_score).desc(),
                ClinicalFact.observed_at.desc(),
            )
        )
        return (await session.execute(stmt)).scalars().all()

    @staticmethod
    async def _load_recent_episodes(
        session: AsyncSession, patient_id: uuid.UUID
    ) -> Sequence[EpisodicMemory]:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=_RECENT_EPISODE_DAYS)
        stmt = (
            select(EpisodicMemory)
            .where(
                EpisodicMemory.patient_id == patient_id,
                EpisodicMemory.occurred_at >= cutoff,
            )
            .order_by(
                EpisodicMemory.importance.desc(), EpisodicMemory.occurred_at.desc()
            )
            .limit(_MAX_EPISODES)
        )
        return (await session.execute(stmt)).scalars().all()

    # ------------------------------------------------------------------
    # Vector recall (narrow channel)
    # ------------------------------------------------------------------

    @staticmethod
    def _should_invoke_vector(
        query_text: str, active_facts: list[ClinicalFactDTO]
    ) -> bool:
        """
        Crude heuristic: invoke vector recall when the query mentions terms
        that are not represented in the structured fact set. Production should
        use the gatekeeper's extracted entities instead of substring matching.
        """
        query_lower = query_text.lower()
        names = {f.canonical_name.lower() for f in active_facts}
        return not any(n and n in query_lower for n in names)

    @staticmethod
    async def _semantic_recall(
        session: AsyncSession,
        patient_id: uuid.UUID,
        query_text: str,
        embed_query: "EmbedderProtocol",
    ) -> list[SemanticMemoryDTO]:
        from memory.models.semantic_memory import SemanticMemory  # local: optional pgvector

        vector = await embed_query(query_text)
        # pgvector cosine distance — smaller is closer; similarity = 1 - distance
        stmt = (
            select(
                SemanticMemory,
                (1 - SemanticMemory.embedding.cosine_distance(vector)).label("similarity"),
            )
            .where(
                SemanticMemory.patient_id == patient_id,
                SemanticMemory.decay_score >= _SEMANTIC_DECAY_FLOOR,
            )
            .order_by(SemanticMemory.embedding.cosine_distance(vector))
            .limit(_MAX_SEMANTIC * 3)  # over-fetch, filter by similarity floor
        )
        rows = (await session.execute(stmt)).all()

        out: list[SemanticMemoryDTO] = []
        for row, similarity in rows:
            if similarity < _SEMANTIC_SIMILARITY_FLOOR:
                continue
            dto = SemanticMemoryDTO.model_validate(row)
            dto.similarity = float(similarity)
            out.append(dto)
            row.access_count = (row.access_count or 0) + 1
            row.last_accessed_at = datetime.now(tz=timezone.utc)
            session.add(row)
            if len(out) >= _MAX_SEMANTIC:
                break
        return out

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_strategy(
        episodes: list, semantic: list, used_cache: bool
    ) -> str:
        if used_cache and not episodes and not semantic:
            return "cache_hit"
        if episodes and semantic:
            return "structured_plus_episodic_plus_semantic"
        if semantic:
            return "structured_plus_semantic"
        if episodes:
            return "structured_plus_episodic"
        return "structured_only"

    @staticmethod
    def _estimate_tokens(
        snapshot: PatientStateSnapshot,
        facts: list[ClinicalFactDTO],
        episodes: list[EpisodicMemoryDTO],
        semantic: list[SemanticMemoryDTO],
    ) -> int:
        chars = len(snapshot.summary_text or "")
        chars += sum(len(f.canonical_name) + 40 for f in facts)
        chars += sum(len(e.title) + len(e.description) for e in episodes)
        chars += sum(len(s.content) for s in semantic)
        return chars // _CHARS_PER_TOKEN

    @staticmethod
    async def _log_retrieval(
        session: AsyncSession, ctx: MemoryContext, query_text: str
    ) -> None:
        row = RetrievalLog(
            patient_id=ctx.patient_id,
            request_id=ctx.request_id,
            query_text=query_text,
            routing_mode="longitudinal",  # caller may override before insert
            retrieved_fact_ids=[f.id for f in ctx.active_facts],
            retrieved_episode_ids=[e.id for e in ctx.recent_episodes],
            retrieved_semantic_ids=[s.id for s in ctx.semantic_recalls],
            retrieval_strategy=ctx.decision.strategy,
            prompt_tokens_in=ctx.estimated_tokens,
        )
        session.add(row)
        await session.flush()


# Lightweight Protocol so we don't hard-couple to a specific embedder.
from typing import Protocol


class EmbedderProtocol(Protocol):
    async def __call__(self, text: str) -> list[float]: ...  # noqa: E704
