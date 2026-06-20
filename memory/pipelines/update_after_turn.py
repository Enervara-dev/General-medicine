"""
update_after_turn — the per-turn orchestrator.

Called after each user message lands. Runs:
    extraction → safety check → consolidation → snapshot recompute
    → optional summary regeneration → cache invalidate.

Designed to run inline (in-process asyncio task) for now; can be moved to
a background worker (arq / celery) without changing the call site.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from memory.cache.redis_cache import RedisStateCache
from memory.db.session import get_session
from memory.models.conversation_event import ConversationEvent, ConvRole
from memory.schemas.fact import RiskFlag
from memory.schemas.state import PatientStateSnapshot
from memory.services.consolidation import ConsolidationService
from memory.services.extraction import ExtractionService
from memory.services.safety import SafetyService
from memory.services.summarization import SummarizationService

logger = logging.getLogger(__name__)


# Re-summarize when at least this many active facts changed since last summary,
# or if a safety-critical fact was added/removed. Cheap heuristic for now.
_SUMMARY_DELTA_THRESHOLD = 2


@dataclass
class UpdateResult:
    snapshot: PatientStateSnapshot
    risk_flags: list[RiskFlag]
    facts_written: int
    summary_regenerated: bool


async def update_after_turn(
    patient_id: uuid.UUID,
    session_id: str,
    user_utterance: str,
    *,
    request_id: uuid.UUID | None = None,
    extraction: ExtractionService | None = None,
    safety: SafetyService | None = None,
    consolidation: ConsolidationService | None = None,
    summarization: SummarizationService | None = None,
    cache: RedisStateCache | None = None,
) -> UpdateResult:
    extraction = extraction or ExtractionService()
    safety = safety or SafetyService()
    consolidation = consolidation or ConsolidationService()
    summarization = summarization or SummarizationService()
    cache = cache or RedisStateCache()
    request_id = request_id or uuid.uuid4()

    async with get_session() as session:
        # 1) Persist the raw turn (audit log, never prompted).
        event = ConversationEvent(
            patient_id=patient_id,
            session_id=session_id,
            role=ConvRole.USER,
            content=user_utterance,
            request_id=request_id,
        )
        session.add(event)
        await session.flush()

        # 2) Extract fact candidates (LLM, JSON-schema-strict).
        candidates = await extraction.extract(user_utterance)

        # 3) Safety pre-flight (allergy, interaction, critical pairs).
        risk_flags = await safety.check_candidates(session, patient_id, candidates)
        if any(f.severity == "block" for f in risk_flags):
            logger.warning(
                "Safety BLOCK for patient %s on request %s: %s",
                patient_id, request_id,
                [f.message for f in risk_flags if f.severity == "block"],
            )
            # Caller decides how to surface this; we still persist candidates as
            # 'refuted' so the audit trail captures the intent + the block.

        # 4) Consolidate into clinical_fact (merge/supersede/contradict).
        written = await consolidation.consolidate(
            session,
            patient_id,
            candidates,
            source_event_id=event.id,
        )

        # 5) Recompute denormalized snapshot.
        snapshot = await consolidation.recompute_snapshot(session, patient_id)

        # 6) Regenerate prose summary only on material change.
        summary_regenerated = False
        if _should_regenerate_summary(len(written)):
            snapshot.summary_text = await summarization.regenerate(snapshot)
            # write the new summary back via PatientState row
            from memory.models.patient_state import PatientState
            ps = await session.get(PatientState, patient_id)
            if ps is not None:
                ps.summary_text = snapshot.summary_text
                session.add(ps)
            summary_regenerated = True

        # 7) Refresh cache with the freshly-built snapshot.
        await cache.set(snapshot)

    return UpdateResult(
        snapshot=snapshot,
        risk_flags=risk_flags,
        facts_written=len(written),
        summary_regenerated=summary_regenerated,
    )


def _should_regenerate_summary(facts_written: int) -> bool:
    return facts_written >= _SUMMARY_DELTA_THRESHOLD
