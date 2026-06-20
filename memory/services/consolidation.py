"""
ConsolidationService — merge ClinicalFactCandidate[] into clinical_fact rows.

This is where temporal validity, contradiction resolution, and supersede
chains are enforced. The service runs after every turn (or on demand) and
emits a fresh patient_state snapshot when something actually changes.

Decision matrix (existing active fact F, incoming candidate C):

    candidate matches F by (patient, type, canonical_name)?
      ├─ NO  → INSERT C as new active fact
      └─ YES
          └─ C.negated?
                ├─ YES → mark F status='resolved', insert C as 'refuted'-style record
                │        (audit row with status=refuted, contradicts_id=F.id)
                └─ NO
                    └─ C.value differs from F.value materially?
                          ├─ YES → mark F status='superseded',
                          │         insert C with supersedes_id=F.id
                          └─ NO  → bump F.observed_at + recompute confidence;
                                   do not insert a new row
"""

from __future__ import annotations

import logging
import uuid
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from memory.models.clinical_fact import (
    ClinicalFact,
    FactSource,
    FactStatus,
    FactType,
)
from memory.models.patient_state import PatientState
from memory.schemas.fact import ClinicalFactCandidate
from memory.schemas.state import PatientStateSnapshot

logger = logging.getLogger(__name__)


# Importance floor for facts that must never be trimmed during retrieval.
_SAFETY_CRITICAL_TYPES = {FactType.ALLERGY}


class ConsolidationService:
    """Merge fact candidates into the patient's longitudinal record."""

    async def consolidate(
        self,
        session: AsyncSession,
        patient_id: uuid.UUID,
        candidates: Iterable[ClinicalFactCandidate],
        *,
        source_event_id: uuid.UUID | None,
    ) -> list[ClinicalFact]:
        """
        Apply candidates against existing active facts. Returns the new /
        updated ClinicalFact rows (for downstream logging).

        Caller is responsible for the surrounding transaction.
        """
        written: list[ClinicalFact] = []

        for cand in candidates:
            # Allergies are pinned to importance=1.0 regardless of LLM scoring.
            if cand.fact_type in _SAFETY_CRITICAL_TYPES:
                cand = cand.model_copy(update={"importance": 1.0})

            existing = await self._find_active_match(session, patient_id, cand)

            if existing is None:
                written.append(
                    await self._insert_new(session, patient_id, cand, source_event_id)
                )
                continue

            if cand.negated:
                written.append(
                    await self._mark_resolved(session, existing, cand, source_event_id)
                )
                continue

            if self._value_materially_differs(existing.value, cand.value):
                written.append(
                    await self._supersede(session, existing, cand, patient_id, source_event_id)
                )
                continue

            # Same fact, same value → just bump observed_at + recompute confidence.
            existing.observed_at = _now()
            existing.confidence = max(existing.confidence, cand.confidence)
            session.add(existing)
            written.append(existing)

        await session.flush()
        return written

    # ------------------------------------------------------------------
    # patient_state snapshot
    # ------------------------------------------------------------------

    async def recompute_snapshot(
        self,
        session: AsyncSession,
        patient_id: uuid.UUID,
    ) -> PatientStateSnapshot:
        """Rebuild the denormalized snapshot from currently-active facts."""
        active = (
            await session.execute(
                select(ClinicalFact)
                .where(
                    ClinicalFact.patient_id == patient_id,
                    ClinicalFact.status == FactStatus.ACTIVE,
                )
                .order_by(ClinicalFact.importance.desc(), ClinicalFact.observed_at.desc())
            )
        ).scalars().all()

        buckets: dict[str, list[dict]] = {
            "allergies": [], "medications": [], "symptoms": [],
            "conditions": [], "chronic_conditions": [], "unresolved_followups": [],
            "recent_lab_values": [], "lifestyle": [], "preferences": [],
        }
        for f in active:
            entry = {
                "id": str(f.id),
                "name": f.canonical_name,
                "value": f.value,
                "since": f.onset_at.isoformat() if f.onset_at else None,
                "importance": f.importance,
                "decay_score": f.decay_score,
            }
            match f.fact_type:
                case FactType.ALLERGY:    buckets["allergies"].append(entry)
                case FactType.MEDICATION: buckets["medications"].append(entry)
                case FactType.SYMPTOM:    buckets["symptoms"].append(entry)
                case FactType.CONDITION:  buckets["conditions"].append(entry)
                case FactType.LAB_VALUE:  buckets["recent_lab_values"].append(entry)
                case FactType.LIFESTYLE:  buckets["lifestyle"].append(entry)
                case FactType.PREFERENCE: buckets["preferences"].append(entry)

        risk = _derive_risk_level(active)

        state = await session.get(PatientState, patient_id)
        if state is None:
            state = PatientState(patient_id=patient_id)
        state.snapshot = buckets
        state.risk_level = risk
        state.version += 1
        state.last_consolidated_at = _now()
        session.add(state)
        await session.flush()

        return PatientStateSnapshot(
            patient_id=patient_id,
            version=state.version,
            last_consolidated_at=state.last_consolidated_at,
            risk_level=risk,
            summary_text=state.summary_text,
            allergies=buckets["allergies"],
            medications=buckets["medications"],
            symptoms=buckets["symptoms"],
            conditions=buckets["conditions"],
            chronic_conditions=buckets["chronic_conditions"],
            unresolved_followups=buckets["unresolved_followups"],
            recent_lab_values=buckets["recent_lab_values"],
            lifestyle=buckets["lifestyle"],
            preferences=buckets["preferences"],
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    async def _find_active_match(
        session: AsyncSession,
        patient_id: uuid.UUID,
        cand: ClinicalFactCandidate,
    ) -> ClinicalFact | None:
        stmt = (
            select(ClinicalFact)
            .where(
                ClinicalFact.patient_id == patient_id,
                ClinicalFact.status == FactStatus.ACTIVE,
                ClinicalFact.fact_type == cand.fact_type,
                ClinicalFact.canonical_name == cand.canonical_name.strip(),
            )
            .limit(1)
        )
        return (await session.execute(stmt)).scalar_one_or_none()

    @staticmethod
    async def _insert_new(
        session: AsyncSession,
        patient_id: uuid.UUID,
        cand: ClinicalFactCandidate,
        source_event_id: uuid.UUID | None,
    ) -> ClinicalFact:
        row = ClinicalFact(
            patient_id=patient_id,
            fact_type=cand.fact_type,
            canonical_name=cand.canonical_name.strip(),
            normalized_code=cand.normalized_code,
            value=dict(cand.value),
            onset_at=cand.onset_at,
            expires_at=cand.expires_at,
            status=FactStatus.ACTIVE,
            confidence=cand.confidence,
            importance=cand.importance,
            decay_score=1.0,
            source=FactSource.LLM_EXTRACTION,
            source_event_id=source_event_id,
        )
        session.add(row)
        await session.flush()
        return row

    @staticmethod
    async def _mark_resolved(
        session: AsyncSession,
        existing: ClinicalFact,
        cand: ClinicalFactCandidate,
        source_event_id: uuid.UUID | None,
    ) -> ClinicalFact:
        existing.status = FactStatus.RESOLVED
        existing.expires_at = _now()
        session.add(existing)

        # Audit row records the negation explicitly.
        audit = ClinicalFact(
            patient_id=existing.patient_id,
            fact_type=existing.fact_type,
            canonical_name=existing.canonical_name,
            normalized_code=existing.normalized_code,
            value={"negated_by_patient": True, **(cand.value or {})},
            status=FactStatus.REFUTED,
            confidence=cand.confidence,
            importance=existing.importance,
            decay_score=1.0,
            source=FactSource.LLM_EXTRACTION,
            source_event_id=source_event_id,
            contradicts_id=existing.id,
        )
        session.add(audit)
        await session.flush()
        return audit

    @staticmethod
    async def _supersede(
        session: AsyncSession,
        existing: ClinicalFact,
        cand: ClinicalFactCandidate,
        patient_id: uuid.UUID,
        source_event_id: uuid.UUID | None,
    ) -> ClinicalFact:
        existing.status = FactStatus.SUPERSEDED
        session.add(existing)

        new = ClinicalFact(
            patient_id=patient_id,
            fact_type=cand.fact_type,
            canonical_name=cand.canonical_name.strip(),
            normalized_code=cand.normalized_code,
            value=dict(cand.value),
            onset_at=cand.onset_at,
            expires_at=cand.expires_at,
            status=FactStatus.ACTIVE,
            confidence=cand.confidence,
            importance=cand.importance,
            decay_score=1.0,
            source=FactSource.LLM_EXTRACTION,
            source_event_id=source_event_id,
            supersedes_id=existing.id,
        )
        session.add(new)
        await session.flush()
        return new

    @staticmethod
    def _value_materially_differs(old: dict, new: dict) -> bool:
        """
        Are the two value JSONBs meaningfully different?
        For medications, a dose / frequency / route change is material.
        For symptoms, a severity change is material.
        Heuristic; refine per fact_type as production needs emerge.
        """
        if not new:
            return False
        material_keys = ("dose", "unit", "frequency", "route", "severity", "value")
        for k in material_keys:
            if k in new and old.get(k) != new.get(k):
                return True
        return False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _now():
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc)


def _derive_risk_level(active_facts: list[ClinicalFact]) -> str:
    """Pure-Python summary of risk derived from currently-active facts."""
    # Placeholder heuristic — production should use a proper rules engine
    # driven by SafetyService output.
    for f in active_facts:
        if f.fact_type == FactType.SYMPTOM and f.value.get("severity") == "severe":
            return "high"
    return "none"
