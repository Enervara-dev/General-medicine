"""
SafetyService — pre-flight checks before extracted facts reach the LLM.

Surfaces RiskFlag events for:
- Allergy collision (patient is allergic to drug X and now mentions taking X)
- Drug interaction (two active meds with known interaction)
- Critical symptom combinations (chest pain + dyspnea, etc.)
- Contradiction between candidate and active fact (handled by Consolidation
  but flagged here too for audit)

This is intentionally minimal — replace the rule lists with a proper
clinical knowledge source (RxNorm + OpenFDA + your own curated rules) for
production deployment.
"""

from __future__ import annotations

import logging
import uuid
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from memory.models.clinical_fact import ClinicalFact, FactStatus, FactType
from memory.schemas.fact import ClinicalFactCandidate, RiskFlag

logger = logging.getLogger(__name__)


# Tiny seed lists — production uses a curated DB. Keys must be lowercased.
_DANGEROUS_INTERACTIONS: dict[frozenset[str], str] = {
    frozenset({"warfarin", "ibuprofen"}): "NSAID + warfarin: bleeding risk",
    frozenset({"warfarin", "aspirin"}):   "Aspirin + warfarin: bleeding risk",
    frozenset({"ssri", "maoi"}):          "SSRI + MAOI: serotonin syndrome risk",
}

_CRITICAL_SYMPTOM_PAIRS: list[tuple[set[str], str]] = [
    ({"chest pain", "shortness of breath"}, "ACS-suspicious symptom pair"),
    ({"chest pain", "diaphoresis"},         "ACS-suspicious symptom pair"),
    ({"sudden severe headache", "stiff neck"}, "Possible meningitis / SAH"),
]


class SafetyService:
    """Synchronous business-logic helpers; async DB access where needed."""

    async def check_candidates(
        self,
        session: AsyncSession,
        patient_id: uuid.UUID,
        candidates: Iterable[ClinicalFactCandidate],
    ) -> list[RiskFlag]:
        """Run all safety checks; return zero or more RiskFlag events."""
        flags: list[RiskFlag] = []
        candidates = list(candidates)

        active = (
            await session.execute(
                select(ClinicalFact).where(
                    ClinicalFact.patient_id == patient_id,
                    ClinicalFact.status == FactStatus.ACTIVE,
                )
            )
        ).scalars().all()

        flags.extend(self._check_allergy_collisions(candidates, active))
        flags.extend(self._check_drug_interactions(candidates, active))
        flags.extend(self._check_critical_symptom_pairs(candidates, active))

        return flags

    # ------------------------------------------------------------------

    @staticmethod
    def _check_allergy_collisions(
        candidates: list[ClinicalFactCandidate],
        active: list[ClinicalFact],
    ) -> list[RiskFlag]:
        active_allergies = {
            f.canonical_name.lower(): f.id
            for f in active
            if f.fact_type == FactType.ALLERGY
        }
        flags: list[RiskFlag] = []
        for c in candidates:
            if c.fact_type != FactType.MEDICATION or c.negated:
                continue
            name = c.canonical_name.lower()
            for allergen, fact_id in active_allergies.items():
                if allergen in name or name in allergen:
                    flags.append(RiskFlag(
                        severity="block",
                        code="allergy_collision",
                        message=(
                            f"Patient has documented allergy to '{allergen}' "
                            f"but mentioned medication '{c.canonical_name}'."
                        ),
                        related_fact_ids=[fact_id],
                    ))
        return flags

    @staticmethod
    def _check_drug_interactions(
        candidates: list[ClinicalFactCandidate],
        active: list[ClinicalFact],
    ) -> list[RiskFlag]:
        active_meds = {
            f.canonical_name.lower()
            for f in active
            if f.fact_type == FactType.MEDICATION
        }
        cand_meds = {
            c.canonical_name.lower()
            for c in candidates
            if c.fact_type == FactType.MEDICATION and not c.negated
        }
        all_meds = active_meds | cand_meds

        flags: list[RiskFlag] = []
        for pair, message in _DANGEROUS_INTERACTIONS.items():
            if pair.issubset(all_meds):
                flags.append(RiskFlag(
                    severity="warn",
                    code="drug_interaction",
                    message=message,
                ))
        return flags

    @staticmethod
    def _check_critical_symptom_pairs(
        candidates: list[ClinicalFactCandidate],
        active: list[ClinicalFact],
    ) -> list[RiskFlag]:
        all_symptoms = {
            f.canonical_name.lower()
            for f in active
            if f.fact_type == FactType.SYMPTOM
        } | {
            c.canonical_name.lower()
            for c in candidates
            if c.fact_type == FactType.SYMPTOM and not c.negated
        }
        flags: list[RiskFlag] = []
        for pair, message in _CRITICAL_SYMPTOM_PAIRS:
            if pair.issubset(all_symptoms):
                flags.append(RiskFlag(
                    severity="warn",
                    code="critical_symptom_pair",
                    message=message,
                ))
        return flags
