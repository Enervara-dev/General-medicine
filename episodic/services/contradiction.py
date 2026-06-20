"""
ContradictionService — detects clinical contradictions between a new claim
(query text or candidate episode) and the patient's prior episodic memory.

Strategy:
  1. Narrow the candidate set: pull the top-K episodes most semantically
     similar to the new claim (already what RetrieverService does).
  2. Hand those + the new claim to a small Gemini model in JSON mode.
  3. The model returns explicit (prior_episode_id, reason, severity) tuples.

The service NEVER stores anything itself; the caller decides whether to
trigger a clarification or apply a confidence penalty.
"""

from __future__ import annotations

import json
import logging
from typing import Iterable

from graphrag.llm.gemini_client import generate_text_async

from episodic.config import EpisodicConfig
from episodic.prompts.contradiction import CONTRADICTION_SYSTEM_PROMPT
from episodic.schemas.contradiction import (
    Contradiction,
    ContradictionReport,
    ContradictionSeverity,
)
from episodic.schemas.episode import Episode

logger = logging.getLogger(__name__)


# Severity → confidence penalty applied to the new candidate.
_SEVERITY_PENALTIES = {
    ContradictionSeverity.INFO: 0.05,
    ContradictionSeverity.WARNING: 0.15,
    ContradictionSeverity.CRITICAL: 0.35,
}


class ContradictionService:
    def __init__(self, model: str | None = None) -> None:
        self._model = model or EpisodicConfig.CONTRADICTION_MODEL

    async def detect(
        self,
        *,
        user_id: str,
        new_claim: str,
        prior_episodes: Iterable[Episode],
    ) -> ContradictionReport:
        priors = list(prior_episodes)
        if not priors:
            return ContradictionReport(user_id=user_id, has_contradictions=False)

        user_prompt = (
            f"user_id: {user_id}\n\n"
            f"new_claim: {new_claim.strip()}\n\n"
            "prior_episodes (most relevant first):\n"
            + json.dumps(
                [_summarize_prior(e) for e in priors],
                default=str,
            )
        )

        try:
            content = await generate_text_async(
                user_prompt,
                model=self._model,
                system_instruction=CONTRADICTION_SYSTEM_PROMPT,
                temperature=0,
                json_mode=True,
            )
        except Exception as exc:
            logger.exception("Contradiction LLM call failed: %s", exc)
            return ContradictionReport(user_id=user_id, has_contradictions=False)

        if not content:
            return ContradictionReport(user_id=user_id, has_contradictions=False)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Contradiction service returned non-JSON content.")
            return ContradictionReport(user_id=user_id, has_contradictions=False)

        raw_list = parsed.get("contradictions") or []
        contradictions: list[Contradiction] = []
        for raw in raw_list:
            try:
                contradictions.append(Contradiction.model_validate(raw))
            except Exception:
                logger.debug("Skipping malformed contradiction row: %s", raw)

        has_any = bool(parsed.get("has_contradictions")) and bool(contradictions)
        penalty = _aggregate_penalty(contradictions) if has_any else 0.0
        triggers = any(
            c.severity in (ContradictionSeverity.WARNING, ContradictionSeverity.CRITICAL)
            for c in contradictions
        )
        return ContradictionReport(
            user_id=user_id,
            has_contradictions=has_any,
            contradictions=contradictions if has_any else [],
            confidence_penalty=penalty,
            triggers_clarification=triggers,
        )


def _summarize_prior(ep: Episode) -> dict:
    return {
        "prior_episode_id": str(ep.episode_id),
        "timestamp": ep.timestamp.isoformat(),
        "category": ep.category.value,
        "summary": ep.summary,
        "entities": ep.entities.model_dump(),
        "severity": ep.severity.value,
    }


def _aggregate_penalty(contradictions: list[Contradiction]) -> float:
    """Sum severity-weighted penalties, clamped at 0.6 (never zero out a fact entirely)."""
    total = sum(_SEVERITY_PENALTIES.get(c.severity, 0.1) for c in contradictions)
    return min(0.6, total)
