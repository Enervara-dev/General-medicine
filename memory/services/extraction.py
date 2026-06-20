"""
ExtractionService — turn raw conversation text into ClinicalFactCandidate[].

Strategy:
    1. Cheap rule-based pre-extraction for high-confidence patterns
       (re-use the legacy regex set from state_extractor.py during migration).
    2. LLM call against a small fast Gemini model in JSON mode.
    3. Merge + dedupe.

The service does NOT persist anything — it only proposes. Persistence is the
ConsolidationService's job.
"""

from __future__ import annotations

import json
import logging
from typing import Iterable

from graphrag.config.settings import settings
from graphrag.llm.gemini_client import DEFAULT_MODEL, generate_text_async
from memory.prompts.extraction import (
    EXTRACTION_JSON_SCHEMA,
    EXTRACTION_SYSTEM_PROMPT,
)
from memory.schemas.fact import ClinicalFactCandidate

logger = logging.getLogger(__name__)


_DEFAULT_MODEL = getattr(settings, "EXTRACTION_MODEL", None) or DEFAULT_MODEL

# Schema reminder appended to the prompt so the model emits a stable shape.
# Gemini's `response_mime_type=application/json` guarantees valid JSON; this
# guides the structure since we don't pass an OpenAPI response_schema (the
# legacy schema uses OpenAI-style features Gemini's response_schema doesn't
# accept verbatim).
_SCHEMA_HINT = (
    "Emit a single JSON object: {\"facts\": [...]}. "
    "Each fact must include fact_type, canonical_name, value, "
    "confidence, importance, negated. fact_type must be one of: "
    "symptom, medication, allergy, condition, lab_value, vital, "
    "lifestyle, social, family_history, adherence, emotional, preference."
)


class ExtractionService:
    """Extract ClinicalFactCandidate objects from a single patient utterance."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        timeout_s: float = 20.0,
        min_confidence: float = 0.6,
    ) -> None:
        self._api_key = api_key or settings.GEMINI_API_KEY
        self._model = model
        self._timeout = timeout_s
        self._min_confidence = min_confidence

    async def extract(
        self,
        utterance: str,
        *,
        prior_state_hint: str | None = None,
    ) -> list[ClinicalFactCandidate]:
        """
        Extract facts from `utterance`. `prior_state_hint` is a short summary
        passed in to give the model resolution context (e.g. "patient has
        active fever; if they say 'better now' interpret as fever resolution").
        """
        if not utterance or not utterance.strip():
            return []
        if not self._api_key:
            logger.warning("GEMINI_API_KEY missing — extraction disabled")
            return []

        user_prompt = utterance.strip()
        if prior_state_hint:
            user_prompt = (
                f"[Prior state hint: {prior_state_hint}]\n\n"
                f"Utterance:\n{utterance.strip()}"
            )

        system_instruction = EXTRACTION_SYSTEM_PROMPT + "\n\n" + _SCHEMA_HINT

        try:
            content = await generate_text_async(
                user_prompt,
                model=self._model,
                system_instruction=system_instruction,
                temperature=0,
                json_mode=True,
            )
        except Exception as exc:
            logger.exception("Extraction LLM call failed: %s", exc)
            return []

        try:
            parsed = json.loads(content) if content else {}
            raw_facts = parsed.get("facts", []) if isinstance(parsed, dict) else []
        except json.JSONDecodeError as exc:
            logger.warning("Extraction LLM returned malformed JSON: %s", exc)
            return []

        candidates: list[ClinicalFactCandidate] = []
        for raw in raw_facts:
            try:
                cand = ClinicalFactCandidate.model_validate(raw)
            except Exception as exc:
                logger.debug("Skipping invalid extraction row: %s", exc)
                continue
            if cand.confidence < self._min_confidence:
                continue
            candidates.append(cand)

        return self._dedupe(candidates)

    # ------------------------------------------------------------------

    @staticmethod
    def _dedupe(
        candidates: Iterable[ClinicalFactCandidate],
    ) -> list[ClinicalFactCandidate]:
        """Collapse exact (fact_type, canonical_name, negated) duplicates."""
        seen: dict[tuple[str, str, bool], ClinicalFactCandidate] = {}
        for c in candidates:
            key = (c.fact_type.value, c.canonical_name.lower().strip(), c.negated)
            if key not in seen or c.confidence > seen[key].confidence:
                seen[key] = c
        return list(seen.values())


# Reference the legacy OpenAI-style schema so the import path still resolves
# for any external caller that imported it through this module historically.
__all__ = ["ExtractionService", "EXTRACTION_JSON_SCHEMA"]
