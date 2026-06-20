"""
ExtractorService — utterance → EpisodeCandidate via Gemini.

Returns None when the model decides the utterance has no clinical content
(store_memory=false). The caller treats None as "drop, do not store".
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from graphrag.llm.gemini_client import generate_text_async

from episodic.config import EpisodicConfig
from episodic.prompts.extraction import EXTRACTION_SYSTEM_PROMPT
from episodic.schemas.episode import EpisodeCandidate

logger = logging.getLogger(__name__)


class ExtractorService:
    def __init__(self, model: str | None = None) -> None:
        self._model = model or EpisodicConfig.EXTRACTION_MODEL

    async def extract(
        self,
        *,
        user_id: str,
        utterance: str,
    ) -> Optional[EpisodeCandidate]:
        if not utterance or not utterance.strip():
            return None

        user_prompt = (
            f"user_id: {user_id}\n\n"
            f"utterance:\n{utterance.strip()}"
        )
        try:
            content = await generate_text_async(
                user_prompt,
                model=self._model,
                system_instruction=EXTRACTION_SYSTEM_PROMPT,
                temperature=0,
                json_mode=True,
            )
        except Exception as exc:
            logger.exception("Episodic extraction LLM call failed: %s", exc)
            return None

        if not content:
            return None

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.warning("Extractor returned malformed JSON: %s", exc)
            return None

        # Pin user_id from input — model output is hint, not authority.
        parsed["user_id"] = user_id

        try:
            candidate = EpisodeCandidate.model_validate(parsed)
        except Exception as exc:
            logger.warning("Extractor output failed validation: %s", exc)
            return None

        if not candidate.store_memory:
            return None
        if candidate.confidence < 0.6:
            return None

        return candidate
