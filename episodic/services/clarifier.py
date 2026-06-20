"""
ClarifierService — decides whether ONE clarifying question is required.

Contract: returns at most one question per turn. The model is prompted to
honor this; the service enforces it again on the way out so a regression
in the prompt can never leak a multi-question response to the user.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from graphrag.llm.gemini_client import generate_text_async

from episodic.config import EpisodicConfig
from episodic.prompts.clarification import CLARIFICATION_SYSTEM_PROMPT
from episodic.schemas.clarification import (
    ClarificationQuestion,
    ClarificationResponse,
)
from episodic.schemas.episode import EpisodeCandidate

logger = logging.getLogger(__name__)


class ClarifierService:
    def __init__(self, model: str | None = None) -> None:
        self._model = model or EpisodicConfig.CLARIFICATION_MODEL

    async def evaluate(
        self,
        *,
        user_id: str,
        utterance: str,
        candidate: EpisodeCandidate | None = None,
        contradiction_hint: str | None = None,
    ) -> ClarificationResponse:
        user_prompt = self._build_user_prompt(
            user_id=user_id,
            utterance=utterance,
            candidate=candidate,
            contradiction_hint=contradiction_hint,
        )

        try:
            content = await generate_text_async(
                user_prompt,
                model=self._model,
                system_instruction=CLARIFICATION_SYSTEM_PROMPT,
                temperature=0,
                json_mode=True,
            )
        except Exception as exc:
            logger.exception("Clarifier LLM call failed: %s", exc)
            return ClarificationResponse(needs_clarification=False)

        if not content:
            return ClarificationResponse(needs_clarification=False)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Clarifier returned non-JSON content; skipping.")
            return ClarificationResponse(needs_clarification=False)

        # Enforce the at-most-one-question contract here, regardless of model output.
        raw_qs = (parsed.get("questions") or [])[: EpisodicConfig.MAX_CLARIFICATIONS_PER_TURN]
        questions: list[ClarificationQuestion] = []
        for raw in raw_qs:
            try:
                questions.append(ClarificationQuestion.model_validate(raw))
            except Exception:
                logger.debug("Skipping malformed clarification question: %s", raw)

        needs = bool(parsed.get("needs_clarification")) and bool(questions)
        return ClarificationResponse(
            needs_clarification=needs,
            questions=questions if needs else [],
        )

    @staticmethod
    def _build_user_prompt(
        *,
        user_id: str,
        utterance: str,
        candidate: EpisodeCandidate | None,
        contradiction_hint: str | None,
    ) -> str:
        parts: list[str] = [
            f"user_id: {user_id}",
            f"utterance: {utterance.strip()}",
        ]
        if candidate is not None:
            parts.append(
                "candidate_episode: "
                + json.dumps(candidate.model_dump(mode="json"), default=str)
            )
        if contradiction_hint:
            parts.append(f"contradiction_hint: {contradiction_hint}")
        return "\n\n".join(parts)
