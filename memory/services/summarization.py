"""
SummarizationService — generate the "Current Patient State" prose paragraph.

Unlike the legacy summarizer (which compressed turns), this service generates
prose from the *structured snapshot*. Re-runs only when consolidation reports
a material change to avoid burning LLM calls on every turn.

Cheap fallback: a deterministic template renderer. The LLM path is opt-in
and triggered only on a meaningful diff between old and new snapshots.
"""

from __future__ import annotations

import json
import logging
from typing import Iterable

from graphrag.config.settings import settings
from graphrag.llm.gemini_client import DEFAULT_LITE_MODEL, generate_text_async
from memory.schemas.state import PatientStateSnapshot

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = getattr(settings, "SUMMARIZATION_MODEL", None) or DEFAULT_LITE_MODEL

_SYSTEM_PROMPT = """You write the 'Current Patient State' paragraph for a
medical assistant. Two short paragraphs maximum. No lists, no headings.

Strictly factual. Mention allergies first (always), then active medications,
then current symptoms and their duration, then chronic conditions. Skip empty
categories. Do not diagnose, recommend, or speculate. Do not pad. If the
patient has no facts, return: 'No clinical history on file yet.'"""


class SummarizationService:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        timeout_s: float = 15.0,
    ) -> None:
        self._api_key = api_key or settings.GEMINI_API_KEY
        self._model = model
        self._timeout = timeout_s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def regenerate(self, snapshot: PatientStateSnapshot) -> str:
        """LLM-based prose generation. Use when accuracy matters."""
        if snapshot.is_empty():
            return "No clinical history on file yet."
        if not self._api_key:
            return self.render_template(snapshot)

        try:
            content = await generate_text_async(
                self._snapshot_to_prompt(snapshot),
                model=self._model,
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.2,
            )
            return (content or "").strip() or self.render_template(snapshot)
        except Exception as exc:
            logger.warning("Summarization LLM failed (%s); using template fallback.", exc)
            return self.render_template(snapshot)

    @staticmethod
    def render_template(snapshot: PatientStateSnapshot) -> str:
        """Deterministic fallback — no LLM needed."""
        parts: list[str] = []
        if snapshot.allergies:
            parts.append("Allergies: " + _names(snapshot.allergies) + ".")
        if snapshot.medications:
            parts.append("Active medications: " + _names(snapshot.medications) + ".")
        if snapshot.symptoms:
            parts.append("Current symptoms: " + _names(snapshot.symptoms) + ".")
        if snapshot.conditions:
            parts.append("Conditions: " + _names(snapshot.conditions) + ".")
        return " ".join(parts) or "No clinical history on file yet."

    @staticmethod
    def _snapshot_to_prompt(snapshot: PatientStateSnapshot) -> str:
        return json.dumps(
            snapshot.model_dump(mode="json", exclude={"summary_text"}),
            indent=2,
        )


def _names(entries: Iterable[dict]) -> str:
    return ", ".join(e["name"] for e in entries if e.get("name"))
