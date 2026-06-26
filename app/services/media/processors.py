"""
Media processors — one per processing route, behind a shared interface.

- ``PhotoProcessor``      captions a photograph for clinical context. The raw
                          image is also forwarded to the answer LLM (the pipeline
                          attaches it), so this caption is supporting context, not
                          the whole interpretation.
- ``DocumentProcessor``   runs structured text extraction on reports/documents
                          (transcribe + pull key findings). This is the
                          "document extraction" path: it deliberately does NOT
                          ask the model to visually interpret the page, and the
                          raw image is NOT forwarded to the answer LLM.

Both return a metadata-only ``MediaArtifact``. The LLM call is injected so each
processor is unit-testable without the network and provider-swappable.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol, Sequence, runtime_checkable

from graphrag.config.settings import settings
from graphrag.llm.gemini_client import MediaLike, generate_text_async

from app.services.media.classifier import GenerateFn
from app.services.media.types import (
    ImageClassification,
    MediaArtifact,
    ProcessingRoute,
    UploadedImage,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class MediaProcessor(Protocol):
    """Turn a validated image + its classification into metadata."""

    route: ProcessingRoute

    async def process(
        self, image: UploadedImage, classification: ImageClassification
    ) -> MediaArtifact: ...


def _base_artifact(
    image: UploadedImage, classification: ImageClassification, route: ProcessingRoute
) -> MediaArtifact:
    return MediaArtifact(
        category=classification.category,
        route=route,
        mime_type=image.mime_type,
        size_bytes=image.size_bytes,
        filename=image.filename,
    )


class PhotoProcessor:
    """Captions clinical/general photographs via the multimodal LLM."""

    route = ProcessingRoute.MULTIMODAL_LLM

    _SYSTEM = (
        "You are a clinician describing an uploaded patient photo for triage. "
        "In 1-2 plain sentences, describe only what is visibly present (body "
        "region, visible features, colour, distribution). Do NOT diagnose, do "
        "NOT speculate beyond what is visible, and never invent detail. If the "
        "image is not a clinical photo, say briefly what it shows."
    )

    def __init__(self, *, model: str | None = None, generate: GenerateFn = generate_text_async) -> None:
        self._model = model or settings.VISION_MODEL
        self._generate = generate

    async def process(
        self, image: UploadedImage, classification: ImageClassification
    ) -> MediaArtifact:
        artifact = _base_artifact(image, classification, self.route)
        media: Sequence[MediaLike] = [image.as_part()]
        try:
            caption = await self._generate(
                "Describe this image for clinical context.",
                model=self._model,
                system_instruction=self._SYSTEM,
                temperature=0.2,
                media=media,
            )
            artifact.caption = (caption or "").strip() or None
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            logger.warning("Photo captioning failed: %s", exc)
        return artifact


class DocumentProcessor:
    """Extracts structured text/findings from report/document images (OCR path)."""

    route = ProcessingRoute.DOCUMENT_EXTRACTION

    _SYSTEM = (
        "You are a medical document extraction engine. Transcribe and extract "
        "the clinically relevant content from the uploaded document image. Output "
        "STRICT JSON only:\n"
        '{"summary": "<one-line description of the document>", '
        '"facts": ["<atomic finding, value with unit, or statement>", ...]}\n'
        "Rules: extract verbatim values (e.g. 'Hemoglobin 9.1 g/dL (low)'); do "
        "NOT diagnose or interpret beyond what is written; never invent values. "
        "If nothing is legible, return an empty facts list."
    )

    def __init__(self, *, model: str | None = None, generate: GenerateFn = generate_text_async) -> None:
        self._model = model or settings.VISION_MODEL
        self._generate = generate

    async def process(
        self, image: UploadedImage, classification: ImageClassification
    ) -> MediaArtifact:
        artifact = _base_artifact(image, classification, self.route)
        media: Sequence[MediaLike] = [image.as_part()]
        try:
            raw = await self._generate(
                "Extract the content of this medical document.",
                model=self._model,
                system_instruction=self._SYSTEM,
                temperature=0,
                json_mode=True,
                media=media,
            )
            summary, facts = self._parse(raw)
            artifact.caption = summary
            artifact.extracted_facts = facts
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            logger.warning("Document extraction failed: %s", exc)
        return artifact

    @staticmethod
    def _parse(raw: str) -> tuple[str | None, list[str]]:
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Document extractor returned non-JSON: %r", (raw or "")[:200])
            return None, []
        summary = str(obj.get("summary", "")).strip() or None
        facts_raw = obj.get("facts") or []
        facts = [str(f).strip() for f in facts_raw if str(f).strip()]
        return summary, facts


__all__ = ["MediaProcessor", "PhotoProcessor", "DocumentProcessor"]
