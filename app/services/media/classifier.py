"""
Image classification — the query-understanding stage for uploads.

Runs BEFORE any downstream processing so the pipeline can route reports away
from visual interpretation (objective 3). It's a thin, lightweight multimodal
call that returns one ``ImageCategory`` + confidence; the heavy lifting
(captioning / extraction) happens later, only on the chosen route.

The LLM call is injected (``generate``) so the classifier is unit-testable
without the network and so the provider can be swapped.
"""

from __future__ import annotations

import json
import logging
from typing import Awaitable, Callable, Optional, Sequence

from graphrag.config.settings import settings
from graphrag.llm.gemini_client import MediaLike, generate_text_async

from app.services.media.types import (
    ImageCategory,
    ImageClassification,
    UploadedImage,
)

logger = logging.getLogger(__name__)

# Async LLM call signature the classifier depends on (subset of generate_text_async).
GenerateFn = Callable[..., Awaitable[str]]

_CATEGORY_VALUES = ", ".join(c.value for c in ImageCategory if c is not ImageCategory.UNKNOWN)

_SYSTEM_PROMPT = f"""You are an image triage classifier for a medical assistant.
Look at the image and classify what KIND of thing it is. Output STRICT JSON only.

Choose exactly one `category` from:
* clinical_photo          - a photo of a body part for clinical assessment (skin rash, wound, swelling, eye, mouth, nail).
* general_photo           - an everyday non-clinical photo (food, scenery, object, pet, screenshot).
* lab_report             - a laboratory results document/printout (blood tests, panels, reference ranges, numeric values).
* radiology_report       - a written radiology/imaging REPORT (X-ray/CT/MRI/ultrasound findings as text).
* document               - a generic document/scan that is not clearly one of the medical report types.
* other_medical_document  - a medical document that is not a lab or radiology report (prescription, discharge summary, referral).

CRITICAL RULES:
* A lab report, radiology report, prescription, or any page of structured medical
  TEXT is a DOCUMENT, never a clinical_photo — even if photographed with a phone.
  Classify by CONTENT (text/tables/values) not by how it was captured.
* Only use clinical_photo for an actual photograph of a body/anatomy for visual assessment.
* When unsure between a document type and a photo, prefer the document type if the
  image is dominated by printed text, tables, or numeric values.

Output JSON shape (no prose, no markdown):
{{"category": "<one of: {_CATEGORY_VALUES}>", "confidence": 0-100, "rationale": "<short reason>"}}
"""


class ImageClassifier:
    """Classifies an ``UploadedImage`` into an ``ImageCategory``."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        generate: GenerateFn = generate_text_async,
    ) -> None:
        self._model = model or settings.VISION_MODEL
        self._generate = generate

    async def classify(self, image: UploadedImage) -> ImageClassification:
        media: Sequence[MediaLike] = [image.as_part()]
        try:
            raw = await self._generate(
                "Classify this uploaded image.",
                model=self._model,
                system_instruction=_SYSTEM_PROMPT,
                temperature=0,
                json_mode=True,
                media=media,
            )
            return self._parse(raw)
        except Exception as exc:  # noqa: BLE001 — classification must never crash the upload
            logger.warning("Image classification failed, defaulting to UNKNOWN: %s", exc)
            return ImageClassification(
                category=ImageCategory.UNKNOWN, confidence=0.0, rationale="classifier error"
            )

    @staticmethod
    def _parse(raw: str) -> ImageClassification:
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Classifier returned non-JSON: %r", (raw or "")[:200])
            return ImageClassification(ImageCategory.UNKNOWN, 0.0, "unparseable response")

        try:
            category = ImageCategory(str(obj.get("category", "")).strip().lower())
        except ValueError:
            category = ImageCategory.UNKNOWN

        raw_conf = obj.get("confidence", 0)
        try:
            conf = float(raw_conf)
        except (TypeError, ValueError):
            conf = 0.0
        if conf > 1.0:  # model emitted 0-100
            conf /= 100.0
        conf = max(0.0, min(1.0, conf))

        return ImageClassification(
            category=category,
            confidence=conf,
            rationale=str(obj.get("rationale", "")).strip(),
        )


__all__ = ["ImageClassifier", "GenerateFn"]
