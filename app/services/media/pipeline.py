"""
MediaPipeline — orchestrates one upload end to end.

    validate → classify → route → process → store → assemble attachment

Each stage is a separate, injectable collaborator (validation, classifier,
storage, processors) so this class only owns sequencing, not policy. The output
is a ``MediaResult`` the orchestrator folds into a normal turn; raw bytes never
leave this layer except into ``MediaStorage``.
"""

from __future__ import annotations

import logging

from graphrag.config.settings import Settings

from app.services.media.classifier import ImageClassifier
from app.services.media.processors import (
    DocumentProcessor,
    MediaProcessor,
    PhotoProcessor,
)
from app.services.media.router import route_for_category
from app.services.media.storage import LocalMediaStorage, MediaStorage
from app.services.media.types import (
    MediaAttachment,
    MediaResult,
    ProcessingRoute,
)
from app.services.media.validation import validate_image

logger = logging.getLogger(__name__)


class MediaPipeline:
    """Process an uploaded image into a turn attachment."""

    def __init__(
        self,
        *,
        settings: Settings,
        storage: MediaStorage,
        classifier: ImageClassifier,
        processors: dict[ProcessingRoute, MediaProcessor],
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._classifier = classifier
        self._processors = processors

    @classmethod
    def from_settings(cls, settings: Settings) -> "MediaPipeline":
        """Default wiring used by the app container."""
        return cls(
            settings=settings,
            storage=LocalMediaStorage(settings.MEDIA_STORAGE_DIR),
            classifier=ImageClassifier(),
            processors={
                ProcessingRoute.MULTIMODAL_LLM: PhotoProcessor(),
                ProcessingRoute.DOCUMENT_EXTRACTION: DocumentProcessor(),
            },
        )

    async def process(
        self,
        *,
        data: bytes,
        mime_type: str | None,
        filename: str | None,
        query: str | None,
    ) -> MediaResult:
        """Run the full pipeline. Raises ``MediaValidationError`` on bad input."""
        image = validate_image(
            data=data,
            mime_type=mime_type,
            filename=filename,
            max_bytes=self._settings.MEDIA_MAX_UPLOAD_BYTES,
            allowed_mime_types=self._settings.MEDIA_ALLOWED_MIME_TYPES,
        )

        classification = await self._classifier.classify(image)
        route = route_for_category(classification.category)
        processor = self._processors[route]
        artifact = await processor.process(image, classification)

        # Persist bytes (storage owns them); record only the URI in metadata.
        try:
            artifact.storage_uri = await self._storage.save(
                data=image.data, mime_type=image.mime_type
            )
        except Exception as exc:  # noqa: BLE001 — answering shouldn't fail on storage
            logger.warning("Media storage failed (continuing without URI): %s", exc)

        # Only the photo route forwards the raw image to the answer LLM; the
        # document route is grounded purely on extracted text.
        parts = [image.as_part()] if route == ProcessingRoute.MULTIMODAL_LLM else []

        attachment = MediaAttachment(
            parts=parts,
            context_text=artifact.to_context_block(),
            memory_note=artifact.to_memory_note(),
            artifact=artifact,
        )
        return MediaResult(
            attachment=attachment,
            effective_query=self._effective_query(query, route),
        )

    @staticmethod
    def _effective_query(query: str | None, route: ProcessingRoute) -> str:
        text = (query or "").strip()
        if text:
            return text
        if route == ProcessingRoute.DOCUMENT_EXTRACTION:
            return "I've uploaded a medical document. Please review it and explain what it means."
        return "I've uploaded an image. What can you tell me about it?"


__all__ = ["MediaPipeline"]
