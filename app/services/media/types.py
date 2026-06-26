"""
Core types for the media (image upload) layer.

These DTOs are the contracts between the upload → validation → classification →
routing → processing → memory stages. Keeping them in one place lets every stage
depend on shapes, not on each other, and makes adding a new modality (audio,
video) or a new image category a localised change.

Nothing here holds onto raw bytes beyond the in-flight ``UploadedImage`` /
``MediaPart``; what crosses into memory is ``MediaArtifact``, which is
metadata-only by construction (see objective 5: never store raw image bytes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field


class MediaModality(str, Enum):
    """Top-level kind of an attachment. Image-only today; the enum is the seam
    for AUDIO / VIDEO / etc. later."""

    IMAGE = "image"


class ImageCategory(str, Enum):
    """
    What an uploaded image actually is. Drives routing.

    Add a new category here + one line in ``router.DOCUMENT_CATEGORIES`` (if it
    is a document) and the rest of the pipeline picks it up.
    """

    CLINICAL_PHOTO = "clinical_photo"          # skin, wound, rash, swelling, etc.
    GENERAL_PHOTO = "general_photo"            # non-clinical everyday photo
    DOCUMENT = "document"                      # generic document/scan
    LAB_REPORT = "lab_report"                  # structured laboratory results
    RADIOLOGY_REPORT = "radiology_report"      # imaging report text (not the scan)
    OTHER_MEDICAL_DOCUMENT = "other_medical_document"  # discharge summary, Rx, etc.
    UNKNOWN = "unknown"


class ProcessingRoute(str, Enum):
    """How a classified image is handled downstream."""

    MULTIMODAL_LLM = "multimodal_llm"          # send the image to the vision LLM
    DOCUMENT_EXTRACTION = "document_extraction"  # OCR/structured text extraction


@dataclass(frozen=True)
class MediaPart:
    """
    A single non-text part bound for the multimodal LLM.

    Structurally matches ``graphrag.llm.gemini_client.MediaLike`` so it can be
    handed straight to the LLM client without coupling that boundary to this
    package.
    """

    data: bytes
    mime_type: str


@dataclass(frozen=True)
class UploadedImage:
    """A validated upload. Exists only for the duration of one request."""

    data: bytes
    mime_type: str
    size_bytes: int
    filename: str | None = None

    def as_part(self) -> MediaPart:
        return MediaPart(data=self.data, mime_type=self.mime_type)


@dataclass(frozen=True)
class ImageClassification:
    """Output of the classification stage."""

    category: ImageCategory
    confidence: float
    rationale: str = ""


class MediaArtifact(BaseModel):
    """
    Metadata-only record of a processed upload — the ONLY thing that reaches
    memory or the API response. Deliberately has no field for raw bytes.
    """

    modality: MediaModality = MediaModality.IMAGE
    category: ImageCategory
    route: ProcessingRoute
    mime_type: str
    size_bytes: int
    filename: str | None = None
    storage_uri: str | None = None
    caption: str | None = None
    extracted_facts: list[str] = Field(default_factory=list)

    def to_context_block(self) -> str:
        """Render the artifact as a prompt context block for the answer LLM."""
        lines = [f"=== UPLOADED {self.category.value.replace('_', ' ').upper()} ==="]
        if self.caption:
            lines.append(self.caption.strip())
        if self.extracted_facts:
            lines.append("Key findings:")
            lines.extend(f"- {f}" for f in self.extracted_facts)
        return "\n".join(lines).strip()

    def to_memory_note(self) -> str:
        """Compact one-block note stored with the conversation turn (no bytes)."""
        bits = [f"[Attached {self.category.value.replace('_', ' ')}]"]
        if self.caption:
            bits.append(self.caption.strip())
        if self.extracted_facts:
            bits.append("Findings: " + "; ".join(self.extracted_facts))
        return " ".join(bits).strip()


@dataclass
class MediaAttachment:
    """
    Everything the orchestrator needs to fold one upload into a turn.

    - ``parts``        raw media for the multimodal LLM (empty on the document route)
    - ``context_text`` prompt block injected into the answer call
    - ``memory_note``  metadata-only note appended to the stored turn
    - ``artifact``     metadata returned to the API caller
    """

    parts: list[MediaPart]
    context_text: str
    memory_note: str
    artifact: MediaArtifact


@dataclass
class MediaResult:
    """Result of running the media pipeline on one upload."""

    attachment: MediaAttachment
    effective_query: str


class MediaValidationError(ValueError):
    """Raised when an upload fails MIME/size/content validation."""


__all__ = [
    "MediaModality",
    "ImageCategory",
    "ProcessingRoute",
    "MediaPart",
    "UploadedImage",
    "ImageClassification",
    "MediaArtifact",
    "MediaAttachment",
    "MediaResult",
    "MediaValidationError",
]
