"""
Media (image upload) layer.

A self-contained, modular pipeline that turns an uploaded file into a turn
attachment the orchestrator can fold into the existing text flow:

    validate → classify → route → process → store → attachment

Public surface:
    MediaPipeline           — entry point (``from_settings`` for default wiring)
    MediaResult / MediaAttachment / MediaArtifact — outputs
    MediaValidationError    — validation failure (endpoint maps to HTTP 400/413)
    ImageCategory / ProcessingRoute — classification + routing enums
"""

from app.services.media.pipeline import MediaPipeline
from app.services.media.types import (
    ImageCategory,
    MediaArtifact,
    MediaAttachment,
    MediaResult,
    MediaValidationError,
    ProcessingRoute,
)

__all__ = [
    "MediaPipeline",
    "MediaResult",
    "MediaAttachment",
    "MediaArtifact",
    "MediaValidationError",
    "ImageCategory",
    "ProcessingRoute",
]
