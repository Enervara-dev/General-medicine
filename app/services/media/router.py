"""
Routing: image category -> processing route.

A single explicit table so the policy is reviewable at a glance and a new
category is one line. Documents/reports bypass visual interpretation and go to
structured extraction (objective 4); photographs go to the multimodal LLM.
"""

from __future__ import annotations

from app.services.media.types import ImageCategory, ProcessingRoute

# Categories that are structured medical/text documents — extract, don't "look at".
DOCUMENT_CATEGORIES: frozenset[ImageCategory] = frozenset(
    {
        ImageCategory.DOCUMENT,
        ImageCategory.LAB_REPORT,
        ImageCategory.RADIOLOGY_REPORT,
        ImageCategory.OTHER_MEDICAL_DOCUMENT,
    }
)


def route_for_category(category: ImageCategory) -> ProcessingRoute:
    """
    Map a classified category to its processing route.

    Documents → DOCUMENT_EXTRACTION. Everything else (clinical/general photos,
    and UNKNOWN — treated as a photo so we still attempt a helpful answer) →
    MULTIMODAL_LLM.
    """
    if category in DOCUMENT_CATEGORIES:
        return ProcessingRoute.DOCUMENT_EXTRACTION
    return ProcessingRoute.MULTIMODAL_LLM


__all__ = ["DOCUMENT_CATEGORIES", "route_for_category"]
