"""
Specialty registry — the single place that turns "which of the 7 clinical
specialties is this request for" into the config values that used to be
hardcoded per standalone service (persona text, gatekeeper system prompt,
relevance threshold, red-flag patterns, entity-priority types, PMS service
identity, and optional ingestion-time data-quality maps).

Public surface:
    SpecialtyConfig   — typed, immutable config for one specialty.
    SpecialtyRegistry  — lookup table; falls back to "general_medicine".
    get_registry()     — the process-wide default registry (lazy singleton).
    DEFAULT_SPECIALTY  — "general_medicine", used when a request omits the
                          field entirely (preserves pre-unification behaviour).
"""

from __future__ import annotations

from app.specialty.models import SpecialtyConfig
from app.specialty.registry import (
    DEFAULT_SPECIALTY,
    SPECIALTY_KEYS,
    SpecialtyRegistry,
    build_default_registry,
    get_registry,
)

__all__ = [
    "DEFAULT_SPECIALTY",
    "SPECIALTY_KEYS",
    "SpecialtyConfig",
    "SpecialtyRegistry",
    "build_default_registry",
    "get_registry",
]
