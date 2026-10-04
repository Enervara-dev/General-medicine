"""
general_medicine — the pre-existing, already-live GM behavior.

Persona and gatekeeper prompt are imported DIRECTLY from the modules that
have always defined them (``prompt_layers.layer_core_identity()``'s default
and ``analyzer.SYSTEM_PROMPT``) rather than duplicated as a hand-copied
string here — a hand-copied duplicate can silently drift from the live
source (e.g. an incidental whitespace difference during transcription, which
is exactly what happened on the first pass here and was caught by
tests/unit/test_specialty_registry.py's equality assertion) with no
guarantee of staying in sync afterward. Importing makes the two permanently,
structurally identical instead of relying on a one-time copy.

This is the DEFAULT_SPECIALTY fallback, so a request that omits ``specialty``
entirely resolves to this config and produces byte-identical prompts to the
pre-unification behavior, by construction.
"""

from __future__ import annotations

from app.services.orchestration.prompt_layers import layer_core_identity
from app.specialty.models import SpecialtyConfig
from graphrag.query_understanding.analyzer import SYSTEM_PROMPT as _ANALYZER_SYSTEM_PROMPT

PERSONA = layer_core_identity()
GATEKEEPER_SYSTEM_PROMPT = _ANALYZER_SYSTEM_PROMPT

CONFIG = SpecialtyConfig(
    key="general_medicine",
    display_name="General Medicine",
    persona=PERSONA,
    gatekeeper_system_prompt=GATEKEEPER_SYSTEM_PROMPT,
    relevance_threshold=0,  # GM has no relevance gate upstream of the analyzer
    red_flag_patterns=(),  # GM's red-flag detection lives inside the gatekeeper prompt above
    entity_priority_types=(),  # GM's priority types are per-QueryType, not per-specialty (query_config.py)
    pinecone_namespace=None,  # GM's existing index has no namespace (AUDIT_REPORT.md §3)
    source_service="general-medicine",
)

__all__ = ["CONFIG", "PERSONA", "GATEKEEPER_SYSTEM_PROMPT"]
