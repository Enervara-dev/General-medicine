"""
SpecialtyConfig — the typed shape of everything that used to be a hardcoded
module-level constant in one of the 7 standalone services (General-medicine
plus the 6 clones: Cardiology, Dermatology, ENT, Ophthalmology, Orthopaedics,
RAG-pulmonology / "pulmonology").

Per the unification audit (AUDIT_REPORT.md §2), specialty coupling in every
one of those 7 services lives almost entirely in prompt/text content, not in
the retrieval or storage layer. This dataclass is the parameterization
surface for that content: one instance per specialty, selected per request,
instead of one Python module per specialty service.

Fields are grouped by what they replace:
    persona / gatekeeper_system_prompt   -> the hardcoded identity text in
        prompt_layers.layer_core_identity() (GM) / graphrag/domain/answer_prompt.py
        (clones), and the analyzer SYSTEM_PROMPT (GM) /
        graphrag/domain/prompts.py::GATEKEEPER_SYSTEM_PROMPT (clones).
    relevance_threshold                  -> e.g. CARDIOLOGY_RELEVANCE_THRESHOLD
    red_flag_patterns                    -> graphrag/domain/clinical_policy.py
    entity_priority_types                -> graphrag/domain/entity_rules.py
    pinecone_namespace                   -> graphrag/domain/vocabulary.py
        (NOT used to filter retrieval yet — see graphrag/retrieval/interface.py.
        Pinecone behaviour is deliberately left unchanged in this phase; this
        field only documents what a future specialty-aware retriever would key
        on, per AUDIT_REPORT.md §3.)
    source_service                       -> app/services/pms/producer.py's
        SPECIALTY_SERVICE constant (PMS SourceRef.service identity)
    entity_overrides / relation_type_by_target_type / canonical_entities
        -> ingestion-time data-quality maps, generalized from Orthopaedics'
        chunking/postprocessing/ (see chunking/postprocessing/postprocessor.py
        in this repo). Empty by default (no-op) for every specialty that
        doesn't have curated maps yet; this does NOT run against any live
        ingestion pipeline in this change — it's a reusable component only.

All fields except ``key`` and ``display_name`` have safe defaults so a new
specialty can be registered with just a persona + gatekeeper prompt and grow
the rest incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpecialtyConfig:
    # Stable identifier used on the wire (ChatRequest.specialty) and as the
    # registry lookup key. Snake_case, e.g. "general_medicine", "cardiology".
    key: str
    display_name: str

    # --- prompt layer -------------------------------------------------- #
    # Replaces prompt_layers.layer_core_identity()'s hardcoded persona text.
    persona: str
    # Full system prompt for the gatekeeper/query-analyzer LLM call. Replaces
    # analyzer.SYSTEM_PROMPT (GM) / GATEKEEPER_SYSTEM_PROMPT (clones). Each
    # specialty's prompt is used verbatim — specialties are NOT diffed/merged
    # against each other, preserving each one's existing clinical behavior.
    gatekeeper_system_prompt: str

    # --- gatekeeper / routing tuning ------------------------------------ #
    relevance_threshold: int = 75
    red_flag_patterns: tuple[str, ...] = ()
    entity_priority_types: tuple[str, ...] = ()

    # --- retrieval (informational only in this phase; see
    # graphrag/retrieval/interface.py — Pinecone/Neo4j behavior is unchanged) --
    pinecone_namespace: str | None = None

    # --- PMS identity ---------------------------------------------------- #
    # Mirrors the assertion's `azp`. Sent as PmsMemoryEventV1.source.service.
    # Default matches the pre-unification hardcoded value so a request that
    # omits `specialty` entirely produces byte-identical PMS events to before.
    source_service: str = "general-medicine"

    # --- ingestion-time data-quality maps (see chunking/postprocessing/) -- #
    # Empty dict = no-op for this specialty (preserves current behavior).
    entity_overrides: dict[str, str] = field(default_factory=dict)
    relation_type_by_target_type: dict[str, str] = field(default_factory=dict)
    canonical_entities: dict[str, str] = field(default_factory=dict)


__all__ = ["SpecialtyConfig"]
