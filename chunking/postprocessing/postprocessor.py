"""
Generic ingestion-time data-quality pipeline — entity override -> relation
repair -> alias canonicalization.

This is a generalization of Orthopaedics/chunking/postprocessing/
{postprocessor,entity_overrides,relation_repair,canonicalization}.py (see
AUDIT_REPORT.md §7): the ONLY one of the 6 specialty clones that had this
3-stage data-quality pass, and it was never backported to the other 5 or to
General-medicine. The mechanism here is specialty-agnostic — the curated
maps live on SpecialtyConfig (``entity_overrides``,
``relation_type_by_target_type``, ``canonical_entities``), empty by default
(no-op) for every specialty that doesn't have them yet. Orthopaedics is the
only specialty with real data today (app/specialty/content/orthopaedics.py).

Deliberately schema-agnostic (duck-typed), not tied to
chunking.schemas.models.MicroChunk: GM's own MicroChunk uses a different
entity-type taxonomy (a strict Literal enum of generic clinical categories)
than the clones' free-form per-specialty entity types, and relations
reference entities BY NAME in GM's schema vs BY ID in the clones' schema.
Supporting both conventions without forcing a schema change is the point of
this module. A "chunk" here is anything exposing ``.entities`` (each with
``.name``, ``.type``, and optionally ``.id``/``.aliases``) and ``.relations``
(each with ``.source``, ``.target``, ``.type``) — entities and relations are
mutated in place, same contract as the Orthopaedics original.

NOT wired into any live ingestion pipeline by this change — see
AUDIT_REPORT.md's "do not touch Pinecone/Neo4j ingestion" constraint. This is
a reusable component only; wiring it into chunking/pipeline/manager.py for a
given specialty's re-ingestion run is a separate, later step.

Usage:
    from chunking.postprocessing import postprocess_chunk
    postprocess_chunk(chunk, specialty_config)  # no-op if the config has no maps
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from app.specialty.models import SpecialtyConfig

logger = logging.getLogger(__name__)


class _EntityLike(Protocol):
    name: str
    type: str


class _RelationLike(Protocol):
    source: str
    target: str
    type: str


class _ChunkLike(Protocol):
    entities: list[Any]
    relations: list[Any]


def _snake(name: str) -> str:
    """Lowercase snake_case id from a name (mirrors the Orthopaedics original)."""
    s = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return s or "unknown"


# ---------------------------------------------------------------------------
# Stage 1 — Entity overrides
# ---------------------------------------------------------------------------

def apply_entity_overrides(chunk: _ChunkLike, entity_overrides: dict[str, str]) -> int:
    """
    For each entity whose lowercased name is a key in ``entity_overrides``,
    replace its ``.type`` with the override value. Returns the count changed.
    No-op (returns 0) when ``entity_overrides`` is empty.
    """
    if not entity_overrides:
        return 0
    overridden = 0
    for entity in chunk.entities:
        override_type = entity_overrides.get(str(entity.name).lower())
        if override_type is not None and entity.type != override_type:
            entity.type = override_type
            overridden += 1
    if overridden:
        logger.debug("Entity override: corrected %d entities", overridden)
    return overridden


# ---------------------------------------------------------------------------
# Stage 2 — Relation repair
# ---------------------------------------------------------------------------

def repair_relations(
    chunk: _ChunkLike,
    relation_type_by_target_type: dict[str, str],
    *,
    generic_relation_type: str = "ASSOCIATED_WITH",
) -> int:
    """
    Replace generic ``generic_relation_type`` relations with a specific type
    inferred from the TARGET entity's type (via
    ``relation_type_by_target_type``, keyed by lowercased entity type).

    Entities are looked up by whichever key they expose: ``.id`` when
    present (the clones' convention), falling back to ``.name`` (GM's
    convention, where ClinicalRelation.source/target carry entity names).
    Only relations whose current type matches ``generic_relation_type``
    (case-insensitive) are touched. Returns the count repaired. No-op
    (returns 0) when ``relation_type_by_target_type`` is empty.
    """
    if not relation_type_by_target_type:
        return 0

    type_by_key: dict[str, str] = {}
    for e in chunk.entities:
        entity_id = getattr(e, "id", None)
        if entity_id:
            type_by_key[str(entity_id)] = str(e.type).lower()
        type_by_key[str(e.name)] = str(e.type).lower()

    repaired = 0
    for rel in chunk.relations:
        if str(rel.type).upper() != generic_relation_type.upper():
            continue
        target_type = type_by_key.get(str(rel.target))
        if target_type is None:
            continue
        new_type = relation_type_by_target_type.get(target_type)
        if new_type is not None:
            rel.type = new_type
            repaired += 1
    if repaired:
        logger.debug("Relation repair: %d %s -> specific", repaired, generic_relation_type)
    return repaired


# ---------------------------------------------------------------------------
# Stage 3 — Canonicalization
# ---------------------------------------------------------------------------

def canonicalize_entities(chunk: _ChunkLike, canonical_entities: dict[str, str]) -> int:
    """
    Normalize entity names to canonical forms (keyed by lowercased alias ->
    canonical name). The original name is preserved in ``entity.aliases``
    when that attribute exists; ``entity.id`` is regenerated from the
    canonical name ONLY when the entity already has an ``id`` attribute
    (GM's name-keyed schema has none, so GM entities are renamed in place
    without an id remap). Relation ``source``/``target`` references are
    re-pointed to match, whichever key convention (id or name) they used.
    Entities/relations that collapse to the same key are de-duplicated.
    Returns the count of entities canonicalized. No-op (returns 0) when
    ``canonical_entities`` is empty.
    """
    if not canonical_entities:
        return 0

    key_remap: dict[str, str] = {}  # old key (id or name) -> new key
    changed = 0

    for entity in chunk.entities:
        canonical = canonical_entities.get(str(entity.name).lower())
        if canonical is None:
            continue

        old_name = entity.name
        has_id = hasattr(entity, "id")
        old_key = str(getattr(entity, "id", old_name))

        if hasattr(entity, "aliases") and old_name.lower() != canonical.lower():
            if old_name not in entity.aliases:
                entity.aliases.append(old_name)

        entity.name = canonical
        new_key = _snake(canonical) if has_id else canonical
        if has_id:
            entity.id = new_key
        key_remap[old_key] = new_key
        # Also map the old plain name, in case relations reference by name
        # even when entities carry an id (defensive — covers mixed schemas).
        key_remap[old_name] = new_key
        changed += 1

    if not key_remap:
        return 0

    for rel in chunk.relations:
        if str(rel.source) in key_remap:
            rel.source = key_remap[str(rel.source)]
        if str(rel.target) in key_remap:
            rel.target = key_remap[str(rel.target)]

    # De-duplicate entities that collapsed to the same key.
    dedup_key = "id" if hasattr(chunk.entities[0], "id") else "name" if chunk.entities else "name"
    seen: dict[str, Any] = {}
    for e in chunk.entities:
        k = str(getattr(e, dedup_key, e.name))
        if k in seen:
            kept = seen[k]
            if hasattr(e, "aliases") and hasattr(kept, "aliases"):
                for a in e.aliases:
                    if a not in kept.aliases:
                        kept.aliases.append(a)
        else:
            seen[k] = e
    chunk.entities = list(seen.values())

    # De-duplicate relations that collapsed to the same (source, target, type).
    rel_seen: dict[tuple, Any] = {}
    for r in chunk.relations:
        key = (r.source, r.target, r.type)
        if key not in rel_seen:
            rel_seen[key] = r
    chunk.relations = list(rel_seen.values())

    if changed:
        logger.debug("Canonicalization: remapped %d entities", changed)
    return changed


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def postprocess_chunk(chunk: _ChunkLike, specialty_config: SpecialtyConfig) -> _ChunkLike:
    """
    Apply all three stages, in order, using ``specialty_config``'s maps.

    Entity overrides must precede relation repair (which reads entity
    types); canonicalization runs last (it may change entity keys).
    Mutates ``chunk`` in place and returns it for fluent chaining. A
    specialty with no curated maps (every specialty except orthopaedics
    today) is a complete no-op — safe to call unconditionally.
    """
    apply_entity_overrides(chunk, specialty_config.entity_overrides)
    repair_relations(chunk, specialty_config.relation_type_by_target_type)
    canonicalize_entities(chunk, specialty_config.canonical_entities)
    return chunk


__all__ = [
    "apply_entity_overrides",
    "repair_relations",
    "canonicalize_entities",
    "postprocess_chunk",
]
