"""
Tests for the generalized ingestion-time data-quality pipeline
(chunking.postprocessing), using Orthopaedics' real maps — the only
specialty with curated entity-override / relation-repair / canonicalization
data (AUDIT_REPORT.md §7). Also proves the mechanism is schema-agnostic
(works against both an id-keyed chunk, the clones' convention, and a
name-keyed chunk, GM's own ClinicalRelation convention) and a true no-op for
every other specialty.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.specialty import SPECIALTY_KEYS, build_default_registry
from chunking.postprocessing import postprocess_chunk
from chunking.postprocessing.postprocessor import (
    apply_entity_overrides,
    canonicalize_entities,
    repair_relations,
)


@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


@pytest.fixture(scope="module")
def orthopaedics(registry):
    return registry.get("orthopaedics")


def _id_keyed_chunk():
    """Mimics the Orthopaedics clone's chunk schema: relations reference entity .id."""
    femur = SimpleNamespace(id="femur", name="femur", type="WrongType", aliases=[])
    pain = SimpleNamespace(id="pain", name="pain", type="WrongType", aliases=[])
    mri = SimpleNamespace(id="mri", name="mri", type="Diagnostic_Test", aliases=[])
    rel = SimpleNamespace(source="femur", target="pain", type="ASSOCIATED_WITH")
    return SimpleNamespace(entities=[femur, pain, mri], relations=[rel])


def _name_keyed_chunk():
    """Mimics GM's own ClinicalRelation schema: relations reference entity names, no .id."""
    femur = SimpleNamespace(name="femur", type="WrongType")
    pain = SimpleNamespace(name="pain", type="WrongType")
    rel = SimpleNamespace(source="femur", target="pain", type="ASSOCIATED_WITH")
    return SimpleNamespace(entities=[femur, pain], relations=[rel])


# ---------------------------------------------------------------------------
# Stage 1 — entity overrides
# ---------------------------------------------------------------------------

def test_entity_override_corrects_known_orthopaedic_entity(orthopaedics):
    chunk = _id_keyed_chunk()
    changed = apply_entity_overrides(chunk, orthopaedics.entity_overrides)
    assert changed == 2  # femur, pain; mri was already correctly typed
    femur = next(e for e in chunk.entities if e.name == "femur")
    assert femur.type == "Anatomical_Structure"
    pain = next(e for e in chunk.entities if e.name == "pain")
    assert pain.type == "Symptom"


def test_entity_override_is_noop_with_empty_map():
    chunk = _id_keyed_chunk()
    changed = apply_entity_overrides(chunk, {})
    assert changed == 0
    assert chunk.entities[0].type == "WrongType"


# ---------------------------------------------------------------------------
# Stage 2 — relation repair
# ---------------------------------------------------------------------------

def test_relation_repair_replaces_associated_with_using_target_type(orthopaedics):
    chunk = _id_keyed_chunk()
    apply_entity_overrides(chunk, orthopaedics.entity_overrides)  # must run first
    changed = repair_relations(chunk, orthopaedics.relation_type_by_target_type)
    assert changed == 1
    assert chunk.relations[0].type == "PRESENTS_WITH"  # target (pain) is a Symptom


def test_relation_repair_leaves_non_generic_relations_untouched(orthopaedics):
    chunk = _id_keyed_chunk()
    chunk.relations[0].type = "CAUSES"
    repair_relations(chunk, orthopaedics.relation_type_by_target_type)
    assert chunk.relations[0].type == "CAUSES"


def test_relation_repair_leaves_unmapped_target_type_untouched(orthopaedics):
    chunk = _id_keyed_chunk()
    # mri's type (Diagnostic_Test) via entity_overrides maps to DIAGNOSED_BY,
    # but here we point a relation at an entity type with no mapping.
    chunk.entities[1].type = "SomeUnmappedType"
    repair_relations(chunk, orthopaedics.relation_type_by_target_type)
    assert chunk.relations[0].type == "ASSOCIATED_WITH"


# ---------------------------------------------------------------------------
# Stage 3 — canonicalization
# ---------------------------------------------------------------------------

def test_canonicalization_normalizes_alias_and_remaps_relation_by_id(orthopaedics):
    mri = SimpleNamespace(id="mri", name="mri", type="Diagnostic_Test", aliases=[])
    femur = SimpleNamespace(id="femur", name="femur", type="Anatomical_Structure", aliases=[])
    rel = SimpleNamespace(source="femur", target="mri", type="DIAGNOSED_BY")
    chunk = SimpleNamespace(entities=[femur, mri], relations=[rel])

    changed = canonicalize_entities(chunk, orthopaedics.canonical_entities)

    assert changed == 1
    canon = next(e for e in chunk.entities if e.id == "magnetic_resonance_imaging")
    assert canon.name == "magnetic resonance imaging"
    assert "mri" in canon.aliases
    assert rel.target == "magnetic_resonance_imaging"


def test_canonicalization_on_name_keyed_entities_has_no_id_attr(orthopaedics):
    chunk = SimpleNamespace(
        entities=[SimpleNamespace(name="mri", type="Diagnostic_Test")],
        relations=[SimpleNamespace(source="x", target="mri", type="DIAGNOSED_BY")],
    )
    canonicalize_entities(chunk, orthopaedics.canonical_entities)
    assert chunk.entities[0].name == "magnetic resonance imaging"
    assert not hasattr(chunk.entities[0], "id")
    assert chunk.relations[0].target == "magnetic resonance imaging"


# ---------------------------------------------------------------------------
# Orchestrator — postprocess_chunk
# ---------------------------------------------------------------------------

def test_postprocess_chunk_runs_all_three_stages_in_order(orthopaedics):
    chunk = _id_keyed_chunk()
    postprocess_chunk(chunk, orthopaedics)

    femur = next(e for e in chunk.entities if "femur" in e.aliases or e.name == "femur")
    assert femur.type == "Anatomical_Structure"
    pain = next(e for e in chunk.entities if e.name == "pain")
    assert pain.type == "Symptom"
    assert chunk.relations[0].type == "PRESENTS_WITH"


def test_postprocess_chunk_works_on_name_keyed_schema_too(orthopaedics):
    """Proves the mechanism is NOT Orthopaedics-specific code wired to one
    chunk schema — it works against GM's own name-keyed convention using the
    exact same Orthopaedics data, which is the point of generalizing it."""
    chunk = _name_keyed_chunk()
    postprocess_chunk(chunk, orthopaedics)

    assert chunk.entities[0].type == "Anatomical_Structure"
    assert chunk.entities[1].type == "Symptom"
    assert chunk.relations[0].type == "PRESENTS_WITH"


@pytest.mark.parametrize("key", [k for k in SPECIALTY_KEYS if k != "orthopaedics"])
def test_postprocess_chunk_is_a_true_noop_for_every_other_specialty(registry, key):
    cfg = registry.get(key)
    chunk = _id_keyed_chunk()
    before_types = [e.type for e in chunk.entities]
    before_rel_type = chunk.relations[0].type

    postprocess_chunk(chunk, cfg)

    assert [e.type for e in chunk.entities] == before_types
    assert chunk.relations[0].type == before_rel_type
