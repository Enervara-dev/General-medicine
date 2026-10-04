"""
Tests for app.specialty — the registry that replaces per-service hardcoded
specialty content (persona, gatekeeper prompt, thresholds, red flags, entity
priorities, PMS source_service) with a config lookup.
"""

from __future__ import annotations

import pytest

from app.specialty import SPECIALTY_KEYS, SpecialtyConfig, SpecialtyRegistry, build_default_registry
from app.specialty.content import general_medicine as gm_content


@pytest.fixture(scope="module")
def registry() -> SpecialtyRegistry:
    return build_default_registry()


def test_registry_has_all_seven_specialties(registry: SpecialtyRegistry) -> None:
    assert set(registry.keys()) == set(SPECIALTY_KEYS)
    assert len(registry) == 7


@pytest.mark.parametrize("key", SPECIALTY_KEYS)
def test_every_specialty_has_required_content(registry: SpecialtyRegistry, key: str) -> None:
    cfg = registry.get(key)
    assert cfg.key == key
    assert cfg.display_name
    assert cfg.persona.strip(), f"{key} persona must not be empty"
    assert cfg.gatekeeper_system_prompt.strip(), f"{key} gatekeeper prompt must not be empty"
    assert isinstance(cfg.relevance_threshold, int)
    assert isinstance(cfg.red_flag_patterns, tuple)
    assert isinstance(cfg.entity_priority_types, tuple)


def test_resolve_known_key_returns_matching_config(registry: SpecialtyRegistry) -> None:
    cfg = registry.get("cardiology")
    assert cfg.key == "cardiology"
    assert cfg.source_service == "cardiology"
    assert "cardiologist" in cfg.persona.lower()


def test_resolve_unknown_key_raises_invalid_input(registry: SpecialtyRegistry) -> None:
    """
    A PROVIDED-but-wrong specialty is rejected, not silently defaulted. Per
    app/specialty/registry.py's module docstring: once a specialty selects a
    real Pinecone account/namespace, silently falling back to
    general_medicine for a typo would serve the wrong account's content with
    no signal to the caller — a clear 400 is safer than a quiet wrong answer.
    """
    from app.core.exceptions import InvalidInput

    with pytest.raises(InvalidInput) as exc_info:
        registry.get("not_a_real_specialty")
    assert "not_a_real_specialty" in str(exc_info.value)
    assert exc_info.value.status_code == 400


def test_resolve_none_falls_back_to_general_medicine(registry: SpecialtyRegistry) -> None:
    """Omitting `specialty` entirely (None) is the normal, supported shape
    every pre-unification caller uses — this must never raise."""
    cfg = registry.get(None)
    assert cfg.key == "general_medicine"


def test_resolve_empty_string_falls_back_to_general_medicine(registry: SpecialtyRegistry) -> None:
    cfg = registry.get("")
    assert cfg.key == "general_medicine"


def test_resolve_is_case_and_hyphen_insensitive(registry: SpecialtyRegistry) -> None:
    assert registry.get("Cardiology").key == "cardiology"
    assert registry.get("CARDIOLOGY").key == "cardiology"
    assert registry.get("general-medicine").key == "general_medicine"


def test_general_medicine_config_reproduces_prior_hardcoded_text(registry: SpecialtyRegistry) -> None:
    """
    The general_medicine SpecialtyConfig must carry the EXACT text that used
    to be hardcoded in prompt_layers.layer_core_identity() and
    analyzer.SYSTEM_PROMPT, so a request that omits `specialty` (every
    pre-unification caller) is byte-for-byte unaffected.
    """
    cfg = registry.get("general_medicine")
    assert cfg.persona == gm_content.PERSONA
    assert cfg.gatekeeper_system_prompt == gm_content.GATEKEEPER_SYSTEM_PROMPT


def test_specialty_registry_rejects_incomplete_config_map() -> None:
    with pytest.raises(ValueError, match="missing required specialties"):
        SpecialtyRegistry({"general_medicine": _minimal_config("general_medicine")})


def test_orthopaedics_has_data_quality_maps(registry: SpecialtyRegistry) -> None:
    ortho = registry.get("orthopaedics")
    assert len(ortho.entity_overrides) > 0
    assert len(ortho.relation_type_by_target_type) > 0
    assert len(ortho.canonical_entities) > 0


@pytest.mark.parametrize("key", [k for k in SPECIALTY_KEYS if k != "orthopaedics"])
def test_other_specialties_have_no_data_quality_maps(registry: SpecialtyRegistry, key: str) -> None:
    """
    Per AUDIT_REPORT.md §7, Orthopaedics is the only clone with curated
    ingestion-time data-quality maps — every other specialty must be a no-op
    for chunking.postprocessing.postprocess_chunk.
    """
    cfg = registry.get(key)
    assert cfg.entity_overrides == {}
    assert cfg.relation_type_by_target_type == {}
    assert cfg.canonical_entities == {}


def _minimal_config(key: str) -> SpecialtyConfig:
    return SpecialtyConfig(key=key, display_name=key, persona="p", gatekeeper_system_prompt="g")
