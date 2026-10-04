"""
Tests that specialty parameterization (app.specialty) actually reaches the
composed system prompt, without disturbing the default (no-specialty)
behavior that tests/unit/test_prompt_layers.py locks down.
"""

from __future__ import annotations

from app.services.orchestration.prompt_layers import compose_system_prompt, layer_core_identity
from app.specialty import SPECIALTY_KEYS, build_default_registry


def test_omitting_persona_preserves_default_identity():
    assert layer_core_identity() == layer_core_identity(None)
    assert "general (internal)" in layer_core_identity().lower()


def test_passing_a_persona_overrides_the_default_identity():
    out = layer_core_identity("You are a test persona.")
    assert out == "You are a test persona."
    assert "general (internal)" not in out.lower()


def test_compose_system_prompt_without_specialty_persona_is_unchanged():
    baseline = compose_system_prompt(query_type="symptom_query", risk_level="none", output_format="prose")
    explicit_none = compose_system_prompt(
        query_type="symptom_query", risk_level="none", output_format="prose", specialty_persona=None
    )
    assert baseline == explicit_none


def test_each_specialty_persona_produces_a_distinct_composed_prompt():
    registry = build_default_registry()
    prompts = {
        key: compose_system_prompt(
            query_type="symptom_query",
            risk_level="none",
            output_format="prose",
            specialty_persona=registry.get(key).persona,
        )
        for key in SPECIALTY_KEYS
    }
    # No two specialties should produce an identical system prompt.
    assert len(set(prompts.values())) == len(SPECIALTY_KEYS)


def test_specialty_persona_surfaces_specialty_specific_vocabulary():
    registry = build_default_registry()
    cardiology_prompt = compose_system_prompt(
        query_type="symptom_query",
        risk_level="none",
        output_format="prose",
        specialty_persona=registry.get("cardiology").persona,
    )
    assert "cardiovascular" in cardiology_prompt.lower() or "cardiologist" in cardiology_prompt.lower()

    dermatology_prompt = compose_system_prompt(
        query_type="symptom_query",
        risk_level="none",
        output_format="prose",
        specialty_persona=registry.get("dermatology").persona,
    )
    assert "dermatolog" in dermatology_prompt.lower()


def test_block_mode_also_honors_specialty_persona():
    registry = build_default_registry()
    out = compose_system_prompt(
        query_type="symptom_query",
        risk_level="none",
        output_format="blocks",
        specialty_persona=registry.get("orthopaedics").persona,
    )
    assert "orthopaedic" in out.lower() or "musculoskeletal" in out.lower()
