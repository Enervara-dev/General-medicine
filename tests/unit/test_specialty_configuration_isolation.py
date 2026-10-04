"""
Configuration-isolation tests: mutating one specialty's resolved config, or
the registry's handling of one specialty, must never leak into another
specialty or into a later lookup of the same specialty.

SpecialtyConfig is a frozen dataclass specifically to make the common
footgun (code accidentally mutating a shared config object in place)
impossible — these tests pin that guarantee.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.specialty import build_default_registry


@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


def test_specialty_config_is_frozen(registry):
    cardiology = registry.get("cardiology")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cardiology.persona = "mutated"  # type: ignore[misc]


def test_two_lookups_of_the_same_specialty_return_the_same_object(registry):
    a = registry.get("cardiology")
    b = registry.get("cardiology")
    assert a is b  # registry does not rebuild configs per lookup


def test_red_flag_pattern_tuples_are_independent_across_specialties(registry):
    cardio_flags = set(registry.get("cardiology").red_flag_patterns)
    derm_flags = set(registry.get("dermatology").red_flag_patterns)
    # Disjoint content would be a coincidence; what must hold is that they are
    # genuinely separate tuple objects, not shared/aliased storage.
    assert registry.get("cardiology").red_flag_patterns is not registry.get("dermatology").red_flag_patterns
    assert cardio_flags != derm_flags


def test_entity_override_maps_are_independent_across_specialties(registry):
    ortho_overrides = registry.get("orthopaedics").entity_overrides
    cardio_overrides = registry.get("cardiology").entity_overrides
    assert ortho_overrides is not cardio_overrides
    assert len(ortho_overrides) > 0
    assert cardio_overrides == {}
    # Mutating one specialty's dict (if a caller ever did, by mistake) must
    # not be visible on another specialty's config.
    ortho_overrides["femur"] = "SHOULD_NOT_LEAK"
    assert registry.get("cardiology").entity_overrides.get("femur") is None


def test_resolving_an_unknown_specialty_does_not_mutate_the_registry(registry):
    """get() now raises InvalidInput for an unrecognized key (see
    test_specialty_registry.py::test_resolve_unknown_key_raises_invalid_input)
    — this test only pins that the raise itself has no side effect on the
    registry's contents."""
    import pytest

    from app.core.exceptions import InvalidInput

    before = set(registry.keys())
    with pytest.raises(InvalidInput):
        registry.get("totally-made-up-specialty")
    with pytest.raises(InvalidInput):
        registry.get("another-fake-one")
    after = set(registry.keys())
    assert before == after
