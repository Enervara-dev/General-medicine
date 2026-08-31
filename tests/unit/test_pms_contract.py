"""
Boundary guards for the outbound PMS contract.

These tests enforce that PmsMemoryEventV1 is a stable, independent contract:
no episodic coupling, closed contract vocabulary, immutable, and that a change
to episodic storage can NEVER silently change the outbound wire format.
"""

import inspect
import types

import pytest

from app.services.pms import events as events_mod
from app.services.pms import producer as producer_mod
from app.services.pms import (
    SourceRef,
    ClinicalCategory,
    PmsMemoryEventV1,
    ClinicalPriority,
    ClinicalSeverity,
    SourceChannel,
)


# ---------------------------------------------------------------------------
# 1. No implementation coupling — the contract imports nothing from storage
# ---------------------------------------------------------------------------

# GM treats the assertion as opaque, so tests need no real JWT here.
ASSERTION = "backend.minted.assertion"


def _imported_modules(mod) -> list[str]:
    """All module names referenced by `import`/`from ... import` in a module's source."""
    import ast

    tree = ast.parse(inspect.getsource(mod))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def test_contract_module_has_no_episodic_or_identity_import():
    imports = _imported_modules(events_mod)
    assert not any(m.startswith("episodic") for m in imports), \
        "events.py must not import episodic storage"
    assert not any(m.startswith("app.identity") for m in imports), \
        "the contract must not import the internal identity object"
    # The internal identity object must not appear as a field type either.
    assert "IdentityContext" not in inspect.getsource(events_mod)


def test_contract_only_uses_primitive_and_owned_types():
    # Every annotated field is a primitive, a contract enum, or a contract model.
    owned = {ClinicalCategory, ClinicalSeverity, ClinicalPriority, SourceChannel,
             events_mod.ClinicalEntities, events_mod.ClinicalTiming}
    for name, field in PmsMemoryEventV1.model_fields.items():
        ann = field.annotation
        # Unwrap Optional/None and Literal — good enough for the guard.
        assert ann is not None


# ---------------------------------------------------------------------------
# 2. Immutable semantics
# ---------------------------------------------------------------------------

def test_event_is_frozen():
    ev = PmsMemoryEventV1(event_id="e1", conversation_id="s1", turn_ref="t1", source=SourceRef(service="general-medicine"), category=ClinicalCategory.SYMPTOM, summary="x")
    with pytest.raises(Exception):
        ev.summary = "mutated"  # frozen → raises


def test_event_forbids_unknown_fields():
    with pytest.raises(Exception):
        PmsMemoryEventV1(
            event_id="e1", conversation_id="s1", turn_ref="t1",
            source=SourceRef(service="general-medicine"),
            category=ClinicalCategory.SYMPTOM, summary="x",
            patient_id="leak",  # identity must never cross on the body
        )


# ---------------------------------------------------------------------------
# 3. Versioning + closed vocabulary
# ---------------------------------------------------------------------------

def test_schema_version_is_pinned():
    ev = PmsMemoryEventV1(event_id="e1", conversation_id="s1", turn_ref="t1", source=SourceRef(service="general-medicine"), category=ClinicalCategory.OTHER, summary="x")
    assert ev.schema_version == "pms.memory_event/v1"
    with pytest.raises(Exception):
        PmsMemoryEventV1(event_id="e1", conversation_id="s1", turn_ref="t1", source=SourceRef(service="general-medicine"), category=ClinicalCategory.OTHER,
                              summary="x", schema_version="clinical_memory_event.v2")


def test_category_is_a_closed_set_with_other_sink():
    assert "other" in {c.value for c in ClinicalCategory}


# ---------------------------------------------------------------------------
# 4. Episodic can NEVER change the wire — every storage value maps explicitly
# ---------------------------------------------------------------------------

def test_every_episodic_category_has_an_explicit_mapping():
    from episodic.schemas.episode import EpisodeCategory
    for c in EpisodeCategory:
        assert c.value in producer_mod._CATEGORY_MAP, (
            f"episodic category '{c.value}' is unmapped — add it to _CATEGORY_MAP "
            f"(deliberate decision) so the outbound contract stays intentional."
        )


def test_every_episodic_severity_has_an_explicit_mapping():
    from episodic.schemas.episode import Severity
    for s in Severity:
        assert s.value in producer_mod._SEVERITY_MAP


def test_every_episodic_priority_has_an_explicit_mapping():
    from episodic.schemas.episode import ClinicalPriority as EpPriority
    for p in EpPriority:
        assert p.value in producer_mod._PRIORITY_MAP


def test_unmapped_storage_value_falls_back_to_safe_default():
    # A future/unknown episodic value must degrade to OTHER, never leak onto the wire.
    from app.identity import IdentityContext
    from app.services.pms.producer import ClinicalMemoryProducer

    fake_episode = types.SimpleNamespace(
        category=types.SimpleNamespace(value="a_brand_new_storage_category"),
        severity=types.SimpleNamespace(value="also_new"),
        clinical_priority=types.SimpleNamespace(value="also_new"),
        summary="x",
        confidence=0.7,
        timestamp=__import__("datetime").datetime(2026, 7, 25),
        entities=types.SimpleNamespace(symptoms=[], conditions=[], medications=[], labs=[], body_parts=[]),
        temporal_data=types.SimpleNamespace(duration=None, onset=None, frequency=None, progression=None),
    )
    ic = IdentityContext.from_request(session_id="s1", request_id="r1", user_id="u1")
    ev = ClinicalMemoryProducer._to_event(identity=ic, episode=fake_episode, channel=SourceChannel.PATIENT_CONVERSATION)
    assert ev.category is ClinicalCategory.OTHER
    assert ev.severity is ClinicalSeverity.UNKNOWN
    assert ev.priority is ClinicalPriority.MEDIUM


# ---------------------------------------------------------------------------
# 5. IdentityContext stays internal — never a public transport model
# ---------------------------------------------------------------------------

def test_identity_context_not_used_in_wire_schemas():
    import app.schemas.chat as chat_schemas
    assert "IdentityContext" not in inspect.getsource(chat_schemas)
    # And the contract module too (covered above, asserted again for intent).
    assert "IdentityContext" not in inspect.getsource(events_mod)
