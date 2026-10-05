"""
Default-specialty-lens consistency across the fleet.

Pins the requirement that every non-GM specialty interprets an ambiguous
complaint through its own clinical lens by default, asks specialty-flavored
follow-up questions, and only redirects via `suggested_specialty` when the
complaint is clearly a different body system — never merely because it is
unlocalised. general_medicine is intentionally excluded: it has no relevance
gate upstream and was out of scope for this pass.
"""

from app.specialty.registry import SPECIALTY_KEYS, build_default_registry

NON_GM_SPECIALTIES = tuple(k for k in SPECIALTY_KEYS if k != "general_medicine")


def test_every_non_gm_specialty_has_a_lens_first_persona_clause():
    reg = build_default_registry()
    for key in NON_GM_SPECIALTIES:
        persona = reg.get(key).persona
        assert "lens first" in persona, f"{key}: missing lens-first clause"


def test_every_non_gm_specialty_has_an_outside_scope_clause():
    reg = build_default_registry()
    for key in NON_GM_SPECIALTIES:
        persona = reg.get(key).persona
        assert "clearly outside" in persona, f"{key}: missing outside-scope clause"


def test_every_non_gm_specialty_reinforces_default_lens_in_gatekeeper():
    """The suggestion gate must say ambiguity alone is not grounds to redirect."""
    reg = build_default_registry()
    for key in NON_GM_SPECIALTIES:
        prompt = reg.get(key).gatekeeper_system_prompt
        assert "Default to your own specialty" in prompt, f"{key}: missing reinforcement"
        assert "never merely because it is unlocalised" in prompt, key


def test_general_medicine_is_unchanged_by_this_pass():
    """GM has no relevance gate upstream; explicitly out of scope."""
    reg = build_default_registry()
    gm = reg.get("general_medicine").gatekeeper_system_prompt
    assert "Default to your own specialty" not in gm
    assert "suggested_specialty" not in gm


def test_cardiology_no_longer_carries_the_pulmonology_triage_leftover():
    """
    Regression for a real bug found while closing the lens-parity gap:
    cardiology's triage-probe guidance previously read "blood in sputum...
    known lung disease" — copy-pasted pulmonology content, not a cardiology
    judgment call (unlike the separately-preserved respiratory-leaning risk
    section, which this change does not touch).
    """
    reg = build_default_registry()
    cardiology = reg.get("cardiology").gatekeeper_system_prompt
    assert "blood in sputum" not in cardiology
    assert "radiation to arm/jaw/back" in cardiology


def test_cardiology_persona_matches_the_fleet_template():
    persona = build_default_registry().get("cardiology").persona
    assert "Reason through a cardiology lens first" in persona
    assert "If a query is clearly outside cardiology" in persona


def test_every_specialty_prompt_is_syntactically_sound_python():
    """
    Catches the literal class of bug this change nearly shipped: a corrupted
    `\\n` inside a specialty's PERSONA tuple breaking the module at import.
    build_default_registry() already exercises every content module's import,
    so this is really just documenting why that call is the right guard.
    """
    reg = build_default_registry()
    for key in SPECIALTY_KEYS:
        cfg = reg.get(key)
        assert isinstance(cfg.persona, str) and cfg.persona
        assert isinstance(cfg.gatekeeper_system_prompt, str) and cfg.gatekeeper_system_prompt


def test_lens_description_matches_each_specialtys_own_domain():
    """Spot-check the fleet uses its own body-system vocabulary, not another's."""
    reg = build_default_registry()
    expectations = {
        "cardiology": "cardiovascular",
        "dermatology": "dermatological",
        "ent": "ENT",
        "ophthalmology": "ophthalmic",
        "orthopaedics": "musculoskeletal",
        "pulmonology": "pulmonary",
    }
    for key, term in expectations.items():
        assert term in reg.get(key).persona, f"{key}: expected {term!r} in persona"
