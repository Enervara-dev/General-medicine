"""
Unit tests for the layered system-prompt composer.

The contract locked in here matches the experienced-clinician prompt
style: minimal high-signal questioning, probabilistic ranked
differentials with plain-English mechanisms, gated "consult a doctor"
phrasing (only when red flags or genuine uncertainty warrant it),
silent memory reuse with no re-asking, RAG integrated naturally
without meta-leak or fabrication, and escalation only for severe /
high-risk signs.

Assertions use ``.lower()`` substring matching where exact casing is
not the contract (case-sensitive only when capitalisation carries
weight, like section headers and ALL-CAPS rule emphasis). The
composed prompt fits a ~1100-token budget; tests target rule presence,
not exact wording.
"""

from __future__ import annotations

import pytest

from app.services.orchestration.prompt_layers import (
    compose_system_prompt,
    layer_core_identity,
    layer_formatting_constraints,
    layer_retrieval_grounding,
    layer_runtime_modifiers,
    layer_safety_policy,
    layer_session_state_instructions,
    layer_tool_instructions,
)


# ---------------------------------------------------------------------------
# Layer 1 — Behaviour rules (experienced clinician)
# ---------------------------------------------------------------------------


def test_core_identity_experienced_clinician_persona():
    out = layer_core_identity()
    lo = out.lower()
    assert "experienced gastroenterology clinician" in lo
    assert "senior doctor" in lo
    # Probabilistic clinical reasoning chain is the ethos.
    assert "probabilistic" in lo
    assert "differential" in lo
    assert "mechanism" in lo
    # Anti-defensive / anti-chatbot stance.
    assert "never defensive" in lo
    assert "never a robotic chatbot" in lo
    # Heard → thought about → helped (warmth + reasoning).
    assert "feel heard first" in lo


# ---------------------------------------------------------------------------
# Layer 2 — Safety & evidence constraints
# ---------------------------------------------------------------------------


def test_safety_probabilistic_and_evidence_grounded():
    out = layer_safety_policy().lower()
    assert "probabilistic language" in out
    assert "definitive diagnosis" in out
    assert "evidence-grounded" in out
    assert "established medical knowledge" in out
    # Anti-hallucination — explicit list of things never to invent.
    assert "never invent" in out
    for forbidden in ("symptoms", "mechanisms", "doses", "studies", "guidelines"):
        assert forbidden in out


def test_safety_gates_consult_a_doctor_phrase():
    out = layer_safety_policy().lower()
    # The phrase is no longer chanted — it's explicitly gated.
    assert "do not chant" in out
    assert "\"consult a doctor\"" in out
    # The disclaimer phrase remains available when warranted.
    assert "only a doctor can properly examine and confirm this" in out
    # Conditions for using it.
    assert "red flags" in out
    assert "genuine uncertainty" in out
    # Pairing requirements.
    assert "specific trigger" in out
    assert "timeframe" in out
    # Reject the bolt-on style.
    assert "mechanical bolt-on" in out or "bolt-on" in out


# ---------------------------------------------------------------------------
# Layer 3 — Runtime modifiers (risk + personalisation)
# ---------------------------------------------------------------------------


def test_runtime_personalisation_with_name():
    out = layer_runtime_modifiers(risk_level="none", has_name=True)
    assert "Hey Aarav" in out
    assert "sparingly" in out
    assert "PERSONALISATION" in out


def test_runtime_personalisation_no_name():
    out = layer_runtime_modifiers(risk_level="none", has_name=False).lower()
    assert "no name is known" in out
    assert "never invent" in out
    assert '"patient"' in out and '"user"' in out
    assert "hey aarav" not in out


def test_runtime_risk_critical_surfaces_warning():
    out = layer_runtime_modifiers(risk_level="critical", has_name=False)
    assert "⚠️ CRITICAL" in out
    # Critical block tells the LLM to skip the interview and escalate.
    assert "SKIP the interview" in out or "skip the interview" in out.lower()
    # Personalisation block still follows the risk header.
    assert "PERSONALISATION" in out


def test_runtime_risk_none_omits_header():
    out = layer_runtime_modifiers(risk_level="none", has_name=False)
    assert "⚠️" not in out
    assert "CRITICAL" not in out
    assert "Elevated risk" not in out


# ---------------------------------------------------------------------------
# Layer 4 — Memory & context reuse
# ---------------------------------------------------------------------------


def test_memory_reuse_silent_and_no_reasking():
    out = layer_session_state_instructions()
    lo = out.lower()
    assert "MEMORY & CONTEXT REUSE" in out
    # Memory is used silently, treated as already known.
    assert "silently" in lo
    assert "already known" in lo
    # The hard "never re-ask" rule, with named examples.
    assert "never re-ask" in lo
    for known_field in ("age", "sex", "name", "duration", "history", "meds"):
        assert known_field in lo
    # No restart, no echoing their own words.
    assert "never restart" in lo
    assert "echo" in lo or "summarise" in lo
    # New question takes priority.
    assert "current question is the priority" in lo
    assert "never redirect" in lo


# ---------------------------------------------------------------------------
# Layer 5 — RAG grounding policy
# ---------------------------------------------------------------------------


def test_retrieval_grounding_natural_integration():
    out = layer_retrieval_grounding()
    lo = out.lower()
    assert "CLINICAL KNOWLEDGE GROUNDING" in out
    assert "integrate it" in lo or "integrate it naturally" in lo
    # Paraphrase, never quote chunks verbatim.
    assert "paraphrase" in lo
    assert "never quote" in lo or "never quote chunks" in lo


def test_retrieval_grounding_no_meta_leak():
    out = layer_retrieval_grounding()
    assert "Never reference retrieval" in out
    # Each meta-leak term forbidden by name.
    for term in ("retrieval", "vectors", "summaries", "chunks", "graph", "memory"):
        assert term in out


def test_retrieval_grounding_no_fabrication():
    out = layer_retrieval_grounding().lower()
    assert "never fabricate" in out
    for forbidden in ("study", "dose", "brand", "guideline"):
        assert forbidden in out


# ---------------------------------------------------------------------------
# Layer 6 — Questioning strategy (minimal, high-signal)
# ---------------------------------------------------------------------------


def test_questioning_strategy_minimal_high_signal():
    out = layer_tool_instructions()
    lo = out.lower()
    assert "QUESTIONING STRATEGY" in out
    assert "high-signal" in lo
    # Materially-changing-the-differential threshold.
    assert "materially change" in lo
    assert "differential" in lo or "plan" in lo
    # Permission to skip questions if case is clear.
    assert "skip questions" in lo


def test_questioning_strategy_hard_caps():
    out = layer_tool_instructions().lower()
    # Hard cap: 1 per turn; 3 across the conversation.
    assert "hard cap" in out
    assert "1 question per turn" in out or "one question per turn" in out
    assert "3 follow-up" in out or "3 follow" in out


def test_questioning_strategy_every_question_explains_why():
    out = layer_tool_instructions().lower()
    assert "medical reasoning" in out
    # Worked example anchors the cadence.
    assert "chest pain" in out
    # Anti-padding rules.
    assert "never vague" in out
    assert "never multiple" in out
    assert "fill space" in out


# ---------------------------------------------------------------------------
# Layer 7 — Response format + escalation policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query_type", [
    "symptom_query", "diagnosis_query", "diagnosis",
    "medication_query", "treatment_query", "drug_interaction",
    "guideline", "lab_interpretation", "prognosis", "unknown",
])
def test_format_substantive_uses_response_format(query_type: str):
    out = layer_formatting_constraints(query_type=query_type)
    assert "RESPONSE FORMAT" in out
    assert "substantive clinical" in out.lower()


def test_format_substantive_natural_prose_not_labelled():
    out = layer_formatting_constraints(query_type="symptom_query").lower()
    assert "flowing natural prose" in out
    # No labelled headings / no A/B/C bullets in the output.
    assert "no labelled headings" in out or "no labelled" in out
    assert "no a/b/c" in out or "no a/b/c bullets" in out


def test_format_substantive_ranked_differential_with_mechanism():
    out = layer_formatting_constraints(query_type="symptom_query")
    lo = out.lower()
    assert "probabilistic ranked differential" in lo
    assert "2–3 likely causes" in out
    assert "mechanism" in lo
    # Mechanism example — reflux + valve atop the stomach.
    assert "reflux" in lo
    assert "valve" in lo


def test_format_substantive_why_this_not_that():
    out = layer_formatting_constraints(query_type="symptom_query").lower()
    assert "why-this-not-that" in out
    assert "what their pattern fits and what it doesn't" in out


def test_format_substantive_today_actions_specific():
    out = layer_formatting_constraints(query_type="symptom_query").lower()
    # Specific TODAY actions — not vague.
    assert "today" in out
    for specific in ("dose", "timing", "food", "posture", "fluids"):
        assert specific in out
    # The negative exemplar is named.
    assert "rest and water" in out
    assert "this week" in out


def test_format_substantive_next_step_timeframe_and_trigger():
    out = layer_formatting_constraints(query_type="symptom_query").lower()
    assert "timeframe" in out
    assert "trigger" in out
    # Generic deflection is rejected.
    assert "never a generic" in out
    assert "see a doctor" in out  # named as the anti-pattern


def test_format_substantive_concise_word_target():
    out = layer_formatting_constraints(query_type="symptom_query").lower()
    assert "concise" in out
    assert "actionable" in out
    assert "180 words" in out


def test_format_substantive_includes_escalation_policy():
    out = layer_formatting_constraints(query_type="symptom_query")
    assert "ESCALATION POLICY" in out
    # Emergency numbers gated to life-threatening only.
    assert "112" in out and "102" in out and "108" in out
    # Life-threatening red flags named.
    for red_flag in (
        "vomiting blood",
        "severe chest pain",
        "can't breathe",
        "collapse",
        "black tarry stool",
    ):
        assert red_flag in out
    # Routine GI complaints excluded from emergency-number territory.
    assert "NEVER show emergency numbers for routine" in out
    for routine in ("bloating", "acidity", "burping", "constipation"):
        assert routine in out


def test_format_substantive_escalation_skip_and_escalate():
    out = layer_formatting_constraints(query_type="symptom_query")
    assert "SKIP the interview" in out
    assert "escalate to emergency services" in out
    # High-risk-but-not-emergency path: timeframe + trigger, no 112.
    assert "high-risk-but-not-emergency" in out.lower()
    assert "specific trigger" in out.lower()


def test_format_non_substantive_is_short_and_strips_clinical_scaffold():
    out = layer_formatting_constraints(query_type="greeting")
    # Non-substantive — natural 1–2 sentences, no clinical scaffolding.
    assert "RESPONSE FORMAT" in out
    assert "non-substantive" in out.lower()
    assert "1–2 sentences" in out
    # No differential, no escalation, no doctor talk.
    assert "ESCALATION POLICY" not in out
    assert "RANKED DIFFERENTIAL" not in out
    assert "ranked differential" not in out.lower()
    # No interview scaffolding.
    assert "QUESTIONING STRATEGY" not in out
    assert "STEP 1" not in out
    assert "STEP 2" not in out


# ---------------------------------------------------------------------------
# Composer — joins, skips, idempotent, budget
# ---------------------------------------------------------------------------


def test_compose_joins_all_layers_for_substantive_with_name_critical():
    out = compose_system_prompt(
        query_type="symptom_query",
        risk_level="critical",
        has_name=True,
    )
    # Markers from every non-empty layer must appear in the composed prompt.
    assert "experienced gastroenterology clinician" in out         # L1
    assert "SAFETY & EVIDENCE" in out                              # L2
    assert "⚠️ CRITICAL" in out and "Hey Aarav" in out             # L3
    assert "MEMORY & CONTEXT REUSE" in out                         # L4
    assert "CLINICAL KNOWLEDGE GROUNDING" in out                   # L5
    assert "QUESTIONING STRATEGY" in out                           # L6
    assert "RESPONSE FORMAT" in out                                # L7
    assert "ESCALATION POLICY" in out                              # L7


def test_compose_skips_empty_layers_for_low_risk_no_name_greeting():
    out = compose_system_prompt(
        query_type="greeting",
        risk_level="none",
        has_name=False,
    )
    # Risk header is suppressed when risk_level is none.
    assert "⚠️ CRITICAL" not in out
    assert "Elevated risk" not in out
    # Personalisation still present but in the no-name variant.
    assert "no name is known" in out.lower()
    assert "Hey Aarav" not in out
    # Greeting → short non-substantive response format branch.
    assert "1–2 sentences" in out
    # Escalation policy is gated to substantive replies.
    assert "ESCALATION POLICY" not in out


def test_compose_idempotent_pure_function():
    args = dict(query_type="symptom_query", risk_level="low", has_name=True)
    a = compose_system_prompt(**args)
    b = compose_system_prompt(**args)
    assert a == b


def test_compose_no_blank_line_runs():
    out = compose_system_prompt(
        query_type="symptom_query",
        risk_level="none",
        has_name=False,
    )
    # Layers are joined with "\n\n"; no triple-newline runs should appear.
    assert "\n\n\n" not in out


def test_compose_defaults_safe():
    # No kwargs other than query_type — defaults risk=none, has_name=False.
    out = compose_system_prompt(query_type="symptom_query")
    assert "experienced gastroenterology clinician" in out
    assert "RESPONSE FORMAT" in out
    assert "⚠️" not in out
    assert "Hey Aarav" not in out
    assert "no name is known" in out.lower()


# ---------------------------------------------------------------------------
# Budget check — soft cap aligned with the bumped SYSTEM_PROMPT_MAX_TOKENS
# ---------------------------------------------------------------------------


def test_compose_typical_path_fits_token_budget():
    """
    The composed prompt for the substantive-no-name-no-risk path should fit
    within roughly 1150 tokens (~4600 chars, conservative 4-chars/token).
    The budget was bumped from 400 → 1150 to allow proper clinical-
    reasoning scaffolding (ranked differential, mechanism, why-this-not-
    that, specific today-actions, timeframe+trigger, escalation policy,
    RAG grounding, questioning strategy).
    """
    out = compose_system_prompt(
        query_type="symptom_query",
        risk_level="none",
        has_name=False,
    )
    chars = len(out)
    assert chars <= 4600, (
        f"Composed prompt is {chars} chars (~{chars // 4} tokens); "
        f"tighten layer text or re-evaluate budget."
    )
