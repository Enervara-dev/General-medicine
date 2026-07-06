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
    layer_block_plan,
    layer_core_identity,
    layer_formatting_constraints,
    layer_output_contract,
    layer_retrieval_grounding,
    layer_runtime_modifiers,
    layer_safety_policy,
    layer_session_state_instructions,
    layer_tool_instructions,
)
from graphrag.schemas.blocks import BLOCK_TYPES


# ---------------------------------------------------------------------------
# Layer 1 — Behaviour rules (experienced clinician)
# ---------------------------------------------------------------------------


def test_core_identity_experienced_clinician_persona():
    out = layer_core_identity()
    lo = out.lower()
    # General-medicine physician, NOT a single specialty.
    assert "general" in lo and ("physician" in lo or "medicine" in lo)
    assert "gastroenterology clinician" not in lo
    # Breadth across body systems (not GI-locked).
    assert "respiratory" in lo or "cardiac" in lo or "neurological" in lo
    assert "thoughtful" in lo or "doctor in clinic" in lo
    # Probabilistic clinical reasoning chain is the ethos.
    assert "probabilistic" in lo
    assert "differential" in lo
    assert "mechanism" in lo
    # Anti-defensive / anti-chatbot stance.
    assert "never defensive" in lo
    assert "robotic" in lo
    # Heard → thought about → helped (warmth + reasoning).
    assert "patient's concern" in lo or "patient" in lo


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
    assert "clinically useful" in lo or "materially" in lo or "diagnosis" in lo
    assert "diagnosis" in lo or "management" in lo
    # Permission to defer questions or diagnosis when evidence is still limited.
    assert "gathering information" in lo or "defer" in lo or "guessing" in lo


def test_questioning_strategy_hard_caps():
    out = layer_tool_instructions().lower()
    # Hard cap: 1 per turn.
    assert "one follow-up question" in out.lower() or "one question" in out.lower()
    assert "at most" in out.lower()


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
# Layer 7a — PROSE response format (untouched /chat + /chat/stream paths)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query_type", [
    "symptom_query", "diagnosis_query", "diagnosis",
    "medication_query", "treatment_query", "drug_interaction",
    "guideline", "lab_interpretation", "prognosis", "unknown",
])
def test_prose_substantive_uses_response_format(query_type: str):
    out = layer_formatting_constraints(query_type=query_type)
    assert "RESPONSE FORMAT" in out
    assert "substantive clinical" in out.lower()
    assert "flowing natural prose" in out.lower()


def test_prose_substantive_includes_escalation_policy():
    out = layer_formatting_constraints(query_type="symptom_query")
    assert "ESCALATION POLICY" in out
    assert "112" in out and "102" in out and "108" in out
    assert "SKIP the interview" in out


def test_prose_does_not_leak_query_header():
    out = layer_formatting_constraints(query_type="symptom_query")
    # The "(query:" parenthetical that used to bleed into answers is gone.
    assert "(query:" not in out
    assert "Never repeat it" in out


def test_prose_non_substantive_is_short():
    out = layer_formatting_constraints(query_type="greeting")
    assert "non-substantive" in out.lower()
    assert "1–2 sentences" in out
    assert "ESCALATION POLICY" not in out


# ---------------------------------------------------------------------------
# Layer 7b — BLOCK plan (NDJSON /chat/blocks path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query_type", [
    "symptom_query", "diagnosis_query", "diagnosis",
    "medication_query", "treatment_query", "drug_interaction",
    "guideline", "lab_interpretation", "prognosis", "unknown",
])
def test_block_plan_substantive(query_type: str):
    out = layer_block_plan(query_type=query_type)
    assert "BLOCK PLAN" in out
    assert "substantive clinical reply" in out.lower()
    assert "summary" in out
    assert "next_steps" in out


def test_block_plan_symptom_leads_with_empathy_and_confidence_stopping():
    """Symptom queries should start conversationally and stop once the diagnosis is confident."""
    out = layer_block_plan(query_type="symptom_query")
    lo = out.lower()
    assert "empathy" in lo or "empathetic" in lo
    assert "follow-up" in lo or "question" in lo
    assert "confidence" in lo or "high confidence" in lo
    assert "80" in out or "80%" in lo or "high" in lo
    assert "condition_list" not in out or "only emit condition_list" in lo
    assert "follow_up_questions" in out


def test_block_plan_followups_gated_by_allow_flag():
    on = layer_block_plan(query_type="symptom_query", allow_followups=True)
    assert "follow_up_questions" in on
    off = layer_block_plan(query_type="symptom_query", allow_followups=False)
    assert "follow_up_questions" not in off


def test_block_plan_terminal_drops_followups_and_notes_closing_turn():
    out = layer_block_plan(query_type="symptom_query", terminal=True, allow_followups=True)
    assert "at most ONE high-signal" not in out
    assert "closing/assessment turn" in out
    assert "do not emit a follow_up_questions" in out.lower()


def test_block_plan_critical_risk_structure():
    out = layer_block_plan(query_type="symptom_query", risk_level="critical")
    assert "CRITICAL RISK" in out
    assert 'severity "critical"' in out
    assert "summary" in out
    assert "condition_list" in out
    assert "next_steps" in out
    assert "Do NOT emit follow_up_questions" in out


def test_block_plan_greeting_single_summary():
    out = layer_block_plan(query_type="greeting")
    assert "BLOCK PLAN" in out and "greeting" in out.lower()
    assert "one `summary`" in out
    assert "condition_list" not in out


def test_block_plan_non_substantive_single_summary():
    out = layer_block_plan(query_type="followup_query")
    assert "non-substantive" in out.lower()
    assert "one `summary`" in out
    assert "No condition_list" in out
    assert "no follow_up_questions" in out


# ---------------------------------------------------------------------------
# Layer 8 — Output contract (NDJSON)
# ---------------------------------------------------------------------------


def test_output_contract_is_ndjson_and_lists_all_block_types():
    out = layer_output_contract()
    assert "OUTPUT CONTRACT" in out
    assert "NDJSON" in out
    # Single source of truth: every BLOCK_TYPES value is named.
    for bt in BLOCK_TYPES:
        assert bt in out
    # Forbids array/wrapping/markdown and shows the two-line example.
    lo = out.lower()
    assert "one json block object per line" in lo
    assert "the entire reply must be json only" in lo
    assert "no surrounding array" in lo or "no array" in lo
    assert "no prose outside the json" in lo
    assert '{"type":"summary"' in out
    assert '"steps"' in out
    assert '"conditions"' in out
    assert '"description"' in out
    assert 'do not rename fields' in lo or 'required fields' in lo


# ---------------------------------------------------------------------------
# Composer — joins, skips, idempotent, budget
# ---------------------------------------------------------------------------


def test_compose_joins_all_layers_for_substantive_with_name_critical():
    # Default (prose) mode — the untouched /chat path.
    out = compose_system_prompt(
        query_type="symptom_query",
        risk_level="critical",
        has_name=True,
    )
    # Markers from every non-empty layer must appear in the composed prompt.
    assert "experienced physician practising general" in out       # L1
    assert "SAFETY & EVIDENCE" in out                              # L2
    assert "⚠️ CRITICAL" in out and "Hey Aarav" in out             # L3
    assert "MEMORY & CONTEXT REUSE" in out                         # L4
    assert "CLINICAL KNOWLEDGE GROUNDING" in out                   # L5
    assert "QUESTIONING STRATEGY" in out                           # L6
    assert "RESPONSE FORMAT" in out and "ESCALATION POLICY" in out  # L7 prose
    # Prose mode does NOT carry the NDJSON contract.
    assert "OUTPUT CONTRACT" not in out


def test_compose_prose_is_default_no_block_contract():
    out = compose_system_prompt(query_type="symptom_query")
    assert "RESPONSE FORMAT" in out
    assert "OUTPUT CONTRACT" not in out
    assert "BLOCK PLAN" not in out


def test_compose_blocks_mode_appends_output_contract_last():
    out = compose_system_prompt(query_type="symptom_query", output_format="blocks")
    assert "BLOCK PLAN" in out
    # The NDJSON contract is always the final layer in block mode.
    assert out.index("OUTPUT CONTRACT") > out.index("BLOCK PLAN")
    assert out.rstrip().endswith("}")  # ends on the example's closing brace
    # Block mode replaces the prose format layer.
    assert "RESPONSE FORMAT" not in out


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
    # Greeting → short prose non-substantive branch (default mode).
    assert "1–2 sentences" in out
    # No clinical escalation scaffolding for a greeting.
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
    # No kwargs other than query_type — defaults risk=none, has_name=False, prose.
    out = compose_system_prompt(query_type="symptom_query")
    assert "experienced physician practising general" in out
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
    within roughly 1300 tokens (~5200 chars, conservative 4-chars/token).
    The cap was raised 4600 → 5200 when the NDJSON OUTPUT CONTRACT layer was
    added, then 5200 → 5500 when the identity broadened from a single specialty
    to a general-medicine physician with an explicit body-system scope.
    """
    out = compose_system_prompt(
        query_type="symptom_query",
        risk_level="none",
        has_name=False,
    )
    chars = len(out)
    assert chars <= 5500, (
        f"Composed prompt is {chars} chars (~{chars // 4} tokens); "
        f"tighten layer text or re-evaluate budget."
    )
