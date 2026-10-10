"""
Two regressions, both fixed at the shared response-generation layer rather
than per specialty:

1. Emoji/emoticon removal from clinical responses. Root cause had two parts:
   (a) two Python string literals in app/services/orchestration/pipeline.py
       (the follow-up suffix, the canned emergency message) injected an emoji
       directly, independent of any LLM; (b) nothing told the model to avoid
       emojis and nothing scrubbed its output if it added one anyway. Fixed
       by removing the literal emoji at the two source sites, adding ONE
       shared "never use emojis" instruction to layer_safety_policy() (which
       composes into every specialty's prompt, both prose and block mode),
       and adding a deterministic graphrag.text.sanitize backstop applied at
       every point a clinical response is produced: _answer_async, stream()'s
       per-chunk loop, stream_blocks()'s per-block loop, and both SOAP
       generators.

2. Inconsistent / premature specialty-switch suggestions. Root cause: each
   specialty's gatekeeper prompt asked the model to "only suggest another
   specialty when YOUR OWN relevance score is low", but nothing enforced that
   relationship in code — a model could set a high suggested_specialty
   confidence on a turn it also scored as highly relevant to its own
   specialty, an internally contradictory claim that nothing caught. Fixed by
   adding a deterministic cross-check in the ONE shared validator,
   _extract_suggested_specialty: it now reads the gatekeeper's own
   {specialty}_relevance score and requires it to actually be below the
   specialty's relevance_threshold, failing CLOSED (rejecting the suggestion)
   when that score is missing or unreadable. This is enforced in code, so it
   holds regardless of how any one specialty's prompt is worded.
"""

import pytest

from app.services.orchestration.pipeline import (
    SUGGESTED_SPECIALTY_MIN_CONFIDENCE,
    AsyncOrchestrator,
    _canned_message,
    _extract_suggested_specialty,
)
from app.services.orchestration.prompt_layers import (
    _RISK_TONE,
    compose_system_prompt,
    layer_safety_policy,
)
from graphrag.text.sanitize import sanitize_value, strip_emojis

EMOJI = "\U0001F600"  # a plain grinning-face emoji, used generically below
WARNING = "⚠️"
SIREN = "\U0001F6A8"


# ---------------------------------------------------------------------------
# 1a. strip_emojis — the leaf sanitizer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        (f"{SIREN} Medical Emergency: call 112", "Medical Emergency: call 112"),
        (f"Great news {EMOJI} this looks mild", "Great news this looks mild"),
        (f"{WARNING} CRITICAL RISK here", "CRITICAL RISK here"),
        ("Feeling better :) thanks", "Feeling better thanks"),
        ("Feeling worse :-( please help", "Feeling worse please help"),
        ("Take this at 3:30 and a 1:1 ratio dose", "Take this at 3:30 and a 1:1 ratio dose"),
        ("history → mechanism → plan", "history → mechanism → plan"),
        ("  - sub-item still indented", "  - sub-item still indented"),
        ("no emoji here", "no emoji here"),
        ("", ""),
        (None, None),
    ],
)
def test_strip_emojis_removes_emoji_preserves_meaning_and_formatting(text, expected):
    assert strip_emojis(text) == expected


def test_strip_emojis_handles_a_grapheme_split_across_two_calls():
    """Simulates two adjacent streamed chunks, each sanitized independently
    (the real per-token streaming case) — no emoji survives in either half."""
    chunk_a = strip_emojis(f"Great news {EMOJI}")
    chunk_b = strip_emojis(f"{EMOJI} indeed, all clear")
    assert EMOJI not in chunk_a and EMOJI not in chunk_b


# ---------------------------------------------------------------------------
# 1b. sanitize_value — the generic walker used for blocks and SOAP
# ---------------------------------------------------------------------------

def test_sanitize_value_scrubs_a_plain_string():
    assert sanitize_value(f"hello {EMOJI} world") == "hello world"


def test_sanitize_value_scrubs_a_list_of_strings():
    assert sanitize_value(["ok", f"not ok {EMOJI}"]) == ["ok", "not ok"]


def test_sanitize_value_scrubs_a_dict_of_strings():
    out = sanitize_value({"subjective": f"patient reports pain {EMOJI}", "plan": "rest"})
    assert out == {"subjective": "patient reports pain", "plan": "rest"}


def test_sanitize_value_scrubs_a_pydantic_block_and_nested_models():
    from graphrag.schemas.blocks import Condition, ConditionListBlock, ConditionListData

    block = ConditionListBlock(
        type="condition_list",
        data=ConditionListData(conditions=[Condition(name=f"Tension headache {EMOJI}", likelihood="likely")]),
    )
    cleaned = sanitize_value(block.data)
    assert cleaned.conditions[0].name == "Tension headache"


def test_sanitize_value_scrubs_question_block_options():
    from graphrag.schemas.blocks import QuestionBlock, QuestionData

    block = QuestionBlock(
        type="question",
        data=QuestionData(question=f"Which of these {EMOJI}?", options=[f"Headache {EMOJI}", "Cough"]),
    )
    cleaned = sanitize_value(block.data)
    assert cleaned.question == "Which of these ?"
    assert cleaned.options == ["Headache", "Cough"]


def test_sanitize_value_leaves_non_string_fields_untouched():
    from graphrag.schemas.blocks import AnswerStateBlock, AnswerStateData

    block = AnswerStateBlock(type="answer_state", data=AnswerStateData(show_doctor_summary=True))
    cleaned = sanitize_value(block.data)
    assert cleaned.show_doctor_summary is True


# ---------------------------------------------------------------------------
# 1c. Hardcoded source literals — the two emoji-independent-of-the-model sites
# ---------------------------------------------------------------------------

def test_canned_emergency_message_carries_no_emoji():
    msg = _canned_message("emergency_redirect")
    assert strip_emojis(msg) == msg
    assert "Medical Emergency" in msg  # meaning preserved
    assert "112" in msg  # the actionable number is still there


def test_canned_refuse_and_crisis_messages_carry_no_emoji():
    for action in ("refuse", "mental_health_crisis"):
        msg = _canned_message(action)
        assert strip_emojis(msg) == msg


# ---------------------------------------------------------------------------
# 1d. The shared prompt layer — one instruction, every specialty, both modes
# ---------------------------------------------------------------------------

def test_layer_safety_policy_instructs_against_emojis():
    assert "NEVER use emojis" in layer_safety_policy()


def test_risk_tone_strings_carry_no_emoji():
    for level, text in _RISK_TONE.items():
        assert strip_emojis(text) == text, level


@pytest.mark.parametrize("output_format", ["prose", "blocks"])
def test_composed_prompt_carries_the_no_emoji_instruction_in_both_modes(output_format):
    prompt = compose_system_prompt(query_type="symptom_query", output_format=output_format)
    assert "NEVER use emojis" in prompt


def test_composed_prompt_carries_the_instruction_for_every_specialty_persona():
    """layer_core_identity swaps in the FULL persona, replacing the default
    text — confirm the shared safety layer still runs alongside it."""
    for persona in ("CARDIOLOGY_PERSONA", "DERMATOLOGY_PERSONA", "ANY_SPECIALTY_PERSONA"):
        prompt = compose_system_prompt(query_type="symptom_query", specialty_persona=persona)
        assert "NEVER use emojis" in prompt
        assert persona in prompt


# ---------------------------------------------------------------------------
# 1e. The deterministic backstop actually runs on the response-generation path
# ---------------------------------------------------------------------------

async def test_answer_async_sanitizes_an_emoji_laden_llm_response(monkeypatch):
    async def fake_generate_text_async(user_prompt, *, model, system_instruction, temperature, media=None):
        return f"This looks like a mild case {EMOJI} rest and fluids should help."

    monkeypatch.setattr("graphrag.llm.gemini_client.generate_text_async", fake_generate_text_async)

    orchestrator = AsyncOrchestrator.__new__(AsyncOrchestrator)
    answer = await AsyncOrchestrator._answer_async(
        orchestrator,
        query="cough",
        memory_context="",
        conversation_history="",
        vector_context="",
        graph_context="",
        query_type="symptom_query",
        goal="cause identification",
    )
    assert EMOJI not in answer
    assert answer == "This looks like a mild case rest and fluids should help."


async def test_answer_async_handles_llm_failure_without_crashing_the_sanitizer(monkeypatch):
    async def failing(*a, **kw):
        raise RuntimeError("upstream unavailable")

    monkeypatch.setattr("graphrag.llm.gemini_client.generate_text_async", failing)
    orchestrator = AsyncOrchestrator.__new__(AsyncOrchestrator)
    answer = await AsyncOrchestrator._answer_async(
        orchestrator, query="x", memory_context="", conversation_history="",
        vector_context="", graph_context="", query_type="symptom_query", goal="g",
    )
    assert answer == ""


async def test_soap_generator_sanitizes_emoji_in_every_section(monkeypatch):
    import json

    from app.services.soap import generate_soap_async
    from Memory_Layer.session_memory.models import SessionMemory

    async def fake_generate_text_async(user_prompt, *, model, system_instruction, temperature, media=None, json_mode=False):
        return json.dumps({
            "subjective": f"Patient reports headache {EMOJI}",
            "objective": "No vitals recorded.",
            "assessment": f"Likely tension headache {SIREN}",
            "plan": "OTC analgesic; review in 48h.",
            "unavailable": [f"Vitals not recorded {EMOJI}"],
        })

    monkeypatch.setattr("graphrag.llm.gemini_client.generate_text_async", fake_generate_text_async)
    sections = await generate_soap_async(SessionMemory(), model="stub-model")

    for key in ("subjective", "objective", "assessment", "plan"):
        assert EMOJI not in sections[key] and SIREN not in sections[key]
    assert all(EMOJI not in item for item in sections["unavailable"])
    assert sections["subjective"] == "Patient reports headache"
    assert sections["assessment"] == "Likely tension headache"


# ---------------------------------------------------------------------------
# 2. Specialty-redirect consistency — the deterministic own-relevance gate
# ---------------------------------------------------------------------------

def _suggestion(**over):
    base = {"slug": "dermatology", "confidence": 0.95, "reason_code": "skin_condition", "display_message": "msg"}
    base.update(over)
    return base


def test_contradictory_claim_is_rejected_high_own_relevance_but_suggests_anyway():
    """The exact bug this fixes: a model claiming it IS highly relevant to
    its own specialty while ALSO suggesting a redirect, which used to pass
    through on prompt-wording trust alone."""
    analysis = {"intent": "symptom_query", "ent_relevance": 90, "suggested_specialty": _suggestion()}
    assert _extract_suggested_specialty(analysis, current_specialty="ent", relevance_threshold=75) is None


def test_legitimate_low_relevance_suggestion_is_accepted():
    analysis = {"intent": "symptom_query", "ent_relevance": 20, "suggested_specialty": _suggestion()}
    out = _extract_suggested_specialty(analysis, current_specialty="ent", relevance_threshold=75)
    assert out is not None
    assert out.slug == "dermatology"


def test_own_relevance_exactly_at_threshold_is_rejected_not_below():
    analysis = {"intent": "symptom_query", "ent_relevance": 75, "suggested_specialty": _suggestion()}
    assert _extract_suggested_specialty(analysis, current_specialty="ent", relevance_threshold=75) is None


def test_missing_own_relevance_score_fails_closed():
    analysis = {"intent": "symptom_query", "suggested_specialty": _suggestion()}
    assert _extract_suggested_specialty(analysis, current_specialty="ent", relevance_threshold=75) is None


@pytest.mark.parametrize("bad", ["low", None, True, [], {}])
def test_malformed_own_relevance_score_fails_closed(bad):
    analysis = {"intent": "symptom_query", "ent_relevance": bad, "suggested_specialty": _suggestion()}
    assert _extract_suggested_specialty(analysis, current_specialty="ent", relevance_threshold=75) is None


def test_default_threshold_of_zero_accepts_nothing():
    """A caller that forgets to pass relevance_threshold gets the safe
    default: no real relevance score (0-100) is ever below 0."""
    analysis = {"intent": "symptom_query", "ent_relevance": 10, "suggested_specialty": _suggestion()}
    assert _extract_suggested_specialty(analysis, current_specialty="ent") is None


def test_self_suggestion_still_rejected_alongside_the_new_check():
    analysis = {"intent": "symptom_query", "ent_relevance": 10, "suggested_specialty": _suggestion(slug="ent")}
    assert _extract_suggested_specialty(analysis, current_specialty="ent", relevance_threshold=75) is None


def test_low_confidence_still_rejected_alongside_the_new_check():
    analysis = {
        "intent": "symptom_query", "ent_relevance": 10,
        "suggested_specialty": _suggestion(confidence=SUGGESTED_SPECIALTY_MIN_CONFIDENCE - 0.01),
    }
    assert _extract_suggested_specialty(analysis, current_specialty="ent", relevance_threshold=75) is None


# ---------------------------------------------------------------------------
# 2b. Drift guard — the prompt's stated threshold must match the enforced one
# ---------------------------------------------------------------------------

def test_every_non_gm_specialtys_stated_threshold_matches_its_enforced_threshold():
    """
    The prompt text still names its threshold in prose ("below 75") for the
    model's own benefit; this pins that number against the ACTUAL value the
    shared validator enforces (SpecialtyConfig.relevance_threshold), so the
    two can never silently drift apart even though they live in different
    places in the same file.
    """
    import re

    from app.specialty.registry import build_default_registry

    reg = build_default_registry()
    for key in ("cardiology", "dermatology", "ent", "ophthalmology", "orthopaedics", "pulmonology"):
        cfg = reg.get(key)
        m = re.search(rf"{key}_relevance is LOW \(below (\d+)\)", cfg.gatekeeper_system_prompt)
        assert m, f"{key}: could not find the stated threshold in its own prompt"
        assert int(m.group(1)) == cfg.relevance_threshold, key
