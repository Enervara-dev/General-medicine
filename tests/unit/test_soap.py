"""Unit tests for the doctor-summary flag block + SOAP note generation."""

from app.services.soap import build_soap_context, parse_soap
from graphrag.schemas.blocks import BLOCK_TYPES, BlockAdapter
from Memory_Layer.session_memory.models import Message, Role, SessionMemory


# ---------------------------------------------------------------------------
# answer_state block + sticky flag
# ---------------------------------------------------------------------------

def test_answer_state_block_is_a_valid_type():
    assert "answer_state" in BLOCK_TYPES
    block = BlockAdapter.validate_python(
        {"type": "answer_state", "data": {"show_doctor_summary": True}}
    )
    assert block.type == "answer_state"
    assert block.data.show_doctor_summary is True


def test_doctor_summary_ready_defaults_false_and_persists():
    s = SessionMemory(session_id="x")
    assert s.doctor_summary_ready is False
    s.doctor_summary_ready = True
    # Round-trips through serialization (Redis stores the whole model).
    restored = SessionMemory.model_validate(s.model_dump())
    assert restored.doctor_summary_ready is True


# ---------------------------------------------------------------------------
# SOAP context assembly (grounding source)
# ---------------------------------------------------------------------------

def _session_with_convo() -> SessionMemory:
    s = SessionMemory(session_id="soap")
    s.add_turn(Message(role=Role.USER, content="I have a fever"))
    s.add_turn(Message(role=Role.ASSISTANT, content="How long have you had it?"))
    s.add_turn(Message(role=Role.USER, content="5 days, about 102F"))
    s.state.symptoms = ["fever"]
    s.state.duration = ["5 days"]
    s.state.severity = ["102F"]
    return s


def test_build_soap_context_includes_transcript_and_state():
    ctx = build_soap_context(_session_with_convo())
    assert "Patient: I have a fever" in ctx
    assert "Assistant: How long have you had it?" in ctx
    assert "Symptoms: fever" in ctx
    assert "Duration: 5 days" in ctx


def test_build_soap_context_empty_session():
    ctx = build_soap_context(SessionMemory(session_id="empty"))
    assert "No conversation" in ctx


# ---------------------------------------------------------------------------
# SOAP parsing (tolerant, non-fabricating)
# ---------------------------------------------------------------------------

def test_parse_soap_valid_json():
    raw = (
        '{"subjective":"Fever 5 days","objective":"Temp 102F reported",'
        '"assessment":"Likely viral","plan":"Hydrate, monitor",'
        '"unavailable":["No exam performed"]}'
    )
    note = parse_soap(raw)
    assert note["subjective"] == "Fever 5 days"
    assert note["objective"] == "Temp 102F reported"
    assert note["unavailable"] == ["No exam performed"]


def test_parse_soap_strips_code_fence():
    raw = '```json\n{"subjective":"x","objective":"y","assessment":"z","plan":"p","unavailable":[]}\n```'
    note = parse_soap(raw)
    assert note["subjective"] == "x"
    assert note["plan"] == "p"


def test_parse_soap_garbage_returns_safe_fallback():
    note = parse_soap("not json at all")
    # No fabricated content; the gap is flagged instead.
    assert note["subjective"] == ""
    assert note["assessment"] == ""
    assert note["unavailable"] and "could not be generated" in note["unavailable"][0]


def test_parse_soap_missing_fields_default_empty():
    note = parse_soap('{"subjective":"only this"}')
    assert note["subjective"] == "only this"
    assert note["objective"] == ""
    assert note["plan"] == ""
    assert note["unavailable"] == []
