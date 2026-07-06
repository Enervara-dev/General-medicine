import pytest

from graphrag.validators.answer_validator import (
    _repair_obj,
    _validate_object,
    iter_blocks,
)


def test_iter_blocks_parses_multiple_complete_json_objects_from_one_chunk():
    token_stream = iter([
        '{"type":"summary","data":{"text":"Hello there"}}',
        '  ',
        '{"type":"next_steps","data":{"steps":["Call your doctor"]}}',
    ])

    blocks = list(iter_blocks(token_stream, terminal=False))

    assert [block.type for block in blocks] == ["summary", "next_steps"]
    assert blocks[0].data.text == "Hello there"
    assert blocks[1].data.steps == ["Call your doctor"]


# ---------------------------------------------------------------------------
# Schema-reliability: repair near-misses instead of dropping them (issue 2)
# ---------------------------------------------------------------------------


def test_repair_strips_hallucinated_extra_keys():
    # extra="forbid" would drop this; repair strips the unknown key.
    obj = {"type": "warning", "data": {"text": "watch this", "severity": "caution", "urgent": True}}
    block = _validate_object(obj)
    assert block is not None
    assert block.type == "warning"
    assert not hasattr(block.data, "urgent")


@pytest.mark.parametrize("raw, expected", [
    ("high", "critical"), ("severe", "critical"), ("emergency", "critical"),
    ("moderate", "caution"), ("warning", "caution"), ("medium", "caution"),
    ("low", "info"), ("note", "info"),
    ("banana", "caution"),  # unknown → safe default
])
def test_repair_coerces_out_of_enum_severity(raw, expected):
    block = _validate_object({"type": "warning", "data": {"text": "x", "severity": raw}})
    assert block is not None
    assert block.data.severity == expected


def test_repair_prunes_non_string_list_items():
    # A null/number in a list fails strict validation; repair prunes it rather
    # than dropping the whole block.
    block = _validate_object({"type": "next_steps", "data": {"steps": ["do this", None, 42, "and that"]}})
    assert block is not None
    assert block.data.steps == ["do this", "and that"]


def test_repair_remaps_single_string_under_wrong_key():
    # Observed live: key_points emitted with `text` instead of `points`.
    block = _validate_object({"type": "key_points", "data": {"text": "one important point"}})
    assert block is not None
    assert block.data.points == ["one important point"]


def test_unrepairable_block_still_dropped():
    # Empty text can't be fabricated — stays dropped, never repaired into junk.
    assert _validate_object({"type": "summary", "data": {"text": ""}}) is None
    assert _repair_obj({"type": "bogus", "data": {}}) is None


def test_valid_block_is_untouched_by_repair_path():
    block = _validate_object({"type": "summary", "data": {"text": "fine"}})
    assert block is not None and block.data.text == "fine"


# ---------------------------------------------------------------------------
# Never resolve to zero blocks (issue 2 — guaranteed renderable output)
# ---------------------------------------------------------------------------


def test_all_malformed_stream_yields_fallback_summary():
    tokens = iter(['not json at all\n', '{"type":"nope","data":{}}\n'])
    blocks = list(iter_blocks(tokens, terminal=False))
    assert len(blocks) == 1
    assert blocks[0].type == "summary"
    assert "couldn't" in blocks[0].data.text.lower()


def test_followup_block_capped_to_one_question():
    # Contract: at most one question per turn. A block with several is truncated.
    tokens = iter(['{"type":"follow_up_questions","data":{"questions":["q1","q2","q3"]}}\n'])
    blocks = list(iter_blocks(tokens, terminal=False))
    assert [b.type for b in blocks] == ["follow_up_questions"]
    assert blocks[0].data.questions == ["q1"]


def test_stream_with_one_valid_block_has_no_fallback():
    tokens = iter(['garbage\n', '{"type":"summary","data":{"text":"ok"}}\n'])
    blocks = list(iter_blocks(tokens, terminal=False))
    assert [b.type for b in blocks] == ["summary"]
    assert blocks[0].data.text == "ok"
