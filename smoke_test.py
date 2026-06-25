"""
Smoke tests for NDJSON STAGE-4 block streaming.

Runs fully offline — Gemini is stubbed at the token-stream level. Covers the
acceptance criteria from the task spec:

  * streamed lines each validate as a Block (incremental render)
  * a malformed line is dropped; valid lines before/after still stream
  * terminal=True drops follow_up_questions
  * critical risk -> warning(critical)+summary+condition_list+next_steps
  * refuse / out-of-scope / emergency stream blocks, not strings
  * first block reaches the client before the model finishes (no buffering)

Run directly:  python smoke_test.py
Or via pytest: pytest smoke_test.py
"""

from __future__ import annotations

import asyncio
import os

# Make GeminiLLM construct without real credentials.
os.environ.setdefault("GEMINI_API_KEY", "test-key")

from graphrag.domain.messages import (
    canned_blocks_for,
    emergency_blocks,
    is_terminal_turn,
    out_of_scope_blocks,
    refusal_blocks,
)
from graphrag.schemas.blocks import AnswerResponse, Block, BlockAdapter
from graphrag.validators.answer_validator import (
    aiter_blocks,
    block_to_line,
    iter_blocks,
    render_blocks_text,
    validate_line,
)

# A realistic model emission: tokens split across line boundaries on purpose.
_GOOD_TOKENS = [
    '{"type":"summary","data":',
    '{"text":"Night cough has several causes."}}\n',
    '{"type":"condition_list","data":{"conditions":[',
    '{"name":"Post-nasal drip","likelihood":"most likely","description":"mucus drips and irritates the throat"}]}}\n',
    '{"type":"follow_up_questions","data":{"questions":["Any heartburn?"]}}\n',
    '{"type":"next_steps","data":{"steps":["Try a saline rinse tonight."]}}',  # no trailing newline
]


def test_each_line_validates_as_block():
    blocks = list(iter_blocks(iter(_GOOD_TOKENS), terminal=False))
    types = [b.type for b in blocks]
    assert types == ["summary", "condition_list", "follow_up_questions", "next_steps"], types
    # Every block round-trips through the wire encoder + re-validates.
    for b in blocks:
        line = block_to_line(b)
        assert line.endswith("\n")
        assert validate_line(line) is not None


def test_malformed_line_dropped_neighbours_survive():
    tokens = [
        '{"type":"summary","data":{"text":"ok"}}\n',
        'this is not json at all\n',
        '{"type":"summary","data":{"text":"x","EXTRA":1}}\n',   # extra key -> forbidden
        '{"type":"next_steps","data":{"steps":["go"]}}\n',
    ]
    blocks = list(iter_blocks(iter(tokens), terminal=False))
    assert [b.type for b in blocks] == ["summary", "next_steps"]


def test_terminal_drops_followups():
    blocks = list(iter_blocks(iter(_GOOD_TOKENS), terminal=True))
    types = [b.type for b in blocks]
    assert "follow_up_questions" not in types
    assert types == ["summary", "condition_list", "next_steps"], types


def test_async_stream_matches_sync():
    async def _astream():
        for tok in _GOOD_TOKENS:
            yield tok

    async def _run():
        return [b.type async for b in aiter_blocks(_astream(), terminal=True)]

    types = asyncio.run(_run())
    assert types == ["summary", "condition_list", "next_steps"], types


def test_critical_risk_block_shape():
    """A critical-risk emission validates as warning(critical)+summary+condition_list+next_steps."""
    tokens = [
        '{"type":"warning","data":{"text":"Seek emergency care now.","severity":"critical"}}\n',
        '{"type":"summary","data":{"text":"These signs can be serious."}}\n',
        '{"type":"condition_list","data":{"conditions":[{"name":"Cardiac cause","likelihood":"possible"}]}}\n',
        '{"type":"next_steps","data":{"steps":["Call emergency services now."]}}\n',
    ]
    blocks = list(iter_blocks(iter(tokens), terminal=True))
    assert [b.type for b in blocks] == ["warning", "summary", "condition_list", "next_steps"]
    assert blocks[0].data.severity == "critical"

    # And the block-mode prompt actually instructs that structure on critical risk.
    from app.services.orchestration.prompt_layers import compose_system_prompt

    prompt = compose_system_prompt(
        query_type="symptom_query", risk_level="critical", output_format="blocks"
    )
    assert "CRITICAL RISK" in prompt
    assert "OUTPUT CONTRACT" in prompt
    assert "Do NOT emit follow_up_questions" in prompt


def test_canned_paths_emit_blocks_not_strings():
    for builder in (refusal_blocks, out_of_scope_blocks, emergency_blocks):
        blocks = builder()
        assert blocks, "builder returned no blocks"
        for b in blocks:
            assert not isinstance(b, str), "canned path emitted a raw string"
            assert hasattr(b, "type")
            # Each must be a valid Block (round-trips through the adapter).
            assert validate_line(block_to_line(b)) is not None

    assert [b.type for b in refusal_blocks()] == ["summary"]
    assert [b.type for b in out_of_scope_blocks()] == ["summary"]
    assert [b.type for b in emergency_blocks()] == ["warning", "next_steps"]
    assert canned_blocks_for("emergency_redirect")[0].type == "warning"
    assert canned_blocks_for("refuse")[0].type == "summary"


def test_first_block_before_stream_finishes():
    """The first valid block must be yielded before the token stream is exhausted."""
    consumed = {"n": 0}

    def lazy_tokens():
        for tok in _GOOD_TOKENS:
            consumed["n"] += 1
            yield tok

    gen = iter_blocks(lazy_tokens(), terminal=False)
    first = next(gen)
    assert first.type == "summary"
    # Only the first two token chunks are needed to complete the first line —
    # the rest of the stream has NOT been consumed yet.
    assert consumed["n"] < len(_GOOD_TOKENS), consumed["n"]


def test_terminal_helper_and_render():
    assert is_terminal_turn(turn_count=99, analysis={}) is True
    assert is_terminal_turn(turn_count=0, analysis={"final_action": "emergency_redirect"}) is True
    assert is_terminal_turn(turn_count=0, analysis={"final_action": "retrieve"}) is False
    # Confidence-based stopping: high confidence closes early even at turn 0.
    assert is_terminal_turn(turn_count=0, analysis={"diagnostic_confidence": 85}) is True
    assert is_terminal_turn(turn_count=0, analysis={"diagnostic_confidence": 40}) is False

    blocks = list(iter_blocks(iter(_GOOD_TOKENS), terminal=False))
    text = render_blocks_text(blocks)
    assert "Night cough" in text and "Post-nasal drip" in text


def test_generate_blocks_with_stubbed_gemini(monkeypatch=None):
    """End-to-end GeminiLLM.generate_blocks with the token stream stubbed."""
    import graphrag.llm.gemini_client as gc
    import graphrag.llm.gemini_llm as gl

    def fake_stream(*, user_prompt, model=None, system_instruction=None, temperature=None):
        yield from _GOOD_TOKENS

    # Patch the symbol the GeminiLLM module actually calls, and skip client init.
    orig_stream = gl.generate_stream
    orig_get_client = gc.get_client
    gl.generate_stream = fake_stream
    gc.get_client = lambda: object()
    try:
        llm = gl.GeminiLLM()
        blocks = list(
            llm.generate_blocks(
                query_text="why do I cough at night",
                vector_context="",
                graph_context="",
                terminal=True,
            )
        )
    finally:
        gl.generate_stream = orig_stream
        gc.get_client = orig_get_client

    types = [b.type for b in blocks]
    assert "follow_up_questions" not in types  # terminal drop
    assert types == ["summary", "condition_list", "next_steps"], types

    # Full block list also validates as an AnswerResponse for non-stream consumers.
    AnswerResponse(blocks=blocks)


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
