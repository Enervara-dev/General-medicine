import pytest

from graphrag.validators.answer_validator import iter_blocks


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
