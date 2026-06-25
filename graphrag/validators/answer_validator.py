"""
Per-line answer-block validation with partial recovery.

STAGE 4 streams NDJSON: one JSON block object per line. We never wait for the
whole answer — each time a newline completes a line we parse + validate it as a
single ``Block``, forward it if valid, and log+drop it if not. This mirrors the
chunker's partial-recovery posture (``chunking/validators/schema_validator.py``):
one bad record must not sink the batch.

Two stream drivers are provided because the codebase has two stacks:
    iter_blocks(...)   — sync generator   (legacy CLI / GraphRAGPipeline)
    aiter_blocks(...)  — async generator  (FastAPI AsyncOrchestrator)

Both:
    * buffer raw tokens and split on "\\n",
    * yield each valid Block as soon as its line completes (first block reaches
      the client ASAP — no full-response buffering),
    * drop ``follow_up_questions`` blocks when ``terminal=True`` (contract D:
      a closing/assessment turn must not ask more questions),
    * flush the trailing line at end-of-stream (the model may omit the final \\n).
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Iterable, Iterator

from pydantic import ValidationError

from graphrag.schemas.blocks import Block, BlockAdapter

logger = logging.getLogger(__name__)

_TERMINAL_DROP_TYPE = "follow_up_questions"


def _validate_object(obj: object) -> Block | None:
    """Validate a parsed JSON object against the block schema."""
    try:
        return BlockAdapter.validate_python(obj)
    except ValidationError as exc:
        # Compact the pydantic error so logs stay one-line-ish.
        reasons = "; ".join(e.get("msg", "?") for e in exc.errors())
        logger.warning("answer block dropped — schema violation (%s): %r", reasons, repr(obj)[:200])
        return None


def validate_line(line: str) -> Block | None:
    """
    Parse and validate ONE NDJSON line as a single Block.

    Returns the Block on success, or None (with a logged reason) when the line
    is blank, not valid JSON, or not a valid Block. Never raises.
    """
    stripped = line.strip()
    if not stripped:
        return None

    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        logger.warning("answer block dropped — invalid JSON (%s): %r", exc, stripped[:200])
        return None

    return _validate_object(obj)


def _extract_complete_objects(buffer: str) -> tuple[list[object], str]:
    """Extract any complete JSON objects from the buffer, preserving a trailing partial fragment."""
    decoder = json.JSONDecoder()
    objects: list[object] = []
    remainder = buffer

    while True:
        stripped = remainder.lstrip()
        if not stripped:
            return objects, ""

        try:
            obj, end = decoder.raw_decode(stripped)
        except json.JSONDecodeError:
            return objects, stripped

        objects.append(obj)
        remainder = stripped[end:]


def _keep(block: Block | None, *, terminal: bool) -> Block | None:
    """Apply the terminal follow-up-questions drop rule to a validated block."""
    if block is None:
        return None
    if terminal and block.type == _TERMINAL_DROP_TYPE:
        logger.info("answer block dropped — follow_up_questions on terminal turn")
        return None
    return block


def iter_blocks(token_stream: Iterable[str], *, terminal: bool) -> Iterator[Block]:
    """
    Sync: turn a token stream into a stream of validated Blocks (partial recovery).
    """
    buffer = ""
    for token in token_stream:
        if not token:
            continue
        buffer += token
        objects, remainder = _extract_complete_objects(buffer)
        if objects:
            for obj in objects:
                kept = _keep(_validate_object(obj), terminal=terminal)
                if kept is not None:
                    yield kept
            buffer = remainder

    # Flush trailing content if it forms a complete object; otherwise drop it.
    kept = _keep(validate_line(buffer), terminal=terminal)
    if kept is not None:
        yield kept


async def aiter_blocks(token_stream: AsyncIterator[str], *, terminal: bool) -> AsyncIterator[Block]:
    """
    Async: turn an async token stream into a stream of validated Blocks.
    """
    buffer = ""
    async for token in token_stream:
        if not token:
            continue
        buffer += token
        objects, remainder = _extract_complete_objects(buffer)
        if objects:
            for obj in objects:
                kept = _keep(_validate_object(obj), terminal=terminal)
                if kept is not None:
                    yield kept
            buffer = remainder

    kept = _keep(validate_line(buffer), terminal=terminal)
    if kept is not None:
        yield kept


def block_to_line(block: Block) -> str:
    """Encode a Block as one NDJSON line (trailing newline included)."""
    return json.dumps(block.model_dump(mode="json"), ensure_ascii=False) + "\n"


def render_blocks_text(blocks: list[Block]) -> str:
    """
    Flatten validated blocks to plain text for session/episodic memory.

    The wire format is structured blocks, but memory stores a readable string
    (rolling summaries, history). This is lossy-but-faithful: it preserves the
    content, not the block envelope.
    """
    parts: list[str] = []
    for block in blocks:
        t = block.type
        d = block.data
        if t == "summary":
            parts.append(d.text)
        elif t == "key_points":
            parts.extend(f"- {p}" for p in d.points)
        elif t == "bullet_list":
            if d.title:
                parts.append(d.title)
            parts.extend(f"- {i}" for i in d.items)
        elif t == "follow_up_questions":
            parts.extend(f"? {q}" for q in d.questions)
        elif t == "warning":
            parts.append(f"[{d.severity}] {d.text}")
        elif t == "next_steps":
            parts.extend(f"-> {s}" for s in d.steps)
        elif t == "condition_list":
            for c in d.conditions:
                bits = [c.name]
                if c.likelihood:
                    bits.append(f"({c.likelihood})")
                if c.description:
                    bits.append(f"— {c.description}")
                parts.append(" ".join(bits))
    return "\n".join(parts).strip()
