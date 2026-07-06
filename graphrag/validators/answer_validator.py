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

# The `data` fields each block type actually defines. Anything else is a
# hallucinated key that `extra="forbid"` would reject — we strip it instead.
_ALLOWED_DATA_FIELDS: dict[str, set[str]] = {
    "summary": {"text"},
    "key_points": {"points"},
    "bullet_list": {"title", "items"},
    "follow_up_questions": {"questions"},
    "warning": {"text", "severity"},
    "next_steps": {"steps"},
    "condition_list": {"conditions"},
}

# List-valued data fields — empty/blank entries are pruned before validation.
_LIST_FIELDS: tuple[str, ...] = ("points", "items", "questions", "steps")

# Map the severities a model commonly emits onto the three the schema allows.
_SEVERITY_ALIASES: dict[str, str] = {
    "info": "info", "information": "info", "informational": "info",
    "note": "info", "low": "info", "mild": "info", "minor": "info",
    "caution": "caution", "warning": "caution", "warn": "caution",
    "moderate": "caution", "medium": "caution", "concern": "caution",
    "critical": "critical", "severe": "critical", "high": "critical",
    "danger": "critical", "emergency": "critical", "urgent": "critical",
}


def _repair_obj(obj: object) -> dict | None:
    """
    Best-effort coercion of a near-miss block dict into schema shape.

    Handles the common, safe-to-fix failure modes — hallucinated extra keys, a
    non-canonical ``severity``, and blank list entries — WITHOUT fabricating any
    content. Returns a cleaned dict, or None if the object isn't a repairable
    block. The caller re-validates strictly, so a bad repair is still dropped.
    """
    if not isinstance(obj, dict):
        return None
    btype = obj.get("type")
    allowed = _ALLOWED_DATA_FIELDS.get(btype) if isinstance(btype, str) else None
    data = obj.get("data")
    if allowed is None or not isinstance(data, dict):
        return None

    # Remap a common field-name confusion: a single-string payload given under
    # the wrong key. e.g. key_points/{text} -> key_points/{points:[text]}.
    if btype in ("key_points", "next_steps", "follow_up_questions") and isinstance(data.get("text"), str):
        target = {"key_points": "points", "next_steps": "steps", "follow_up_questions": "questions"}[btype]
        if not data.get(target):
            data = {**data, target: [data["text"]]}
    if btype == "summary" and not data.get("text"):
        for alt in ("points", "steps", "questions"):
            if isinstance(data.get(alt), list) and data[alt]:
                data = {**data, "text": " ".join(str(x) for x in data[alt])}
                break

    clean = {k: v for k, v in data.items() if k in allowed}  # strip extra keys

    if btype == "warning":
        sev = str(clean.get("severity", "")).strip().lower()
        clean["severity"] = _SEVERITY_ALIASES.get(sev, "caution")

    for field in _LIST_FIELDS:
        val = clean.get(field)
        if isinstance(val, list):
            clean[field] = [s for s in val if isinstance(s, str) and s.strip()]

    return {"type": btype, "data": clean}


def _validate_object(obj: object) -> Block | None:
    """Validate a parsed JSON object against the block schema, repairing near-misses."""
    try:
        return BlockAdapter.validate_python(obj)
    except ValidationError as exc:
        repaired = _repair_obj(obj)
        if repaired is not None and repaired != obj:
            try:
                block = BlockAdapter.validate_python(repaired)
                logger.info(
                    "answer block repaired (%s)",
                    obj.get("type") if isinstance(obj, dict) else "?",
                )
                return block
            except ValidationError:
                pass  # repair didn't help — fall through to the drop path
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
    """
    Extract complete JSON objects from the buffer, preserving a trailing partial.

    Recovers across malformed lines: if the buffer can't be decoded from the
    current position but a newline delimits a complete (garbage) line, that line
    is dropped and parsing continues after it — so one unparseable line never
    swallows the valid blocks that follow. A tail with no newline is treated as
    an incomplete object and kept for the next chunk.
    """
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
            nl = stripped.find("\n")
            if nl == -1:
                return objects, stripped  # incomplete tail — wait for more tokens
            bad = stripped[:nl].strip()
            if bad:
                logger.warning("answer line dropped — unparseable: %r", bad[:200])
            remainder = stripped[nl + 1:]  # skip the garbage line, keep going
            continue

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


# Guaranteed-renderable fallback so a stream never resolves to zero blocks (e.g.
# the model emitted only malformed lines). Built fresh per use — Blocks are cheap.
_FALLBACK_TEXT = (
    "Sorry — I couldn't put together a full answer just now. Could you rephrase "
    "that or add a little more detail?"
)


def _fallback_block() -> Block:
    from graphrag.schemas.blocks import SummaryBlock, SummaryData

    return SummaryBlock(type="summary", data=SummaryData(text=_FALLBACK_TEXT))


def iter_blocks(token_stream: Iterable[str], *, terminal: bool) -> Iterator[Block]:
    """
    Sync: turn a token stream into a stream of validated Blocks (partial recovery).

    Guarantees at least one valid block: if every line failed validation (even
    after repair), a fallback summary is emitted so the client never renders nothing.
    """
    buffer = ""
    yielded = False
    for token in token_stream:
        if not token:
            continue
        buffer += token
        objects, remainder = _extract_complete_objects(buffer)
        if objects:
            for obj in objects:
                kept = _keep(_validate_object(obj), terminal=terminal)
                if kept is not None:
                    yielded = True
                    yield kept
            buffer = remainder

    # Flush trailing content if it forms a complete object; otherwise drop it.
    kept = _keep(validate_line(buffer), terminal=terminal)
    if kept is not None:
        yielded = True
        yield kept

    if not yielded:
        logger.warning("answer stream produced no valid blocks — emitting fallback summary")
        yield _fallback_block()


async def aiter_blocks(token_stream: AsyncIterator[str], *, terminal: bool) -> AsyncIterator[Block]:
    """
    Async: turn an async token stream into a stream of validated Blocks.

    Guarantees at least one valid block (see ``iter_blocks``).
    """
    buffer = ""
    yielded = False
    async for token in token_stream:
        if not token:
            continue
        buffer += token
        objects, remainder = _extract_complete_objects(buffer)
        if objects:
            for obj in objects:
                kept = _keep(_validate_object(obj), terminal=terminal)
                if kept is not None:
                    yielded = True
                    yield kept
            buffer = remainder

    kept = _keep(validate_line(buffer), terminal=terminal)
    if kept is not None:
        yielded = True
        yield kept

    if not yielded:
        logger.warning("answer stream produced no valid blocks — emitting fallback summary")
        yield _fallback_block()


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
