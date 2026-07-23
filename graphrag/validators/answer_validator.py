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
import re
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
    "decision": {"verdict", "rationale"},
    "otc_medications": {"medications"},
    "lab_tests": {"tests"},
    "answer_state": {"show_doctor_summary"},
}

# The five verdicts the decision block accepts, plus common aliases a model
# emits (uppercase / spaced / "urgent" / "unsure"). Anything unmapped is left
# as-is and dropped by strict validation.
_VERDICT_ALIASES: dict[str, str] = {
    "yes": "yes", "y": "yes", "safe": "yes", "true": "yes",
    "no": "no", "n": "no", "unsafe": "no", "false": "no",
    "possibly": "possibly", "maybe": "possibly", "possible": "possibly",
    "depends": "possibly", "sometimes": "possibly",
    "seek_urgent_care": "seek_urgent_care", "urgent": "seek_urgent_care",
    "seek urgent care": "seek_urgent_care", "emergency": "seek_urgent_care",
    "er": "seek_urgent_care", "seek care": "seek_urgent_care",
    "insufficient_information": "insufficient_information",
    "insufficient information": "insufficient_information",
    "insufficient": "insufficient_information", "unknown": "insufficient_information",
    "unsure": "insufficient_information", "need more info": "insufficient_information",
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
    # Remap a LIST payload given under the wrong key onto the block's real list
    # field. Covers the model naming the list after itself or another block —
    # e.g. key_points/{bullet_list:[...]} or key_points/{key_points:[...]} ->
    # key_points/{points:[...]}. Without this the whole block is dropped.
    _LIST_TARGET = {
        "key_points": "points", "next_steps": "steps",
        "follow_up_questions": "questions", "bullet_list": "items",
    }
    if btype in _LIST_TARGET:
        target = _LIST_TARGET[btype]
        if not data.get(target):
            for _k, _v in data.items():
                if _k == target:
                    continue
                if isinstance(_v, list) and _v and all(isinstance(x, str) for x in _v):
                    data = {**data, target: _v}
                    break
    # Same idea for object-list blocks (list-of-dicts under the wrong key),
    # e.g. lab_tests/{investigations:[{...}]} -> lab_tests/{tests:[{...}]}.
    _OBJ_LIST_TARGET = {
        "lab_tests": "tests", "otc_medications": "medications",
        "condition_list": "conditions",
    }
    if btype in _OBJ_LIST_TARGET:
        target = _OBJ_LIST_TARGET[btype]
        if not data.get(target):
            for _k, _v in data.items():
                if _k != target and isinstance(_v, list) and _v and isinstance(_v[0], dict):
                    data = {**data, target: _v}
                    break
    if btype == "summary" and not data.get("text"):
        for alt in ("points", "steps", "questions"):
            if isinstance(data.get(alt), list) and data[alt]:
                data = {**data, "text": " ".join(str(x) for x in data[alt])}
                break

    clean = {k: v for k, v in data.items() if k in allowed}  # strip extra keys

    if btype == "warning":
        sev = str(clean.get("severity", "")).strip().lower()
        clean["severity"] = _SEVERITY_ALIASES.get(sev, "caution")

    if btype == "decision":
        verdict = str(clean.get("verdict", "")).strip().lower()
        # Map the model's phrasing onto a canonical verdict; when it's truly
        # unrecognisable, fall back to the safest non-committal outcome rather
        # than dropping the whole decision block.
        clean["verdict"] = _VERDICT_ALIASES.get(verdict, "insufficient_information")

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
    """
    Apply the follow-up-question rules to a validated block:
    - drop a ``follow_up_questions`` block entirely on a terminal turn, and
    - cap it to ONE question otherwise (project contract: ≤1 question/turn),
      matching the prose path's cap so the model can't over-ask via the block.
    """
    if block is None:
        return None
    # Server-only control blocks (e.g. answer_state) must never come FROM the
    # model — the pipeline injects the authoritative one itself. Drop any the
    # model imitated so a stray one can't leak or crowd out real content.
    from graphrag.schemas.blocks import CONTROL_BLOCK_TYPES

    if block.type in CONTROL_BLOCK_TYPES:
        logger.info("answer block dropped — model emitted control block %r", block.type)
        return None
    if block.type == _TERMINAL_DROP_TYPE:
        if terminal:
            logger.info("answer block dropped — follow_up_questions on terminal turn")
            return None
        if len(block.data.questions) > 1:
            block.data.questions = block.data.questions[:1]
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


# JSON structural/key tokens that are never human-facing content — excluded when
# scavenging readable text out of a broken JSON blob.
_JSON_NONCONTENT: frozenset[str] = frozenset({
    "type", "data", "text", "points", "steps", "questions", "items", "summary",
    "warning", "severity", "next_steps", "key_points", "follow_up_questions",
    "condition_list", "conditions", "name", "likelihood", "description",
    "decision", "verdict", "rationale", "otc_medications", "medications",
    "purpose", "dosage", "caution", "bullet_list", "title", "answer_state",
    "show_doctor_summary", "info", "critical", "most likely", "less likely",
    "possible", "yes", "no", "possibly", "seek_urgent_care",
    "insufficient_information",
})


def _text_from_jsonish(raw: str) -> str:
    """
    Extract human-readable text from a broken/truncated JSON answer.

    Pulls the string literals that look like prose (contain a space, reasonably
    long, not a schema key) and joins the first few. Used only as a last resort
    when no block validated — turns a truncated response into a usable summary
    instead of the generic apology.
    """
    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', raw)
    vals: list[str] = []
    for s in strings:
        cleaned = s.strip()
        if len(cleaned) > 15 and " " in cleaned and cleaned.lower() not in _JSON_NONCONTENT:
            # Unescape the common JSON escapes so the text reads naturally.
            cleaned = (
                cleaned.replace("\\n", " ").replace("\\t", " ").replace('\\"', '"').replace("\\/", "/")
            )
            vals.append(cleaned)
    if not vals:
        return ""
    joined = " ".join(vals[:4])
    return re.sub(r"\s+", " ", joined).strip()


def _salvage_prose(raw: str, *, terminal: bool) -> Block | None:
    """
    Rescue a plain-prose answer into a real block.

    The model sometimes ignores the NDJSON contract on thin turns and returns a
    normal sentence instead of a JSON block. Rather than discard that (useful)
    text for the generic apology, wrap it: a clarifying question becomes a
    ``follow_up_questions`` block; anything else becomes a ``summary``. Returns
    None when there's nothing usable (empty, or still-JSON-looking garbage).
    """
    text = (raw or "").strip()
    # Strip a markdown code fence the model may have wrapped the text in.
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text[:4].lower() == "json":
            text = text[4:].strip()
    if not text:
        return None
    # Unparseable JSON (truncated / malformed) reaches here when NO block
    # validated. Rather than the generic apology, pull the human-readable string
    # values out of the broken blob and render those. Only if nothing usable
    # comes back do we fall through to the generic fallback.
    if text[:1] in "{[":
        extracted = _text_from_jsonish(text)
        if not extracted:
            return None
        text = extracted

    from graphrag.schemas.blocks import (
        FollowUpQuestionsBlock,
        FollowUpQuestionsData,
        SummaryBlock,
        SummaryData,
    )

    # A clarifying question on a non-closing turn → a single follow-up chip.
    # Keep only up to the last '?' so trailing filler doesn't ride along.
    if not terminal and "?" in text:
        question = text[: text.rfind("?") + 1].strip()
        if question:
            logger.info("answer prose salvaged into a follow_up_questions block")
            return FollowUpQuestionsBlock(
                type="follow_up_questions",
                data=FollowUpQuestionsData(questions=[question]),
            )
    logger.info("answer prose salvaged into a summary block")
    return SummaryBlock(type="summary", data=SummaryData(text=text))


def iter_blocks(token_stream: Iterable[str], *, terminal: bool) -> Iterator[Block]:
    """
    Sync: turn a token stream into a stream of validated Blocks (partial recovery).

    Guarantees at least one valid block: if every line failed validation (even
    after repair), a fallback summary is emitted so the client never renders nothing.
    """
    buffer = ""
    raw_all = ""
    yielded = False
    for token in token_stream:
        if not token:
            continue
        buffer += token
        raw_all += token
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
        # The model returned no valid block — salvage plain prose into one before
        # resorting to the generic apology.
        salvaged = _keep(_salvage_prose(raw_all, terminal=terminal), terminal=terminal)
        if salvaged is not None:
            yield salvaged
        else:
            logger.warning("answer stream produced no valid blocks — emitting fallback summary")
            yield _fallback_block()


async def aiter_blocks(token_stream: AsyncIterator[str], *, terminal: bool) -> AsyncIterator[Block]:
    """
    Async: turn an async token stream into a stream of validated Blocks.

    Guarantees at least one valid block (see ``iter_blocks``).
    """
    buffer = ""
    raw_all = ""
    yielded = False
    async for token in token_stream:
        if not token:
            continue
        buffer += token
        raw_all += token
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
        salvaged = _keep(_salvage_prose(raw_all, terminal=terminal), terminal=terminal)
        if salvaged is not None:
            yield salvaged
        else:
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
        elif t == "decision":
            parts.append(f"[{d.verdict.replace('_', ' ').upper()}] {d.rationale}")
        elif t == "otc_medications":
            parts.append("OTC options:")
            for m in d.medications:
                bits = [m.name]
                if m.purpose:
                    bits.append(f"— {m.purpose}")
                if m.dosage:
                    bits.append(f"({m.dosage})")
                if m.caution:
                    bits.append(f"[caution: {m.caution}]")
                parts.append("  " + " ".join(bits))
        elif t == "lab_tests":
            parts.append("Recommended tests:")
            for test in d.tests:
                bits = [test.name]
                if test.reason:
                    bits.append(f"— {test.reason}")
                if test.urgency:
                    bits.append(f"[{test.urgency}]")
                parts.append("  " + " ".join(bits))
    return "\n".join(parts).strip()
