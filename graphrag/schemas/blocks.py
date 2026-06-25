"""
Typed UI blocks — the wire contract for STAGE-4 answers.

Every user-facing response is a stream of these blocks, one per NDJSON line.
A block is ``{"type": <literal>, "data": {...}}``; ``Block`` is the discriminated
union over ``type`` so a single line can be validated as exactly one block.

Mirrors the chunker's strict-schema philosophy (``chunking/schemas/models.py``):
Pydantic v2, ``extra="forbid"`` everywhere, non-empty lists enforced with
``min_length=1``. ``BLOCK_TYPES`` is the single source of truth for the set of
valid types — the prompt's OUTPUT_CONTRACT and the validator both read from it.

``AnswerResponse`` wraps a full block list for non-streaming consumers (tests,
batch tooling); the streaming paths emit ``Block`` instances line by line.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

# Forbid unknown keys on every model so a malformed/hallucinated line fails
# validation (and is dropped by the per-line validator) instead of silently
# passing through with junk fields.
_STRICT = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Per-block data payloads
# ---------------------------------------------------------------------------

class SummaryData(BaseModel):
    model_config = _STRICT
    text: str = Field(min_length=1)


class KeyPointsData(BaseModel):
    model_config = _STRICT
    points: list[str] = Field(min_length=1)


class BulletListData(BaseModel):
    model_config = _STRICT
    title: Optional[str] = None
    items: list[str] = Field(min_length=1)


class FollowUpQuestionsData(BaseModel):
    model_config = _STRICT
    questions: list[str] = Field(min_length=1)


class WarningData(BaseModel):
    model_config = _STRICT
    text: str = Field(min_length=1)
    severity: Literal["info", "caution", "critical"]


class NextStepsData(BaseModel):
    model_config = _STRICT
    steps: list[str] = Field(min_length=1)


class Condition(BaseModel):
    model_config = _STRICT
    name: str = Field(min_length=1)
    likelihood: Optional[str] = None
    description: Optional[str] = None


class ConditionListData(BaseModel):
    model_config = _STRICT
    conditions: list[Condition] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Block envelopes (type-discriminated)
# ---------------------------------------------------------------------------

class SummaryBlock(BaseModel):
    model_config = _STRICT
    type: Literal["summary"]
    data: SummaryData


class KeyPointsBlock(BaseModel):
    model_config = _STRICT
    type: Literal["key_points"]
    data: KeyPointsData


class BulletListBlock(BaseModel):
    model_config = _STRICT
    type: Literal["bullet_list"]
    data: BulletListData


class FollowUpQuestionsBlock(BaseModel):
    model_config = _STRICT
    type: Literal["follow_up_questions"]
    data: FollowUpQuestionsData


class WarningBlock(BaseModel):
    model_config = _STRICT
    type: Literal["warning"]
    data: WarningData


class NextStepsBlock(BaseModel):
    model_config = _STRICT
    type: Literal["next_steps"]
    data: NextStepsData


class ConditionListBlock(BaseModel):
    model_config = _STRICT
    type: Literal["condition_list"]
    data: ConditionListData


# Discriminated union — validate one line as exactly one of these by its `type`.
Block = Annotated[
    Union[
        SummaryBlock,
        KeyPointsBlock,
        BulletListBlock,
        FollowUpQuestionsBlock,
        WarningBlock,
        NextStepsBlock,
        ConditionListBlock,
    ],
    Field(discriminator="type"),
]

# Reusable adapter so callers (the validator) don't rebuild it per line.
BlockAdapter: TypeAdapter[Block] = TypeAdapter(Block)

# Single source of truth for the valid type set. Keep in sync with the union
# above; the prompt layer and validator both consume this tuple.
BLOCK_TYPES: tuple[str, ...] = (
    "summary",
    "key_points",
    "bullet_list",
    "follow_up_questions",
    "warning",
    "next_steps",
    "condition_list",
)


class AnswerResponse(BaseModel):
    """Full block list for non-streaming consumers. Streaming emits Block-by-Block."""

    model_config = _STRICT
    blocks: list[Block]


__all__ = [
    "BLOCK_TYPES",
    "AnswerResponse",
    "Block",
    "BlockAdapter",
    # data payloads
    "SummaryData",
    "KeyPointsData",
    "BulletListData",
    "FollowUpQuestionsData",
    "WarningData",
    "NextStepsData",
    "ConditionListData",
    "Condition",
    # envelopes
    "SummaryBlock",
    "KeyPointsBlock",
    "BulletListBlock",
    "FollowUpQuestionsBlock",
    "WarningBlock",
    "NextStepsBlock",
    "ConditionListBlock",
]
