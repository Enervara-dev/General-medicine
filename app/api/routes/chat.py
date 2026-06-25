"""
Chat routes.

POST /chat                — single-shot answer (JSON in, JSON out, prose)
POST /chat/stream         — SSE stream of prose tokens
POST /chat/blocks         — NDJSON stream of typed UI blocks
POST /chat/stream/blocks  — SSE stream of the same typed UI blocks
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.api.deps import ContainerDep
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(req: ChatRequest, request: Request, ctx: ContainerDep) -> ChatResponse:
    """
    Run one full pipeline turn and return the answer + per-stage timing.

    The orchestrator pulls session memory, runs analyzer + retrieval + KG +
    optional episodic context, then asks Gemini for a non-streaming answer.
    """
    request_id = _request_id(request)
    result = await ctx.orchestrator.run(
        query=req.query,
        session_id=req.session_id,
        user_id=req.user_id,
        request_id=request_id,
    )
    return ChatResponse(
        answer=result.answer,
        session_id=result.session_id,
        request_id=result.request_id,
        # The gatekeeper `analysis` block is internal; only surface it when
        # diagnostics are explicitly enabled (default off in production).
        analysis=result.analysis if ctx.settings.EXPOSE_DIAGNOSTICS else None,
        timing_ms=result.timing_ms,
        routing=result.routing,
        followup_questions=result.followup_questions,
    )


@router.post("/chat/blocks")
async def chat_blocks(req: ChatRequest, request: Request, ctx: ContainerDep) -> StreamingResponse:
    """
    NDJSON stream of typed UI blocks (`application/x-ndjson`).

    One JSON block object per line, emitted as generated. The client reads the
    body stream, splits on `\\n`, JSON.parses each complete line, and renders by
    `.type` as it arrives — no text parsing, no waiting for the full response.

    Block types: summary, key_points, bullet_list, follow_up_questions, warning,
    next_steps, condition_list. Each line is `{"type": ..., "data": {...}}`.
    """
    request_id = _request_id(request)
    ndjson = _to_ndjson(
        ctx.orchestrator.stream_blocks(
            query=req.query,
            session_id=req.session_id,
            user_id=req.user_id,
            request_id=request_id,
        )
    )
    return StreamingResponse(
        ndjson,
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering when behind proxy
            "X-Request-ID": request_id,
        },
    )


@router.post("/chat/stream/blocks")
async def chat_stream_blocks(req: ChatRequest, request: Request, ctx: ContainerDep) -> StreamingResponse:
    """
    Server-Sent Events stream of typed UI blocks (`text/event-stream`).

    Same validated blocks as `/chat/blocks`, but SSE-framed for `EventSource`
    and proxy-friendly clients: each block is one `data: <json>\\n\\n` event,
    terminated by a final `data: [DONE]\\n\\n`. The JSON payload is identical
    to an NDJSON line: `{"type": ..., "data": {...}}`.

    Reuses the same `stream_blocks` pipeline as `/chat/blocks` — only the
    transport framing differs.
    """
    request_id = _request_id(request)
    sse_blocks = _blocks_to_sse(
        ctx.orchestrator.stream_blocks(
            query=req.query,
            session_id=req.session_id,
            user_id=req.user_id,
            request_id=request_id,
        )
    )
    return StreamingResponse(
        sse_blocks,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering when behind proxy
            "X-Request-ID": request_id,
        },
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request, ctx: ContainerDep) -> StreamingResponse:
    """
    Server-Sent Events stream.

    Each event is `data: <json>\\n\\n`. Event payload types:
        {"type":"meta","data":{...}}     pipeline metadata before tokens
        {"type":"chunk","data":"..."}    one token / piece of answer text
        {"type":"done","timing_ms":...}  final event; client should close
        {"type":"error","error":{...}}   terminal error
    """
    request_id = _request_id(request)
    sse_stream = _to_sse(
        ctx.orchestrator.stream(
            query=req.query,
            session_id=req.session_id,
            user_id=req.user_id,
            request_id=request_id,
        )
    )
    return StreamingResponse(
        sse_stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering when behind proxy
            "X-Request-ID": request_id,
        },
    )


async def _to_sse(events: AsyncIterator[dict]) -> AsyncIterator[bytes]:
    """Encode dict events as SSE `data: <json>\\n\\n` lines."""
    async for ev in events:
        payload = json.dumps(ev, ensure_ascii=False, default=str)
        yield f"data: {payload}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


async def _to_ndjson(blocks: "AsyncIterator") -> AsyncIterator[bytes]:
    """Encode validated Blocks as NDJSON — one `{...}\\n` line per block."""
    from graphrag.validators.answer_validator import block_to_line

    async for block in blocks:
        yield block_to_line(block).encode("utf-8")


async def _blocks_to_sse(blocks: "AsyncIterator") -> AsyncIterator[bytes]:
    """Encode validated Blocks as SSE — `data: <json>\\n\\n` per block, then [DONE]."""
    async for block in blocks:
        payload = json.dumps(block.model_dump(mode="json"), ensure_ascii=False)
        yield f"data: {payload}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


def _request_id(request: Request) -> str:
    """Read X-Request-ID from headers if the client sent one; mint otherwise."""
    rid = request.headers.get("x-request-id")
    if rid:
        return rid
    rid = uuid.uuid4().hex
    return rid
