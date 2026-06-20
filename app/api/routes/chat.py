"""
Chat routes.

POST /chat         — single-shot answer (JSON in, JSON out)
POST /chat/stream  — SSE stream (phase 3)
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


@router.post("/chat", response_model=ChatResponse)
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
        analysis=result.analysis,
        timing_ms=result.timing_ms,
        routing=result.routing,
        followup_questions=result.followup_questions,
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


def _request_id(request: Request) -> str:
    """Read X-Request-ID from headers if the client sent one; mint otherwise."""
    rid = request.headers.get("x-request-id")
    if rid:
        return rid
    rid = uuid.uuid4().hex
    return rid
