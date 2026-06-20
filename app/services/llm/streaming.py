"""
Sync-iterator → async-generator bridge for the Gemini streaming API.

The `google-genai` SDK exposes `client.models.generate_content_stream(...)`
as a SYNC iterator. FastAPI's StreamingResponse needs an async generator.
We run the producer on a thread and bridge chunks through an asyncio.Queue.

This is the only place that's allowed to mix sync iteration with async
consumption. Everything else in app/ is fully async.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from graphrag.llm.gemini_client import generate_stream

logger = logging.getLogger(__name__)

# Sentinel signalling the producer finished cleanly.
_END = object()


async def stream_gemini_tokens(
    *,
    model: str,
    system_instruction: str | None,
    user_prompt: str,
    temperature: float | None = None,
    queue_max: int = 64,
) -> AsyncIterator[str]:
    """
    Yield Gemini tokens one chunk at a time as an async generator.

    The sync `generate_stream` runs on a background thread; results land in
    an asyncio.Queue that this generator consumes. The thread is allowed to
    raise — the exception is re-raised on the consumer side so the caller's
    `try/except` semantics still work.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max)

    def producer() -> None:
        try:
            for chunk in generate_stream(
                model=model,
                system_instruction=system_instruction,
                user_prompt=user_prompt,
                temperature=temperature,
            ):
                # Block the worker thread (not the event loop) when the
                # consumer can't keep up — Queue.put is sync from threads.
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
        except BaseException as exc:  # noqa: BLE001 — propagate to consumer
            asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
            return
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(_END), loop)

    thread_task = loop.run_in_executor(None, producer)

    try:
        while True:
            item = await queue.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        # Ensure the background thread always finishes — wait briefly.
        try:
            await asyncio.wait_for(asyncio.shield(thread_task), timeout=0.5)
        except (asyncio.TimeoutError, Exception):
            pass
