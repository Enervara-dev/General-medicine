"""
Shared Google Gemini client wrapper.

All Gemini calls in this codebase flow through here so retry policy,
model defaults, and configuration live in exactly one place.

Synchronous helpers (`generate_text`, `generate_stream`) are used by the
ingestion-side scripts. Async helpers (`generate_text_async`) are used by
the memory subsystem which is already on httpx/async.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional, Protocol, Sequence, runtime_checkable

from google import genai
from google.genai import types

from graphrag.config.settings import settings

logger = logging.getLogger(__name__)


@runtime_checkable
class MediaLike(Protocol):
    """
    Minimal shape this client needs to attach a non-text part to a request.

    Any object exposing raw ``data`` bytes and an IANA ``mime_type`` qualifies
    (e.g. ``app.services.media.types.MediaPart``). Keeping it a structural
    Protocol means the LLM boundary stays decoupled from the media layer and is
    trivial to extend to audio/video parts later.
    """

    @property
    def data(self) -> bytes: ...

    @property
    def mime_type(self) -> str: ...


def _build_contents(prompt: str, media: Optional[Sequence[MediaLike]]) -> Any:
    """
    Build the Gemini ``contents`` argument.

    Text-only (``media`` empty/None) returns the bare prompt string, preserving
    the exact behaviour every existing call site relies on. With media, returns
    ``[<part>, ..., prompt]`` so the model sees the image(s) then the prompt.
    """
    if not media:
        return prompt
    parts: list[Any] = [
        types.Part.from_bytes(data=m.data, mime_type=m.mime_type) for m in media
    ]
    parts.append(prompt)
    return parts

# Main model used everywhere in the project. Currently gemini-2.5-flash-lite
# for every call site. If Lite hits free-tier quota again, swap to
# "gemini-2.5-flash" — every override-aware setting in
# graphrag.config.settings falls back to this constant.
DEFAULT_MODEL = "gemini-2.5-flash-lite"

# Kept for backward compatibility — points at the same model. Existing
# callers that reference DEFAULT_LITE_MODEL keep working without changes.
DEFAULT_LITE_MODEL = DEFAULT_MODEL

_client: Optional[genai.Client] = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Add it to your .env "
                "(see .env.example) or process environment."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _build_config(
    *,
    system_instruction: Optional[str] = None,
    temperature: Optional[float] = None,
    json_mode: bool = False,
    response_schema: Any = None,
    max_output_tokens: Optional[int] = None,
) -> types.GenerateContentConfig:
    kwargs: dict[str, Any] = {}
    if system_instruction is not None:
        kwargs["system_instruction"] = system_instruction
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    if json_mode or response_schema is not None:
        kwargs["response_mime_type"] = "application/json"
    if response_schema is not None:
        kwargs["response_schema"] = response_schema
    return types.GenerateContentConfig(**kwargs)


def generate_text(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    system_instruction: Optional[str] = None,
    temperature: Optional[float] = None,
    json_mode: bool = False,
    response_schema: Any = None,
    max_output_tokens: Optional[int] = None,
    media: Optional[Sequence[MediaLike]] = None,
) -> str:
    """
    Synchronous one-shot generation. Returns the model's text (`""` on empty).

    Pass ``media`` (image/document parts) to make the call multimodal; omit it
    and the request is identical to the previous text-only behaviour.
    """
    client = get_client()
    config = _build_config(
        system_instruction=system_instruction,
        temperature=temperature,
        json_mode=json_mode,
        response_schema=response_schema,
        max_output_tokens=max_output_tokens,
    )
    resp = client.models.generate_content(
        model=model,
        contents=_build_contents(prompt, media),
        config=config,
    )
    return resp.text or ""


async def generate_text_async(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    system_instruction: Optional[str] = None,
    temperature: Optional[float] = None,
    json_mode: bool = False,
    response_schema: Any = None,
    max_output_tokens: Optional[int] = None,
    media: Optional[Sequence[MediaLike]] = None,
) -> str:
    """Async variant of `generate_text`. Pass ``media`` for multimodal input."""
    client = get_client()
    config = _build_config(
        system_instruction=system_instruction,
        temperature=temperature,
        json_mode=json_mode,
        response_schema=response_schema,
        max_output_tokens=max_output_tokens,
    )
    resp = await client.aio.models.generate_content(
        model=model,
        contents=_build_contents(prompt, media),
        config=config,
    )
    return resp.text or ""


def generate_stream(
    *,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    system_instruction: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Iterator[str]:
    """Yield text chunks as they arrive from the model."""
    client = get_client()
    config = _build_config(
        system_instruction=system_instruction,
        temperature=temperature,
    )
    stream = client.models.generate_content_stream(
        model=model,
        contents=user_prompt,
        config=config,
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text
