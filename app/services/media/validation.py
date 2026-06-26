"""
Upload validation — the only place that decides whether bytes are acceptable.

Isolated from business logic on purpose: the endpoint and the pipeline both call
``validate_image`` and never re-implement size/MIME checks. Content sniffing
(magic bytes) guards against a spoofed ``Content-Type`` header.
"""

from __future__ import annotations

from app.services.media.types import MediaValidationError, UploadedImage

# Leading magic bytes per supported MIME type. Keeps a declared image/png that
# is actually something else from reaching the model.
_MAGIC: dict[str, tuple[bytes, ...]] = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
}


def _parse_allowed(raw: str) -> set[str]:
    return {m.strip().lower() for m in (raw or "").split(",") if m.strip()}


def validate_image(
    *,
    data: bytes,
    mime_type: str | None,
    filename: str | None,
    max_bytes: int,
    allowed_mime_types: str,
) -> UploadedImage:
    """
    Validate raw upload bytes and return a typed ``UploadedImage``.

    Raises ``MediaValidationError`` (caller maps to HTTP 400/413) when the file
    is empty, oversized, of a disallowed MIME type, or whose content does not
    match its declared type.
    """
    if not data:
        raise MediaValidationError("Uploaded file is empty.")

    size = len(data)
    if size > max_bytes:
        raise MediaValidationError(
            f"File is {size} bytes; the limit is {max_bytes} bytes."
        )

    normalized = (mime_type or "").split(";", 1)[0].strip().lower()
    allowed = _parse_allowed(allowed_mime_types)
    if normalized not in allowed:
        raise MediaValidationError(
            f"Unsupported content type '{mime_type or 'unknown'}'. "
            f"Allowed: {', '.join(sorted(allowed)) or '(none configured)'}."
        )

    magics = _MAGIC.get(normalized)
    if magics and not any(data.startswith(m) for m in magics):
        raise MediaValidationError(
            f"File content does not match declared type '{normalized}'."
        )

    return UploadedImage(
        data=data,
        mime_type=normalized,
        size_bytes=size,
        filename=filename,
    )


__all__ = ["validate_image"]
