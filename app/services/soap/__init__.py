"""
Doctor-facing SOAP note generation.

A SOAP note is produced ON DEMAND (when the patient taps "Show this to your
doctor") from the latest full conversation context — never precomputed, always
regenerated so new patient information is reflected. Strictly grounded in the
conversation; missing clinical information is named, never fabricated.
"""

from app.services.soap.generator import (
    SOAP_SYSTEM_PROMPT,
    build_soap_context,
    generate_soap_async,
    generate_soap_sync,
    parse_soap,
)

__all__ = [
    "SOAP_SYSTEM_PROMPT",
    "build_soap_context",
    "generate_soap_async",
    "generate_soap_sync",
    "parse_soap",
]
