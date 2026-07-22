"""
SOAP note generation — prompt, context assembly, LLM call, and parsing.

Pure helpers (``build_soap_context``, ``parse_soap``) plus two thin entry points
(``generate_soap_async`` for FastAPI, ``generate_soap_sync`` for the CLI) that
differ only in which Gemini call they use. Both build the same context from the
session and parse the model's strict-JSON reply into the four SOAP sections.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from Memory_Layer.session_memory.models import Role, SessionMemory

logger = logging.getLogger(__name__)


SOAP_SYSTEM_PROMPT = """You are a clinical scribe. Turn the conversation between a patient and a medical assistant into a concise, professional SOAP note for a physician.

ABSOLUTE RULES
- Ground EVERYTHING strictly in the conversation provided. NEVER invent symptoms, vitals, measurements, examination findings, test results, diagnoses, or treatments that were not stated.
- If clinically relevant information was not provided, do NOT guess. Name it in the "unavailable" list (e.g. "No vital signs recorded", "Duration of symptoms not stated", "No physical examination performed").
- Preserve uncertainty. Do NOT present an unconfirmed possibility as an established diagnosis. Use hedged, probabilistic language ("likely", "possible", "consistent with").
- Be concise and use standard clinical phrasing. Plain prose per section, no markdown.

SECTIONS
- subjective: The patient-reported story — chief complaint, symptoms, onset/duration, severity, relevant history, current medications, allergies, and anything else the patient explicitly stated. This is the patient's account only.
- objective: ONLY objective data actually present in the conversation — reported measurements (e.g. temperature, BP), test/lab results, or documented findings. If none were provided, say so plainly (e.g. "No objective measurements or examination findings were reported.") and add the gap to "unavailable". Never fabricate an exam.
- assessment: A brief clinical impression based only on the available information, preserving uncertainty. A short differential is fine if the conversation supports it; do not assert a firm diagnosis.
- plan: Summarize ONLY the guidance, next steps, self-care, follow-up, precautions, and escalation advice that were actually given or clearly supported in the conversation. Do not introduce new treatments.

OUTPUT — STRICT JSON, nothing else:
{"subjective":"...","objective":"...","assessment":"...","plan":"...","unavailable":["...","..."]}
No markdown, no code fences, no prose outside the JSON. Every field is required; "unavailable" is a (possibly empty) array of short strings."""


def _render_turn(role: str, content: str) -> str:
    speaker = "Patient" if role == Role.USER.value else "Assistant"
    return f"{speaker}: {content.strip()}"


def build_soap_context(session: SessionMemory) -> str:
    """
    Assemble the full available conversation context for the SOAP prompt.

    Combines the rolling summary of older turns, the recent turn window, and the
    extracted structured clinical state — everything the system knows right now.
    """
    parts: list[str] = []

    if session.summary:
        parts.append("=== EARLIER CONVERSATION (summary) ===\n" + session.summary.strip())

    transcript = [
        _render_turn(t.role, t.content)
        for t in session.recent_turns
        if t.content and t.content.strip()
    ]
    if transcript:
        parts.append("=== CONVERSATION ===\n" + "\n".join(transcript))

    st = session.state
    facts: list[str] = []
    def add(label: str, values: list[str]) -> None:
        clean = [str(v).strip() for v in values if str(v).strip()]
        if clean:
            facts.append(f"{label}: {', '.join(dict.fromkeys(clean))}")

    add("Symptoms", st.symptoms)
    add("Duration", st.duration)
    add("Severity", st.severity)
    add("Medications", st.drugs)
    add("Conditions", st.conditions)
    add("Chronic conditions", st.chronic_conditions)
    add("Allergies", st.allergies)
    if st.demographics:
        demo = ", ".join(f"{k}: {v}" for k, v in st.demographics.items() if v)
        if demo:
            facts.append(f"Demographics: {demo}")
    if facts:
        parts.append("=== EXTRACTED CLINICAL STATE ===\n" + "\n".join(facts))

    if not parts:
        return "(No conversation has taken place yet.)"
    return "\n\n".join(parts)


def parse_soap(raw: str | None) -> dict[str, Any]:
    """
    Parse the model's SOAP reply into the four sections + unavailable list.

    Tolerant of a stray code fence. On any failure returns a safe, non-fabricated
    note that flags the generation problem rather than raising.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text[:4].lower() == "json":
            text = text[4:].strip()

    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("SOAP payload was not a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("SOAP parse failed (%s); returning fallback note", exc)
        return {
            "subjective": "",
            "objective": "",
            "assessment": "",
            "plan": "",
            "unavailable": [
                "The doctor summary could not be generated from this conversation. "
                "Please try again."
            ],
        }

    def s(key: str) -> str:
        val = obj.get(key)
        return val.strip() if isinstance(val, str) else ""

    unavailable = obj.get("unavailable")
    if isinstance(unavailable, list):
        unavailable = [str(u).strip() for u in unavailable if str(u).strip()]
    else:
        unavailable = []

    return {
        "subjective": s("subjective"),
        "objective": s("objective"),
        "assessment": s("assessment"),
        "plan": s("plan"),
        "unavailable": unavailable,
    }


async def generate_soap_async(session: SessionMemory, *, model: str) -> dict[str, Any]:
    """Generate SOAP sections for a session using the async Gemini client."""
    from graphrag.llm.gemini_client import generate_text_async

    context = build_soap_context(session)
    raw = await generate_text_async(
        context,
        model=model,
        system_instruction=SOAP_SYSTEM_PROMPT,
        temperature=0,
        json_mode=True,
    )
    return parse_soap(raw)


def generate_soap_sync(session: SessionMemory, *, model: str) -> dict[str, Any]:
    """Generate SOAP sections for a session using the sync Gemini client (CLI)."""
    from graphrag.llm.gemini_client import generate_text

    context = build_soap_context(session)
    raw = generate_text(
        context,
        model=model,
        system_instruction=SOAP_SYSTEM_PROMPT,
        temperature=0,
        json_mode=True,
    )
    return parse_soap(raw)
