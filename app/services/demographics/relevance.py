"""
Demographic relevance selection + prompt rendering.

Demographics are NOT dumped into every prompt. This module decides, per turn,
WHICH of the seven fields are actually useful for the current medical query, and
renders only those. Selection is deterministic (intent + keyword rules) — no
extra LLM call, so it adds no latency/cost and is fully unit-testable.

Field bundles
    BODY      age, sex, height_cm, weight_kg, bmi   (weight/BMI/dosing/nutrition)
    AGE_SEX   age, sex                              (standard clinical context)
    LOCATION  state, city                           (endemic/regional/local care)

Policy summary
    * Symptom / diagnosis / risk / medication / treatment / follow-up intents get
      AGE_SEX by default — the baseline context any clinician wants.
    * Weight/BMI/nutrition/weight-based-dosing language pulls in the full BODY
      bundle.
    * Sex-specific topics (pregnancy, contraception, prostate, …) ensure sex+age.
    * Age-specific topics (pediatric, geriatric, screening) ensure age.
    * Location-dependent topics (endemic disease, "in my area", local care) pull
      in state/city.
    * Generic knowledge / educational / comparison queries with no personal cue
      get NOTHING — demographics must not colour a textbook answer.
Only fields that are BOTH selected AND populated are rendered.
"""

from __future__ import annotations

from typing import Any, Optional

from app.services.demographics.types import DemographicContextV1

# Intents where age + sex are baseline useful clinical context.
_CLINICAL_INTENTS = frozenset({
    "symptom_query", "diagnosis_query", "risk_assessment",
    "medication_query", "treatment_query", "followup_query",
})

_BODY = ("age", "sex", "height_cm", "weight_kg", "bmi")
_AGE_SEX = ("age", "sex")
_LOCATION = ("state", "city")

# Keyword triggers (substring match on query + rewritten + entities, lowercased).
_BODY_KW = (
    "bmi", "body mass", "weight", "overweight", "underweight", "obes", "obesity",
    "lose weight", "gain weight", "slim", "diet", "calorie", "calories",
    "nutrition", "protein intake", "mg/kg", "per kg", "dose", "dosage", "how much should i take",
)
_LOCATION_KW = (
    "in my area", "near me", "my city", "my state", "my region", "around here",
    "endemic", "outbreak", "epidemic", "climate", "regional", "local ", "locally",
    "vaccination schedule", "malaria", "dengue", "chikungunya", "typhoid",
    "water quality", "air pollution", "altitude",
)
_SEX_KW = (
    "pregnan", "contracept", "birth control", "menstru", "period", "menopaus",
    "prostate", "testosteron", "erectile", "breast", "cervical", "ovarian",
    "uterine", "fertility", "gynec", "pap smear", "mammogram", "psa test",
)
_AGE_KW = (
    "child", "infant", "toddler", "pediatric", "paediatric", "baby", "newborn",
    "elderly", "geriatric", "senior", "old age", "age-appropriate",
    "screening", "for my age", "at my age",
)
# Explicit personalisation cues — the user is asking about THEMSELVES, so even an
# otherwise-generic query warrants baseline age/sex.
_PERSONAL_KW = (
    "should i", "can i", "do i", "am i", "for me", "my ", "i have", "i am",
    "i'm ", "in my case", "given my",
)
# Purely educational intents that default to NO demographics unless a cue fires.
_EDUCATIONAL_INTENTS = frozenset({
    "condition_explanation", "prognosis_query", "prevention_query",
    "lifestyle_query", "procedure_query", "comparison_query", "unknown", "greeting",
})


def _haystack(query: str, analysis: Optional[dict[str, Any]]) -> str:
    parts = [query or ""]
    a = analysis or {}
    rew = a.get("rewritten_query")
    if isinstance(rew, str):
        parts.append(rew)
    ents = a.get("medical_entities") or {}
    if isinstance(ents, dict):
        for v in ents.values():
            if isinstance(v, list):
                parts.extend(str(x) for x in v)
    return " ".join(parts).lower()


def select_relevant_fields(
    demo: DemographicContextV1,
    analysis: Optional[dict[str, Any]],
    query: str,
) -> set[str]:
    """Return the set of demographic field names relevant to this turn."""
    intent = str((analysis or {}).get("intent") or "unknown").strip().lower()
    text = _haystack(query, analysis)
    fields: set[str] = set()

    has_personal_cue = any(k in text for k in _PERSONAL_KW)

    # Baseline clinical context.
    if intent in _CLINICAL_INTENTS:
        fields.update(_AGE_SEX)

    # Topic-driven bundles (independent of intent).
    if any(k in text for k in _BODY_KW):
        fields.update(_BODY)
    if any(k in text for k in _SEX_KW):
        fields.update(("sex", "age"))
    if any(k in text for k in _AGE_KW):
        fields.add("age")
    if any(k in text for k in _LOCATION_KW):
        fields.update(_LOCATION)

    # Educational / generic knowledge: no demographics unless the user made it
    # personal or a topic bundle above already fired.
    if intent in _EDUCATIONAL_INTENTS and not fields:
        if has_personal_cue:
            fields.update(_AGE_SEX)
        # else: leave empty — a textbook answer must not be coloured by demographics.

    return fields


def _render_lines(demo: DemographicContextV1, fields: set[str]) -> list[str]:
    lines: list[str] = []
    if "age" in fields and demo.age is not None:
        lines.append(f"Age: {demo.age}")
    if "sex" in fields and demo.sex:
        lines.append(f"Sex: {demo.sex}")
    if "height_cm" in fields and demo.height_cm is not None:
        lines.append(f"Height: {demo.height_cm} cm")
    if "weight_kg" in fields and demo.weight_kg is not None:
        lines.append(f"Weight: {demo.weight_kg} kg")
    if "bmi" in fields and demo.bmi is not None:
        lines.append(f"BMI: {demo.bmi}")
    if {"state", "city"} & fields:
        loc = ", ".join(p for p in (demo.city if "city" in fields else None,
                                     demo.state if "state" in fields else None) if p)
        if loc:
            lines.append(f"Location: {loc}")
    return lines


def render_demographic_block(
    demo: Optional[DemographicContextV1],
    analysis: Optional[dict[str, Any]],
    query: str,
) -> str:
    """
    Render the relevant, populated demographics as a prompt block, or "".

    Returns an empty string when there is nothing relevant to inject — the caller
    then adds no demographic context at all.
    """
    if demo is None or demo.is_empty():
        return ""
    fields = select_relevant_fields(demo, analysis, query)
    if not fields:
        return ""
    lines = _render_lines(demo, fields)
    if not lines:
        return ""
    body = "\n".join(lines)
    return (
        "=== PATIENT DEMOGRAPHICS (authoritative, current) ===\n"
        f"{body}\n"
        "These are established patient facts — use them when clinically relevant "
        "and never ask the patient to repeat them."
    )


__all__ = ["select_relevant_fields", "render_demographic_block"]
