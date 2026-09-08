"""
Health-profile relevance selection and prompt rendering.

The patient's whole active profile arrives on every request. Very little of it
belongs in any given prompt: a question about what diabetes is should not be
answered in the shadow of the asker's own surgeries, and a full profile dumped
into every turn buries the parts that matter and spends context on the parts
that do not.

Selection is deterministic, driven by the gatekeeper's intent plus keyword
bundles over the query, exactly as the demographics selector works. There is no
extra model call, so this adds no latency and is fully unit-testable.

Sections
    LIFESTYLE    diet, exercise, alcohol, smoking, sleep, water
    WELLBEING    the five 1-10 ratings and relaxation practices
    ALLERGIES    active allergies, plus a confirmed-absence timestamp
    MEDICATIONS  active medications
    CONDITIONS   active conditions, with catalogue display names
    SURGERIES    active past surgeries

Policy summary
    * Medication questions pull MEDICATIONS, CONDITIONS and ALLERGIES together:
      "can I take this" is unanswerable without all three.
    * Symptom, diagnosis and treatment intents pull CONDITIONS and MEDICATIONS,
      the standing clinical picture any clinician would want.
    * Topic keywords pull their own section regardless of intent.
    * A direct "what X do I have" pulls exactly that section.
    * Educational or generic questions with no personal cue pull NOTHING.
"""

from __future__ import annotations

from typing import Any

from app.schemas.chat import HealthProfileEnvelope

LIFESTYLE = "lifestyle"
WELLBEING = "wellbeing"
ALLERGIES = "allergies"
MEDICATIONS = "medications"
CONDITIONS = "conditions"
SURGERIES = "surgeries"

ALL_SECTIONS = (LIFESTYLE, WELLBEING, ALLERGIES, MEDICATIONS, CONDITIONS, SURGERIES)

# Intents where the standing clinical picture is baseline useful context.
_CLINICAL_INTENTS = frozenset({
    "symptom_query", "diagnosis_query", "risk_assessment", "treatment_query",
    "followup_query",
})
# Intents about medicines specifically, where the interaction triad applies.
_MEDICATION_INTENTS = frozenset({"medication_query"})

# Purely educational intents: no profile unless a cue below fires.
_EDUCATIONAL_INTENTS = frozenset({
    "condition_explanation", "prognosis_query", "prevention_query",
    "lifestyle_query", "procedure_query", "comparison_query", "unknown", "greeting",
})

_MEDICATION_KW = (
    "medicine", "medication", "medicines", "medications", "drug", "drugs",
    "tablet", "pill", "capsule", "syrup", "injection", "prescri", "dose",
    "dosage", "mg", "interaction", "interact", "take this", "take it",
    "can i take", "safe to take", "combine", "together with",
)
_ALLERGY_KW = (
    "allerg", "reaction", "rash", "hives", "anaphyla", "intoleran",
    "sensitivity", "side effect", "adverse",
)
_CONDITION_KW = (
    "condition", "conditions", "diagnos", "chronic", "history", "existing",
    "i have", "i've had", "suffer", "my illness", "comorbid",
)
_LIFESTYLE_KW = (
    "diet", "eat", "eating", "food", "nutrition", "calorie", "calories",
    "alcohol", "drink", "drinking", "smoke", "smoking", "tobacco", "cigarette",
    "exercise", "workout", "gym", "physical activity", "sleep", "water",
    "hydration", "lifestyle", "habit",
)
_WELLBEING_KW = (
    "stress", "stressed", "mood", "anxious", "anxiety", "depress", "sad",
    "tired", "fatigue", "exhaust", "energy", "burnout", "burn out",
    "mental health", "relax", "wellbeing", "well-being", "motivation",
)
_SURGERY_KW = (
    "surgery", "surgeries", "surgical", "operation", "operated", "procedure",
    "implant", "transplant", "stitches", "post-op", "postop",
)

# Direct "what X do I have / have I had" asks map to exactly one section.
_DIRECT_SECTION_ASKS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("my allerg", "any allerg", "allergic to", "do i have allerg"), ALLERGIES),
    (("my medication", "my medicine", "what am i taking", "currently taking",
      "am i on any", "my prescri"), MEDICATIONS),
    (("my condition", "my diagnos", "what conditions", "do i have any condition"),
     CONDITIONS),
    (("my surgeries", "my surgery", "surgeries have i", "operations have i",
      "past surgery", "had surgery"), SURGERIES),
    (("my lifestyle", "my diet", "my habits"), LIFESTYLE),
    (("my mood", "my stress", "my wellbeing", "my energy"), WELLBEING),
)

_PERSONAL_KW = (
    "should i", "can i", "do i", "am i", "for me", "my ", "i have", "i am",
    "i'm ", "in my case", "given my",
)


def _haystack(query: str, analysis: dict[str, Any] | None) -> str:
    """Query + rewritten query + extracted entities, lowercased."""
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


def select_relevant_sections(
    profile: HealthProfileEnvelope | None,
    analysis: dict[str, Any] | None,
    query: str,
) -> set[str]:
    """Return the profile section names relevant to this turn. Never raises."""
    if profile is None:
        return set()

    intent = str((analysis or {}).get("intent") or "unknown").strip().lower()
    text = _haystack(query, analysis)
    sections: set[str] = set()

    # Direct asks first: "what allergies do I have" wants that section, and the
    # answer is the list itself.
    for keywords, section in _DIRECT_SECTION_ASKS:
        if any(k in text for k in keywords):
            sections.add(section)

    # Medication questions need the interaction triad. Answering "can I take
    # this" from the medication list alone ignores both the conditions being
    # treated and anything the patient reacts to.
    if intent in _MEDICATION_INTENTS:
        sections.update((MEDICATIONS, CONDITIONS, ALLERGIES))

    # Standing clinical picture for symptom/diagnosis/treatment reasoning.
    if intent in _CLINICAL_INTENTS:
        sections.update((CONDITIONS, MEDICATIONS))

    # Topic keywords, independent of intent.
    if any(k in text for k in _MEDICATION_KW):
        sections.update((MEDICATIONS, ALLERGIES, CONDITIONS))
    if any(k in text for k in _ALLERGY_KW):
        sections.add(ALLERGIES)
    if any(k in text for k in _CONDITION_KW):
        sections.add(CONDITIONS)
    if any(k in text for k in _LIFESTYLE_KW):
        sections.add(LIFESTYLE)
    if any(k in text for k in _WELLBEING_KW):
        sections.update((WELLBEING, CONDITIONS))
    if any(k in text for k in _SURGERY_KW):
        sections.add(SURGERIES)

    # Educational / generic: nothing unless the user made it personal.
    if intent in _EDUCATIONAL_INTENTS and not sections:
        if any(k in text for k in _PERSONAL_KW):
            sections.update((CONDITIONS, MEDICATIONS))
        # else: leave empty. A textbook answer is not about this patient.

    return sections


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt_medication(m: Any) -> str:
    parts = [m.medication_name]
    if m.dose_amount is not None and m.dose_unit:
        amount = int(m.dose_amount) if float(m.dose_amount).is_integer() else m.dose_amount
        parts.append(f"{amount} {m.dose_unit}")
    if m.frequency_count is not None and m.frequency_period:
        parts.append(f"{m.frequency_count}x {m.frequency_period.replace('_', ' ')}")
    if m.time_of_day:
        parts.append("/".join(m.time_of_day))
    if m.food_relation:
        parts.append(m.food_relation.replace("_", " "))
    if m.course_type:
        parts.append(f"({m.course_type.replace('_', ' ')})")
    line = " ".join(str(p) for p in parts)
    if m.reason_or_condition:
        line += f" - for {m.reason_or_condition}"
    return line


def _fmt_allergy(a: Any) -> str:
    line = a.allergen_name
    detail = []
    if a.reaction_types:
        detail.append(", ".join(a.reaction_types))
    if a.severity:
        detail.append(a.severity.replace("_", " "))
    if detail:
        line += " - " + "; ".join(detail)
    return line


def _fmt_condition(c: Any) -> str:
    line = c.display_name or c.condition_code
    detail = []
    if c.since_exact_date:
        detail.append(f"since {c.since_exact_date}")
    elif c.since_bucket:
        detail.append(f"since {c.since_bucket.replace('_', ' ')}")
    if c.currently_troubling:
        detail.append(f"currently troubling: {c.currently_troubling}")
    if c.on_medication:
        detail.append("on medication")
    if detail:
        line += " (" + "; ".join(detail) + ")"
    return line


def _fmt_surgery(s: Any) -> str:
    line = s.surgery_name
    detail = []
    if s.performed_year:
        detail.append(str(s.performed_year))
    if s.current_status:
        detail.append(s.current_status.replace("_", " "))
    if s.reason:
        detail.append(f"for {s.reason}")
    if detail:
        line += " (" + "; ".join(detail) + ")"
    return line


_LIFESTYLE_LABELS = (
    ("diet", "Diet"), ("exercise", "Exercise"), ("alcohol", "Alcohol"),
    ("smoking", "Smoking"), ("sleep_hours", "Sleep (hours)"),
    ("water_cups", "Water (cups/day)"),
)
_WELLBEING_LABELS = (
    ("overall_mood", "Mood"), ("stress_level", "Stress"),
    ("energy_level", "Energy"), ("social_connectedness", "Social connectedness"),
    ("work_academic_pressure", "Work/academic pressure"),
    ("relaxation_practices", "Relaxation practices"),
)


def render_health_profile_block(
    profile: HealthProfileEnvelope | None,
    analysis: dict[str, Any] | None,
    query: str,
) -> str:
    """
    Render the relevant, populated profile sections as a prompt block, or "".

    Only selected sections appear, and within them only populated fields. An
    empty string means no profile context is added at all.
    """
    if profile is None:
        return ""
    sections = select_relevant_sections(profile, analysis, query)
    if not sections:
        return ""

    out: list[str] = []

    if CONDITIONS in sections and profile.conditions is not None:
        condition_items = profile.conditions.items
        if condition_items:
            out.append("Conditions:")
            out.extend(f"- {_fmt_condition(c)}" for c in condition_items)
        elif profile.conditions.none_confirmed_at:
            # Distinct from silence: the patient stated there are none.
            out.append("Conditions: patient has confirmed no known conditions.")

    if MEDICATIONS in sections and profile.medications:
        out.append("Medications:")
        out.extend(f"- {_fmt_medication(m)}" for m in profile.medications)

    if ALLERGIES in sections and profile.allergies is not None:
        allergy_items = profile.allergies.items
        if allergy_items:
            out.append("Allergies:")
            out.extend(f"- {_fmt_allergy(a)}" for a in allergy_items)
        elif profile.allergies.none_confirmed_at:
            out.append("Allergies: patient has confirmed no known allergies.")

    if SURGERIES in sections and profile.surgeries:
        out.append("Past surgeries:")
        out.extend(f"- {_fmt_surgery(s)}" for s in profile.surgeries)

    if LIFESTYLE in sections and profile.lifestyle is not None:
        lines = [
            f"- {label}: {getattr(profile.lifestyle, attr)}"
            for attr, label in _LIFESTYLE_LABELS
            if getattr(profile.lifestyle, attr) is not None
        ]
        if lines:
            out.append("Lifestyle:")
            out.extend(lines)

    if WELLBEING in sections and profile.wellbeing is not None:
        lines = [
            f"- {label}: {getattr(profile.wellbeing, attr)}"
            for attr, label in _WELLBEING_LABELS
            if getattr(profile.wellbeing, attr) is not None
        ]
        if lines:
            out.append("Wellbeing (self-rated 1-10):")
            out.extend(lines)

    if not out:
        return ""

    body = "\n".join(out)
    return (
        "=== PATIENT CLINICAL PROFILE (authoritative, current) ===\n"
        f"{body}\n"
        "These are the patient's own recorded clinical details. Use them "
        "directly and never ask the patient to repeat what is listed here. "
        "A confirmed absence is a positive statement by the patient; an absent "
        "section simply means nothing has been recorded."
    )


__all__ = [
    "ALL_SECTIONS",
    "LIFESTYLE",
    "WELLBEING",
    "ALLERGIES",
    "MEDICATIONS",
    "CONDITIONS",
    "SURGERIES",
    "select_relevant_sections",
    "render_health_profile_block",
]
