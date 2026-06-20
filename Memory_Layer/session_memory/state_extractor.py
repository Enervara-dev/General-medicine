"""
state_extractor.py
──────────────────
Updated state extractor to handle chronic conditions, allergies, and persistence
of previous concerns for the Enervera memory layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .models import Message, RiskLevel, Role, SessionMemory, StructuredState


# ============================================================================
# Heuristic Pattern Registry
# ============================================================================

SYMPTOM_PATTERNS: dict[str, list[str]] = {
    "fever":              [r"\bfever\b", r"\bhigh temperature\b", r"\btemperature\b"],
    "chills":             [r"\bchill(s|ing)?\b", r"\bshiver(ing)?\b"],
    "sore_throat":        [r"\bsore throat\b", r"\bthroat pain\b", r"\bthroat ache\b"],
    "cough":              [r"\bcough(ing)?\b"],
    "shortness_of_breath":[r"\bshortness of breath\b", r"\bbreathing difficult\b", r"\bcan'?t breathe\b"],
    "chest_pain":         [r"\bchest pain\b", r"\bchest tightness\b"],
    "headache":           [r"\bheadache\b", r"\bhead pain\b"],
    "fatigue":            [r"\bfatigue\b", r"\btired\b", r"\bexhausted\b"],
    "nausea":             [r"\bnausea\b", r"\bfeeling sick\b"],
    "dizziness":          [r"\bdizzy\b", r"\bdizziness\b"],
}

# Distinguishing between acute conditions and chronic conditions
CHRONIC_PATTERNS: dict[str, list[str]] = {
    "diabetes":           [r"\bdiabetes\b", r"\bdiabetic\b"],
    "hypertension":       [r"\bhypertension\b", r"\bhigh blood pressure\b", r"\bhigh bp\b"],
    "asthma":             [r"\basthma\b", r"\basthmatic\b"],
    "heart_disease":      [r"\bheart disease\b", r"\bcardiac issue\b"],
    "thyroid":            [r"\bthyroid\b"],
    "arthritis":          [r"\barthritis\b", r"\bjoint pain\b"],
}

CONDITION_PATTERNS: dict[str, list[str]] = {
    "flu":                [r"\bflu\b", r"\binfluenza\b"],
    "strep_throat":       [r"\bstrep throat\b"],
    "covid":              [r"\bcovid\b", r"\bcoronavirus\b"],
    "infection":          [r"\binfection\b", r"\binfected\b"],
    "migraine":           [r"\bmigraine\b"],
}

ALLERGY_PATTERNS: dict[str, list[str]] = {
    "penicillin":    [r"\ballergic to penicillin\b", r"\bpenicillin allergy\b"],
    "pollen":        [r"\bpollen\b", r"\bhay fever\b"],
    "dust":          [r"\bdust allergy\b", r"\ballergic to dust\b"],
    "peanuts":       [r"\bpeanut allergy\b", r"\ballergic to peanuts\b"],
    "shellfish":     [r"\bshellfish allergy\b"],
}

DRUG_PATTERNS: dict[str, list[str]] = {
    "paracetamol":   [r"\bparacetamol\b", r"\bacetaminophen\b", r"\btylenol\b"],
    "ibuprofen":     [r"\bibuprofen\b", r"\badvil\b", r"\bnurofen\b"],
    "aspirin":       [r"\baspirin\b"],
    "insulin":       [r"\binsulin\b"],
    "amoxicillin":   [r"\bamoxicillin\b"],
}

SEVERITY_PATTERNS: dict[str, list[str]] = {
    "mild":     [r"\bmild\b", r"\bslight\b"],
    "moderate": [r"\bmoderate\b", r"\bmedium\b"],
    "severe":   [r"\bsevere\b", r"\bextreme\b", r"\bintense\b", r"\bvery bad\b"],
}

DURATION_RE = re.compile(
    r"(?:for|since|over|past|last)\s+"
    r"(\d+\s+(?:second|minute|hour|day|week|month|year)s?|yesterday|this morning)",
    re.IGNORECASE,
)

AGE_RE  = re.compile(r"\b(\d{1,3})\s*(?:year(?:s)?\s*old|y\.?o\.?)\b", re.IGNORECASE)
SEX_RE  = re.compile(r"\b(male|female|man|woman)\b", re.IGNORECASE)

SEX_NORMALISE = {"man": "male", "woman": "female"}

# ── Name extraction ────────────────────────────────────────────────────────
# We pull a first name when the patient explicitly introduces themselves. The
# answer prompt uses it to address the user naturally instead of as "patient".
#
# High-confidence patterns first — these are explicit declarations and almost
# never produce false positives.
_NAME_EXPLICIT_RES: list[re.Pattern[str]] = [
    re.compile(r"\bmy name is ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bthe name'?s ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bname'?s ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bcall me ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
    re.compile(r"\bthis is ([A-Z][a-zA-Z'\-]{1,30})(?:\s+speaking|\s+here|\s*[,.])"),
    re.compile(r"\bi go by ([A-Za-z][A-Za-z'\-]{1,30})\b", re.IGNORECASE),
]

# Lower-confidence: "I am X" / "I'm X". Only trust if X looks like a name —
# capitalised in the original text AND not a common adjective / state word.
_NAME_SOFT_RE = re.compile(r"\b[Ii]\s*'?\s*[am]{1,2}\s+([A-Z][a-zA-Z'\-]{1,30})\b")

# Common words that can follow "I'm" / "I am" but are NOT names. Lowercased.
_NAME_STOPWORDS: frozenset[str] = frozenset({
    # states / feelings
    "sick", "tired", "fine", "ok", "okay", "good", "bad", "well", "great",
    "happy", "sad", "worried", "scared", "confused", "anxious", "depressed",
    "stressed", "exhausted", "hungry", "thirsty", "dizzy", "nauseous", "dying",
    "fasting", "bleeding", "burning", "shaking", "freezing",
    # statuses
    "married", "single", "pregnant", "diabetic", "allergic", "asthmatic",
    "hypertensive", "vegetarian", "vegan", "lost", "ready", "back", "done",
    "late", "early", "here", "there", "home", "outside", "indoors",
    # progressive verbs after "I'm"
    "having", "feeling", "going", "trying", "looking", "doing", "taking",
    "thinking", "wondering", "asking", "calling", "writing", "experiencing",
    "suffering", "noticing", "starting", "ending", "drinking", "eating",
    # other common
    "afraid", "unsure", "unable", "old", "young", "new", "sorry", "sure",
    "really", "always", "never", "still", "just", "also", "very",
})


def _extract_name(text: str) -> str | None:
    """
    Return the first sensible name found in `text`, or None.

    Prefers explicit declarations ("my name is X", "call me X") before the
    softer "I'm X" pattern. Soft matches are filtered through a stopword list
    so phrases like "I'm sick" / "I'm Diabetic" never produce a "name".
    """
    for pat in _NAME_EXPLICIT_RES:
        m = pat.search(text)
        if m:
            return _format_name(m.group(1))

    m = _NAME_SOFT_RE.search(text)
    if m:
        candidate = m.group(1)
        if candidate.lower() not in _NAME_STOPWORDS:
            return _format_name(candidate)
    return None


def _format_name(raw: str) -> str:
    """Normalise to Title-Case, preserving internal apostrophes and hyphens."""
    return "-".join(part[:1].upper() + part[1:].lower() for part in raw.split("-"))

# ============================================================================
# Helpers
# ============================================================================

def _match_patterns(text: str, pattern_dict: dict[str, list[str]]) -> list[str]:
    found: list[str] = []
    lower = text.lower()
    for name, patterns in pattern_dict.items():
        for pat in patterns:
            if re.search(pat, lower):
                found.append(name)
                break
    return found

def _extract_demographics(text: str) -> dict[str, Any]:
    demo: dict[str, Any] = {}
    age_m = AGE_RE.search(text)
    if age_m: demo["age"] = int(age_m.group(1))
    sex_m = SEX_RE.search(text)
    if sex_m:
        val = sex_m.group(1).lower()
        demo["sex"] = SEX_NORMALISE.get(val, val)
    name = _extract_name(text)
    if name:
        demo["name"] = name
    return demo

def _deduplicate(lst: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in lst if not (x in seen or seen.add(x))]

# ============================================================================
# State Extraction Logic
# ============================================================================

@dataclass
class RawEntities:
    symptoms:           list[str] = field(default_factory=list)
    conditions:         list[str] = field(default_factory=list)
    chronic_conditions: list[str] = field(default_factory=list)
    allergies:          list[str] = field(default_factory=list)
    drugs:              list[str] = field(default_factory=list)
    severity:           list[str] = field(default_factory=list)
    duration:           list[str] = field(default_factory=list)
    demographics:       dict[str, Any] = field(default_factory=dict)
    preferences:        dict[str, Any] = field(default_factory=dict)
    risk_level:         RiskLevel = RiskLevel.NONE

    def all_named_entities(self) -> list[str]:
        """Flat list of all recognised medical terms for discussed_entities."""
        return self.symptoms + self.conditions + self.drugs + self.chronic_conditions + self.allergies

def extract_entities(text: str, message: Message | None = None) -> RawEntities:
    symptoms   = _match_patterns(text, SYMPTOM_PATTERNS)
    conditions = _match_patterns(text, CONDITION_PATTERNS)
    chronic    = _match_patterns(text, CHRONIC_PATTERNS)
    allergies  = _match_patterns(text, ALLERGY_PATTERNS)
    drugs      = _match_patterns(text, DRUG_PATTERNS)
    severity   = _match_patterns(text, SEVERITY_PATTERNS)
    
    duration = [m.group(0).strip() for m in DURATION_RE.finditer(text)]
    demo = _extract_demographics(text)

    risk = RiskLevel.NONE
    if message and message.risk_level:
        risk = message.risk_level
    elif set(symptoms) & {"chest_pain", "shortness_of_breath"}:
        risk = RiskLevel.CRITICAL

    return RawEntities(
        symptoms=symptoms,
        conditions=conditions,
        chronic_conditions=chronic,
        allergies=allergies,
        drugs=drugs,
        severity=severity,
        duration=duration,
        demographics=demo,
        risk_level=risk
    )

def update_preferences(state: StructuredState, patch: RawEntities) -> dict[str, Any]:
    merged = dict(state.preferences or {})
    for key, val in patch.preferences.items():
        merged[key] = val
    return merged

def merge_state(existing: StructuredState, patch: RawEntities) -> StructuredState:
    data = existing.model_copy(deep=True)

    data.symptoms = _deduplicate(data.symptoms + patch.symptoms)
    data.conditions = _deduplicate(data.conditions + patch.conditions)
    data.chronic_conditions = _deduplicate(data.chronic_conditions + patch.chronic_conditions)
    data.allergies = _deduplicate(data.allergies + patch.allergies)
    data.drugs = _deduplicate(data.drugs + patch.drugs)
    data.severity = _deduplicate(data.severity + patch.severity)
    data.duration = _deduplicate(data.duration + patch.duration)

    for k, v in patch.demographics.items():
        data.demographics[k] = v

    # Preferences merge
    data.preferences = update_preferences(data, patch)

    # Maintain a history of concerns for context-aware RAG
    if patch.symptoms:
        data.previous_concerns = _deduplicate(data.previous_concerns + patch.symptoms)

    # Risk only escalates
    risk_order = ["none", "low", "medium", "high", "critical"]
    patch_risk = patch.risk_level.value if hasattr(patch.risk_level, "value") else str(patch.risk_level)
    existing_risk = data.risk_level.value if hasattr(data.risk_level, "value") else str(data.risk_level)

    if risk_order.index(patch_risk.lower()) > risk_order.index(existing_risk.lower()):
        data.risk_level = patch.risk_level

    data.discussed_entities = _deduplicate(
        data.discussed_entities + patch.all_named_entities()
    )

    return data


def extract_state(session: SessionMemory, message: Message) -> StructuredState:
    if message.role != Role.USER:
        return session.state

    raw = extract_entities(message.content, message)
    updated = merge_state(session.state, raw)

    if message.query_type:
        updated.active_task = message.query_type
        updated.last_intent = message.query_type

    return updated
