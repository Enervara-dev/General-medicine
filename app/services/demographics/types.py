"""
DemographicContextV1 — the typed, AI-safe view of a patient's demographics.

Only these seven fields ever reach the LLM. The raw MongoDB user document (which
also carries email, phone, password/token hashes, firebaseUID, _id, etc.) is
NEVER exposed: the repository projects to AI-safe fields at the query, and this
module derives the final shape. ``age`` is derived from ``dateOfBirth`` and
``bmi`` from ``heightCm``/``weightKg`` — both only when the inputs are present.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel

# The ONLY fields read from the Mongo `users` document. Anything not listed here
# never leaves the database (see DemographicsRepository's projection). Keep this
# tuple as the single source of truth for the read projection.
AI_SAFE_MONGO_FIELDS: tuple[str, ...] = (
    "dateOfBirth",
    "sex",
    "heightCm",
    "weightKg",
    "state",
    "city",
)


class DemographicContextV1(BaseModel):
    """AI-safe demographic snapshot. Every field is optional (data is partial)."""

    age: Optional[int] = None            # derived from dateOfBirth
    sex: Optional[str] = None
    height_cm: Optional[int] = None
    weight_kg: Optional[int] = None
    bmi: Optional[float] = None          # derived from height + weight
    state: Optional[str] = None
    city: Optional[str] = None

    def is_empty(self) -> bool:
        """True when no AI-safe field is populated (nothing worth injecting)."""
        return not any(
            v is not None and v != ""
            for v in (self.age, self.sex, self.height_cm, self.weight_kg, self.bmi, self.state, self.city)
        )


def _coerce_dob(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        # Accept ISO-ish strings (e.g. "2002-04-13" or full timestamps).
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            try:
                return datetime.strptime(value[:10], "%Y-%m-%d").date()
            except ValueError:
                return None
    return None


def derive_age(dob: Any, *, today: Optional[date] = None) -> Optional[int]:
    """Whole-years age from a date of birth. None when dob is absent/invalid."""
    d = _coerce_dob(dob)
    if d is None:
        return None
    ref = today or date.today()
    years = ref.year - d.year - ((ref.month, ref.day) < (d.month, d.day))
    # Guard against corrupt future dates / absurd values.
    if years < 0 or years > 130:
        return None
    return years


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):  # bool is an int subclass — reject it
        return None
    if isinstance(value, (int, float)):
        return int(value) if value > 0 else None
    if isinstance(value, str) and value.strip():
        try:
            n = int(float(value))
            return n if n > 0 else None
        except ValueError:
            return None
    return None


def derive_bmi(height_cm: Any, weight_kg: Any) -> Optional[float]:
    """BMI (kg/m²) rounded to 1 dp. None unless BOTH inputs are valid + positive."""
    h = _coerce_int(height_cm)
    w = _coerce_int(weight_kg)
    if not h or not w:
        return None
    metres = h / 100.0
    bmi = w / (metres * metres)
    if bmi <= 0 or bmi > 200:  # sanity clamp against corrupt data
        return None
    return round(bmi, 1)


def _clean_str(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def build_demographic_context_v1(
    doc: dict[str, Any], *, today: Optional[date] = None
) -> DemographicContextV1:
    """
    Project a (already AI-safe) Mongo user doc into DemographicContextV1.

    ``today`` is injectable so age derivation is deterministic in tests.
    """
    height = _coerce_int(doc.get("heightCm"))
    weight = _coerce_int(doc.get("weightKg"))
    return DemographicContextV1(
        age=derive_age(doc.get("dateOfBirth"), today=today),
        sex=_clean_str(doc.get("sex")),
        height_cm=height,
        weight_kg=weight,
        bmi=derive_bmi(height, weight),
        state=_clean_str(doc.get("state")),
        city=_clean_str(doc.get("city")),
    )


__all__ = [
    "AI_SAFE_MONGO_FIELDS",
    "DemographicContextV1",
    "build_demographic_context_v1",
    "derive_age",
    "derive_bmi",
]
