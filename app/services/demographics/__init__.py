"""
Per-turn patient demographics supplied by the Backend.

The Backend (PostgreSQL) owns the patient record and provides a minimal, AI-safe
demographics envelope on each chat request. GM does not query the Backend's
database and maintains no demographic or patient-profile store of its own. An
earlier design opened a direct connection to the Backend's data store and looked
the patient up by id; that coupled a specialty service to another service's
schema and is gone.

The envelope carries only:
    age, sex, height_cm, weight_kg, bmi, state, city

A derived ``age`` travels instead of a date of birth, and nothing outside this
projection is sent, so an email, phone, credential or internal identifier cannot
leak by accident.

Demographics are selected per turn by deterministic relevance rules: only the
fields relevant to the current intent reach the prompt, and a generic
educational question receives none. Height, weight and BMI belong to the BODY
bundle and are withheld unless the question actually concerns them.

The Backend is authoritative for these fields. When authoritative demographics
are present for a turn, conflicting conversational/session values for the same
fields are suppressed rather than presented alongside them.

Demographic data is transient within GM:
    not queried from any Backend database
    not persisted by GM
    not sent to PMS
    not written to episodic memory
    not embedded into vector stores
    not logged

Missing or malformed data fails open: the affected fields are treated as
unavailable and the request continues without a demographic block.

The wider clinical profile (lifestyle, wellbeing, allergies, conditions,
medications, surgeries) arrives on the same request and is handled by
``app.services.profile`` under the same guarantees.
"""



from app.services.demographics.relevance import (

    render_demographic_block,

    select_relevant_fields,

)

from app.services.demographics.types import (

    AI_SAFE_DEMOGRAPHIC_FIELDS,

    DemographicContextV1,

    build_demographic_context_v1,

    derive_age,

    derive_bmi,

)



__all__ = [

    "AI_SAFE_DEMOGRAPHIC_FIELDS",

    "DemographicContextV1",

    "build_demographic_context_v1",

    "derive_age",

    "derive_bmi",

    "render_demographic_block",

    "select_relevant_fields",

]

