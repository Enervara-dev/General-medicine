"""
Demographics — read-only, AI-safe patient demographic context.

Authoritative current patient facts (age/sex/height/weight/BMI/location) are
loaded per-turn from MongoDB `enervara.users` keyed on the request `user_id`,
projected to AI-safe fields, and injected into the answer prompt ONLY when
relevant to the medical query. This layer is deliberately separate from Redis
session memory, episodic memory, and Pinecone retrieval; demographics are
authoritative context, not conversational memory, and are never embedded.

Everything fails open — a Mongo outage or missing data degrades to "no
demographic context" and never breaks /chat or /chat/stream.
"""

from app.services.demographics.relevance import (
    render_demographic_block,
    select_relevant_fields,
)
from app.services.demographics.repository import DemographicsRepository
from app.services.demographics.service import (
    DemographicsService,
    build_demographics_service,
)
from app.services.demographics.types import (
    AI_SAFE_MONGO_FIELDS,
    DemographicContextV1,
    build_demographic_context_v1,
    derive_age,
    derive_bmi,
)

__all__ = [
    "AI_SAFE_MONGO_FIELDS",
    "DemographicContextV1",
    "DemographicsRepository",
    "DemographicsService",
    "build_demographics_service",
    "build_demographic_context_v1",
    "derive_age",
    "derive_bmi",
    "render_demographic_block",
    "select_relevant_fields",
]
