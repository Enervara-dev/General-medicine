"""
Per-turn patient clinical profile supplied by the Backend.

The Backend (PostgreSQL) owns the patient's clinical record and sends the ACTIVE
profile on each chat request, inside the identity envelope. GM never queries the
Backend's database and keeps no profile store of its own.

Sections carried: lifestyle, wellbeing, allergies, conditions, medications,
surgeries. Allergies and conditions are envelopes rather than bare lists so that
"nothing recorded" and "patient confirmed there is none" stay distinguishable.
Archived rows are filtered out by the Backend and never arrive here.
Reproductive health is deliberately absent: there is no table, endpoint or data
source behind it.

Sections are selected per turn by deterministic relevance rules, so only what
the current question needs reaches the prompt. Profile data is transient within
GM: not persisted, not sent to PMS, not written to episodic memory, not
embedded, not logged. Missing or malformed data fails open.
"""

from app.services.profile.relevance import (
    ALL_SECTIONS,
    render_health_profile_block,
    select_relevant_sections,
)

__all__ = [
    "ALL_SECTIONS",
    "render_health_profile_block",
    "select_relevant_sections",
]
