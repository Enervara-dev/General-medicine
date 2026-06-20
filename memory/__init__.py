"""
Longitudinal patient memory subsystem.

See docs/MEMORY_REDESIGN.md for the architecture rationale.

Public surface:
    LongitudinalMemoryAdapter — drop-in replacement for the legacy
                                SessionMemoryAdapter; orchestrates the
                                extraction → consolidation → retrieval flow.
    schemas                   — Pydantic DTOs (MemoryContext, ClinicalFactDTO).
    services                  — application services (extraction, consolidation,
                                retrieval, safety, summarization).
"""

from memory.adapter import LongitudinalMemoryAdapter

__all__ = ["LongitudinalMemoryAdapter"]
