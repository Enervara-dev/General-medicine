# `memory/` — Longitudinal clinical-fact memory (WIP scaffold)

**Status: built but NOT yet wired into the running service.** Nothing imports
`LongitudinalMemoryAdapter` outside this package, so no rows are written today.

This is **intentionally kept** as the starting scaffold for the memory rebuild —
it already implements most of the target architecture (clinical facts as the
unit of memory, consolidation with supersede/contradict, allergy/interaction
safety gates, Postgres + pgvector, provenance/audit).

**Before touching this, read:**
- [`docs/MEMORY_HANDOVER.md`](../docs/MEMORY_HANDOVER.md) — current-state map,
  the seams you must preserve, and the suggested PR sequence.
- [`docs/MEMORY_REDESIGN.md`](../docs/MEMORY_REDESIGN.md) — full schema and
  migration design.

Entry point: `memory/adapter.py::LongitudinalMemoryAdapter`.
