"""
SpecialtyRegistry — resolves a request-time specialty key to its
SpecialtyConfig.

Resolution contract (``get()``):
    * ``None`` or ``""`` (specialty OMITTED) -> "general_medicine". Every
      pre-unification caller omits this field, so this path must never raise
      — it preserves exact pre-unification behavior.
    * An unrecognized, non-empty key (specialty PROVIDED but wrong) ->
      raises ``app.core.exceptions.InvalidInput`` (HTTP 400 via the app's
      existing AppError handler). Once a specialty actually selects a real
      Pinecone account/namespace (graphrag/retrieval/pinecone_accounts.py),
      silently falling back to general_medicine for a typo'd specialty would
      serve a different account's content with no signal to the caller —
      worse than a clear rejection.

These two cases are deliberately asymmetric: omission is a normal,
supported request shape; a wrong value is a client error.
"""

from __future__ import annotations

import logging

from app.specialty.models import SpecialtyConfig

logger = logging.getLogger(__name__)

DEFAULT_SPECIALTY = "general_medicine"

# The 7 specialties this unification covers (AUDIT_REPORT.md scope).
SPECIALTY_KEYS: tuple[str, ...] = (
    "general_medicine",
    "cardiology",
    "dermatology",
    "ent",
    "ophthalmology",
    "orthopaedics",
    "pulmonology",
)


class SpecialtyRegistry:
    """Immutable lookup table of SpecialtyConfig, keyed by specialty key."""

    def __init__(self, configs: dict[str, SpecialtyConfig]) -> None:
        missing = [k for k in SPECIALTY_KEYS if k not in configs]
        if missing:
            raise ValueError(f"SpecialtyRegistry missing required specialties: {missing}")
        self._configs: dict[str, SpecialtyConfig] = dict(configs)

    def get(self, key: str | None) -> SpecialtyConfig:
        """
        Resolve a specialty key.

        ``None``/``""`` -> general_medicine (lenient — specialty omitted).
        Any other unrecognized value -> raises InvalidInput (strict —
        specialty provided but wrong). See module docstring for why these
        two cases are handled differently.
        """
        if not key:
            return self._configs[DEFAULT_SPECIALTY]
        normalized = key.strip().lower().replace("-", "_")
        config = self._configs.get(normalized)
        if config is not None:
            return config

        from app.core.exceptions import InvalidInput

        supported = sorted(self._configs)
        raise InvalidInput(
            f"Unknown specialty {key!r}. Supported specialties: {supported}.",
            details={"specialty": key, "supported": supported},
        )

    def keys(self) -> tuple[str, ...]:
        return tuple(self._configs.keys())

    def __contains__(self, key: str) -> bool:
        return (key or "").strip().lower().replace("-", "_") in self._configs

    def __len__(self) -> int:
        return len(self._configs)


def build_default_registry() -> SpecialtyRegistry:
    """Construct the registry from the bundled per-specialty content modules."""
    from app.specialty.content.cardiology import CONFIG as cardiology
    from app.specialty.content.dermatology import CONFIG as dermatology
    from app.specialty.content.ent import CONFIG as ent
    from app.specialty.content.general_medicine import CONFIG as general_medicine
    from app.specialty.content.ophthalmology import CONFIG as ophthalmology
    from app.specialty.content.orthopaedics import CONFIG as orthopaedics
    from app.specialty.content.pulmonology import CONFIG as pulmonology

    return SpecialtyRegistry(
        {
            "general_medicine": general_medicine,
            "cardiology": cardiology,
            "dermatology": dermatology,
            "ent": ent,
            "ophthalmology": ophthalmology,
            "orthopaedics": orthopaedics,
            "pulmonology": pulmonology,
        }
    )


_default_registry: SpecialtyRegistry | None = None


def get_registry() -> SpecialtyRegistry:
    """Process-wide default registry (lazy singleton; cheap to rebuild in tests)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = build_default_registry()
    return _default_registry


__all__ = [
    "DEFAULT_SPECIALTY",
    "SPECIALTY_KEYS",
    "SpecialtyRegistry",
    "build_default_registry",
    "get_registry",
]
