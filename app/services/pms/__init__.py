"""
PMS (Patient Management System) integration — client abstraction only.

This repository is being prepared to become the PRODUCER of longitudinal
clinical memory. This package defines:

    ClinicalMemoryEventV1  — the versioned event emitted after clinical
                             extraction (the payload PMS will later ingest).
    PMSClient              — the client INTERFACE (no network implementation).
    NullPMSClient          — the default no-op wired today; does NOT call PMS,
                             so all existing behaviour is unchanged.
    ClinicalMemoryProducer — builds the event from an extracted episode +
                             IdentityContext and hands it to the client. This
                             is the integration point where PMS ingestion will
                             later occur.

Nothing here calls PMS. The future HTTP client is a documented placeholder.
"""

from app.services.pms.auth import StaticTokenProvider, TokenProvider
from app.services.pms.client import (
    HttpPMSClient,
    NullPMSClient,
    PMSClient,
    build_pms_client,
)
from app.services.pms.events import (
    ClinicalCategory,
    ClinicalEntities,
    ClinicalMemoryEventV1,
    ClinicalPriority,
    ClinicalSeverity,
    ClinicalTiming,
    EventSource,
    SCHEMA_VERSION,
)
from app.services.pms.producer import ClinicalMemoryProducer

__all__ = [
    "SCHEMA_VERSION",
    "ClinicalCategory",
    "ClinicalEntities",
    "ClinicalMemoryEventV1",
    "ClinicalMemoryProducer",
    "ClinicalPriority",
    "ClinicalSeverity",
    "ClinicalTiming",
    "EventSource",
    "HttpPMSClient",
    "NullPMSClient",
    "PMSClient",
    "StaticTokenProvider",
    "TokenProvider",
    "build_pms_client",
]
