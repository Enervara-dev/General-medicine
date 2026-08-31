"""
PMS (Patient Management System) integration — client abstraction only.

This repository is being prepared to become the PRODUCER of longitudinal
clinical memory. This package defines:

    PmsMemoryEventV1       — the CANONICAL wire contract to PMS, mirrored from
                             the PMS repo. Specialty-agnostic.
    PMSClient              — the client INTERFACE.
    NullPMSClient          — the default no-op; does NOT call PMS, so all
                             existing behaviour is unchanged.
    HttpPMSClient          — the shadow HTTP client, SigV4-signed for VPC
                             Lattice (AWS_IAM) using the ECS task role.
    SigV4RequestSigner     — the httpx.Auth that signs each request.
    USER_ASSERTION_HEADER  — the Backend-minted RS256 assertion GM forwards
                             verbatim; see assertions.py.
    ClinicalMemoryProducer — builds the event from an extracted episode +
                             IdentityContext and hands it to the client. This
                             is the integration point where PMS ingestion will
                             later occur.

Transport auth is SigV4/Lattice. Patient-scope authorization is NOT yet
enforced — see assertions.py for the seam and the known gap.
"""

from app.services.pms.assertions import (
    SCOPE_MEMORY_PURGE,
    SCOPE_MEMORY_READ,
    SCOPE_MEMORY_WRITE,
    USER_ASSERTION_HEADER,
    MissingUserAssertionError,
)
from app.services.pms.client import (
    HttpPMSClient,
    NullPMSClient,
    PMSClient,
    build_pms_client,
)
from app.services.pms.events import (
    SCHEMA_VERSION,
    ClinicalCategory,
    ClinicalEntities,
    ClinicalPriority,
    ClinicalSeverity,
    ClinicalStatus,
    ClinicalTiming,
    Evidence,
    PmsMemoryEventV1,
    SourceChannel,
    SourceRef,
)
from app.services.pms.producer import ClinicalMemoryProducer
from app.services.pms.signing import SigningError, SigV4RequestSigner

__all__ = [
    "SCHEMA_VERSION",
    "SCOPE_MEMORY_PURGE",
    "SCOPE_MEMORY_READ",
    "SCOPE_MEMORY_WRITE",
    "USER_ASSERTION_HEADER",
    "ClinicalCategory",
    "ClinicalEntities",
    "ClinicalMemoryProducer",
    "ClinicalPriority",
    "ClinicalSeverity",
    "ClinicalStatus",
    "ClinicalTiming",
    "Evidence",
    "HttpPMSClient",
    "MissingUserAssertionError",
    "NullPMSClient",
    "PMSClient",
    "PmsMemoryEventV1",
    "SigV4RequestSigner",
    "SigningError",
    "SourceChannel",
    "SourceRef",
    "build_pms_client",
]
