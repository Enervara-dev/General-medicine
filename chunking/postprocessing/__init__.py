"""Generic, specialty-agnostic ingestion-time data-quality pipeline.

See postprocessor.py for the mechanism; app/specialty/content/orthopaedics.py
for the only currently-populated data (ENTITY_OVERRIDES,
RELATION_TYPE_BY_TARGET_TYPE, CANONICAL_ENTITIES).
"""

from chunking.postprocessing.postprocessor import postprocess_chunk

__all__ = ["postprocess_chunk"]
