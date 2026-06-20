"""
Enervera FastAPI service — production deployment layer.

Mounts /chat (orchestration), /chat/stream (SSE), /episodic/* (memory layer),
/health, /healthz/ready, /metrics. Reuses the existing GraphRAG and episodic
packages; this layer adds the async HTTP transport + lifespan + observability.
"""

# Reuse the episodic package's Windows TLS bootstrap so corp-MITM cert
# stores work for both the FastAPI lifespan and downstream Pinecone calls.
import episodic  # noqa: F401
