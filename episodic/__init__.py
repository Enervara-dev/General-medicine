"""
Episodic Memory Layer — clinically-aware episodic memory for the Enervera
medical AI system.

Isolated from the Postgres longitudinal memory subsystem under memory/.
Backed by Pinecone (namespace per user_id, llama-text-embed-v2 embeddings).
"""

# Windows TLS bootstrap: on machines where a corporate proxy / AV / VPN
# intercepts TLS, the cert presented to Python isn't in certifi's Mozilla
# bundle. truststore re-routes Python's ssl module through the Windows
# system trust store, which has the corporate root CA. This is the
# upstream-blessed fix for the "unable to get local issuer certificate"
# error on enterprise Windows machines.
try:
    import truststore as _truststore
    _truststore.inject_into_ssl()
except ImportError:
    # truststore unavailable — fall back to certifi env vars so at least
    # non-MITM environments work.
    import os as _os
    try:
        import certifi as _certifi
        _bundle = _certifi.where()
        _os.environ.setdefault("SSL_CERT_FILE", _bundle)
        _os.environ.setdefault("REQUESTS_CA_BUNDLE", _bundle)
    except ImportError:
        pass
