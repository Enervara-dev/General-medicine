from typing import TYPE_CHECKING, Any

from pinecone import Pinecone

from graphrag.config.settings import Config
from graphrag.utils.logger import get_logger
from graphrag.utils.rate_limit import call_with_retries

if TYPE_CHECKING:
    from app.specialty.models import SpecialtyConfig

logger = get_logger(__name__)


class PineconeRetriever:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        index_name: str | None = None,
        namespace: str = "",
    ):
        """
        ``api_key``/``index_name``/``namespace`` are optional overrides for
        validation/tooling (see graphrag/retrieval/pinecone_accounts.py) —
        omitted, this constructs exactly as before (``Config.PINECONE_API_KEY``
        / ``Config.PINECONE_INDEX_NAME``, no namespace). The live
        AsyncOrchestrator pipeline always omits them, so its behavior is
        unchanged. ``namespace=""`` queries Pinecone's default/unnamed
        namespace, same as never passing one.
        """
        resolved_key = api_key or Config.PINECONE_API_KEY
        if not resolved_key:
            raise ValueError("PINECONE_API_KEY is not set.")
        self.pc = Pinecone(api_key=resolved_key)
        self.index = self.pc.Index(index_name or Config.PINECONE_INDEX_NAME)
        self._namespace = namespace or ""

    def retrieve(
        self,
        query_text: str,
        vector_top_k: int = 15,
        reranker_top_k: int = 5,
        *,
        specialty: "SpecialtyConfig | None" = None,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        1. Embed query
        2. Fetch `vector_top_k` candidates from Pinecone
        3. Rerank with bge-reranker-v2-m3
        4. Return top `reranker_top_k` as plain dicts

        ``specialty`` is accepted but NOT used to filter metadata — see
        graphrag/retrieval/interface.py. No metadata filtering is introduced
        by this parameter; it exists only as a documented forward-compat
        seam.

        ``namespace`` overrides the constructor's namespace for this one
        call — used by graphrag/retrieval/retriever_factory.py so a single
        PineconeRetriever instance (one Pinecone client per account) can
        serve every specialty namespace in that account without rebuilding
        the client per request. Embedding, vector search, and reranking are
        otherwise byte-identical to before this parameter existed.
        """
        effective_namespace = self._namespace if namespace is None else namespace
        logger.info(
            f"\n🔍 [1/3] Vector search  →  candidates: {vector_top_k}  |  "
            f"keep after rerank: {reranker_top_k}"
        )

        # ── 1. Embed ────────────────────────────────────────────────────────
        response = call_with_retries(
            self.pc.inference.embed,
            model="llama-text-embed-v2",
            inputs=[query_text],
            parameters={"input_type": "query", "truncate": "END"},
            operation="Pinecone query embedding",
        )
        query_vector = response[0]["values"]

        # ── 2. Retrieve candidates ──────────────────────────────────────────
        search_result = self.index.query(
            vector=query_vector,
            top_k=vector_top_k,
            include_metadata=True,
            namespace=effective_namespace,
        )
        matches = search_result.get("matches", [])

        if not matches:
            logger.info("❌ No matches found in vector store.")
            return []

        logger.info(f"🔄 {len(matches)} candidates retrieved — reranking...")

        # ── 3. Prepare reranker documents ───────────────────────────────────
        documents = []
        for match in matches:
            md = match.get("metadata", {})
            combined = (
                f"Summary: {md.get('summary', '')}\n"
                f"Key Content (Entities): {', '.join(md.get('entities', []))}"
            )
            documents.append({"id": match["id"], "text": combined})

        # ── 4. Rerank ────────────────────────────────────────────────────────
        try:
            rerank_resp = call_with_retries(
                self.pc.inference.rerank,
                model="bge-reranker-v2-m3",
                query=query_text,
                documents=documents,
                rank_fields=["text"],
                top_n=reranker_top_k,
                return_documents=False,
                operation="Pinecone rerank",
            )

            reranked = []
            for result in rerank_resp.data:
                match = matches[result["index"]]
                match_dict = match.to_dict() if hasattr(match, "to_dict") else dict(match)
                match_dict["score"] = result["score"]
                match_dict["reranked"] = True
                reranked.append(match_dict)

            logger.info(f"✅ Reranking complete → {len(reranked)} chunks selected.")
            return reranked

        except Exception as e:
            logger.error(f"❌ Reranker failed: {e} — falling back to vector scores.")
            fallback = []
            for match in matches[:reranker_top_k]:
                d = match.to_dict() if hasattr(match, "to_dict") else dict(match)
                d["reranked"] = False
                fallback.append(d)
            return fallback
