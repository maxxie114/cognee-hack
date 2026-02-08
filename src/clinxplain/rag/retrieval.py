"""Retrieval: vector search over Qdrant and conversion to LangChain documents."""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from .config import RAGConfig
from .ingestion import get_qdrant_client


def _retrieve_cognee(
    query: str,
    config: RAGConfig,
    *,
    top_k: int | None = None,
    filter_expression: str | None = None,
    source_filter: str | None = None,
    qdrant_client: QdrantClient | None = None,
) -> list[Document]:
    """
    Retrieve via Cognee Search (chunk mode), mapping results to LangChain Documents.
    Lazy-imports Cognee and the Qdrant adapter; runs async search in a sync context.
    """
    import asyncio

    async def _search() -> list[Document]:
        # Import runs registration; do not call register (it is a module, not a function)
        import cognee_community_vector_adapter_qdrant.register  # noqa: F401
        import cognee

        k = top_k if top_k is not None else config.top_k
        dataset_name = config.cognee_dataset_name
        # CHUNK = raw vector search over chunks (closest to current Qdrant behavior)
        try:
            search_type = getattr(cognee.SearchType, "CHUNK", None) or getattr(
                cognee.SearchType, "chunk", None
            )
        except AttributeError:
            search_type = "chunk"
        result = await cognee.search(
            query_text=query,
            query_type=search_type,
            datasets=[dataset_name],
            top_k=k,
        )
        docs: list[Document] = []
        # Cognee may return list of nodes/chunks with content and metadata
        if result is None:
            return docs
        items = result if isinstance(result, list) else getattr(result, "results", []) or []
        for item in items:
            if hasattr(item, "content"):
                content = item.content
            elif isinstance(item, dict):
                content = item.get("content") or item.get("text") or item.get("page_content") or ""
            else:
                content = getattr(item, "text", None) or getattr(item, "page_content", None) or ""
            score = None
            source = ""
            if isinstance(item, dict):
                score = item.get("score")
                source = item.get("source", "")
            else:
                score = getattr(item, "score", None)
                source = getattr(item, "source", "") or ""
            docs.append(
                Document(
                    page_content=content or "",
                    metadata={"source": source, "score": score, "chunk_index": 0},
                )
            )
        # Post-filter by source when document is selected (no data leak)
        if source_filter is not None and source_filter:
            docs = [d for d in docs if (d.metadata.get("source") or "") == source_filter]
        return docs

    try:
        return asyncio.run(_search())
    except RuntimeError as e:
        if "running event loop" in str(e).lower() or "cannot be called" in str(e).lower():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _search())
                return future.result()
        raise


def get_retriever(
    config: RAGConfig | None = None,
) -> Callable[..., list[Document]]:
    """
    Return a retriever callable with the same signature as retrieve().
    Uses Qdrant when config.rag_backend != "cognee"; uses Cognee Search when "cognee" (lazy-import).
    """
    cfg = config or RAGConfig.from_env()
    if cfg.rag_backend == "cognee":
        def cognee_retriever(
            query: str,
            config: RAGConfig | None = None,
            *,
            top_k: int | None = None,
            filter_expression: str | None = None,
            source_filter: str | None = None,
            qdrant_client: QdrantClient | None = None,
        ) -> list[Document]:
            return _retrieve_cognee(
                query,
                config or cfg,
                top_k=top_k,
                filter_expression=filter_expression,
                source_filter=source_filter,
                qdrant_client=qdrant_client,
            )
        return cognee_retriever
    return retrieve


def retrieve(
    query: str,
    config: RAGConfig | None = None,
    *,
    top_k: int | None = None,
    filter_expression: str | None = None,
    source_filter: str | None = None,
    qdrant_client: QdrantClient | None = None,
) -> list[Document]:
    """
    Embed query, run vector search on Qdrant, return LangChain Documents.

    Args:
        query: User query string.
        config: RAG config; uses RAGConfig.from_env() if None.
        top_k: Number of chunks to return; uses config.top_k if None.
        filter_expression: Optional filter; reserved for future use.
        source_filter: When set, only return chunks with payload source == source_filter (document-scoped).
        qdrant_client: Optional Qdrant client.

    Returns:
        List of LangChain Document with page_content and metadata (source, chunk_index, score).
    """
    config = config or RAGConfig.from_env()
    client = get_qdrant_client(config, qdrant_client)
    k = top_k if top_k is not None else config.top_k

    if not client.collection_exists(config.collection_name):
        return []

    embeddings = OpenAIEmbeddings(model=config.embedding_model)
    query_vector = embeddings.embed_query(query)

    query_filter = None
    if source_filter:
        query_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    hits = client.query_points(
        collection_name=config.collection_name,
        query=query_vector,
        limit=k,
        with_payload=True,
        query_filter=query_filter,
    ).points

    docs: list[Document] = []
    for hit in hits:
        payload = hit.payload or {}
        # Qdrant returns score as similarity (higher = better); we store as score in [0,1]
        score = float(hit.score) if hit.score is not None else None
        metadata: dict[str, Any] = {
            "source": payload.get("source", ""),
            "chunk_index": payload.get("chunk_index", 0),
            "score": score,
        }
        docs.append(
            Document(
                page_content=payload.get("content", ""),
                metadata=metadata,
            )
        )
    return docs


def lookup_cache(
    query: str,
    config: RAGConfig | None = None,
    qdrant_client: QdrantClient | None = None,
) -> str | None:
    """
    Check Qdrant cache for a semantically similar query.

    Returns:
        Cached answer string if hit, else None.
    """
    config = config or RAGConfig.from_env()
    client = get_qdrant_client(config, qdrant_client)

    if not config.cache_ttl or config.cache_ttl <= 0:
        return None

    if not client.collection_exists(config.cache_collection_name):
        # Create cache collection if it doesn't exist
        from qdrant_client.models import VectorParams, Distance
        client.create_collection(
            collection_name=config.cache_collection_name,
            vectors_config=VectorParams(size=config.embedding_dimensions, distance=Distance.COSINE),
        )
        return None

    embeddings = OpenAIEmbeddings(model=config.embedding_model)
    query_vector = embeddings.embed_query(query)

    hits = client.query_points(
        collection_name=config.cache_collection_name,
        query=query_vector,
        limit=config.cache_top_k,
        with_payload=True,
    ).points

    if not hits:
        return None

    hit = hits[0]
    score = float(hit.score) if hit.score is not None else 0.0

    if score >= config.cache_distance_threshold:
        payload = hit.payload or {}
        # Check TTL if enabled (optional implementation detail, for now we just return if score matches)
        # In a full implementation, we'd check time.time() - payload.get("ts") < config.cache_ttl
        return payload.get("answer")
    
    return None


def save_to_cache(
    query: str,
    answer: str,
    config: RAGConfig | None = None,
    qdrant_client: QdrantClient | None = None,
) -> None:
    """
    Save query and answer to Qdrant cache.
    """
    config = config or RAGConfig.from_env()
    client = get_qdrant_client(config, qdrant_client)

    if not config.cache_ttl or config.cache_ttl <= 0:
        return

    # Ensure collection exists
    if not client.collection_exists(config.cache_collection_name):
        from qdrant_client.models import VectorParams, Distance
        client.create_collection(
            collection_name=config.cache_collection_name,
            vectors_config=VectorParams(size=config.embedding_dimensions, distance=Distance.COSINE),
        )

    embeddings = OpenAIEmbeddings(model=config.embedding_model)
    query_vector = embeddings.embed_query(query)

    import time
    from qdrant_client.models import PointStruct
    import uuid

    point_id = str(uuid.uuid4())
    
    client.upsert(
        collection_name=config.cache_collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector=query_vector,
                payload={
                    "query": query,
                    "answer": answer,
                    "ts": time.time(),
                }
            )
        ]
    )
