"""RAG pipeline configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

# Load .env so QDRANT_URL, OPENAI_API_KEY, etc. are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _validate_qdrant_url(url: str) -> str:
    """Ensure QDRANT_URL is a valid HTTP(S) URL."""
    if not url:
        return url
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            "QDRANT_URL must use http:// or https://. "
            "Example: http://localhost:6333 or https://xxx.aws.cloud.qdrant.io"
        )
    return url.rstrip("/")


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""

    # Backend: qdrant (default) or cognee
    rag_backend: str = field(
        default_factory=lambda: (os.getenv("RAG_BACKEND") or "qdrant").strip().lower()
    )
    # Cognee dataset name when RAG_BACKEND=cognee (distinct from collection_name)
    cognee_dataset_name: str = field(
        default_factory=lambda: os.getenv("COGNEE_DATASET_NAME", "clin_xplain_data")
    )

    # Qdrant
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str | None = field(default_factory=lambda: os.getenv("QDRANT_API_KEY") or None)
    collection_name: str = "rag_docs"

    # Embedding (OpenAI by default; dimensions must match model)
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536  # text-embedding-3-small

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 5
    score_threshold: float | None = None

    # LLM (for generation in LangGraph)
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # LLM cache (fast repeated queries; 0 = disable)
    cache_ttl: int = 3600
    cache_collection_name: str = "qa_cache"
    cache_top_k: int = 3
    cache_distance_threshold: float = 0.80

    @property
    def index_name(self) -> str:
        """Alias for collection_name (backward compatibility)."""
        return self.collection_name

    @classmethod
    def from_env(cls) -> RAGConfig:
        """Build config from environment variables."""
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        if "YOUR_" in qdrant_url or not qdrant_url.strip():
            qdrant_url = "http://localhost:6333"
        _validate_qdrant_url(qdrant_url)
        return cls(
            rag_backend=(os.getenv("RAG_BACKEND") or "qdrant").strip().lower(),
            cognee_dataset_name=os.getenv("COGNEE_DATASET_NAME", "clin_xplain_data"),
            qdrant_url=qdrant_url,
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            collection_name=os.getenv("RAG_INDEX_NAME", "rag_docs"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimensions=int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "1536")),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            llm_model=os.getenv("RAG_LLM_MODEL", "gpt-4o-mini"),
            llm_temperature=float(os.getenv("RAG_LLM_TEMPERATURE", "0.0")),
            cache_ttl=int(os.getenv("RAG_CACHE_TTL", "3600")),
        )
