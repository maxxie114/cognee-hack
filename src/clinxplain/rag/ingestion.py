"""Document ingestion: loaders, chunking, embedding, and indexing into Qdrant."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import RAGConfig


def get_qdrant_client(config: RAGConfig, client: QdrantClient | None = None) -> QdrantClient:
    """Return a Qdrant client from config or the provided client."""
    if client is not None:
        return client
    return QdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
    )


def load_documents(
    path: str | Path,
    glob: str = "**/*.pdf",
    loader_type: str = "auto",
) -> list[Document]:
    """
    Load documents from a directory or file.

    Args:
        path: Directory path or file path.
        glob: Glob pattern for DirectoryLoader (e.g. "**/*.pdf", "**/*.txt").
        loader_type: "auto", "pdf", "text", or "markdown". For "auto", uses DirectoryLoader with multiple suffixes.

    Returns:
        List of LangChain Document objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_file():
        single_path = str(path)
        if single_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(single_path)
        elif single_path.lower().endswith(".md"):
            loader = TextLoader(single_path)
        else:
            loader = TextLoader(single_path)
        return loader.load()

    loaders_map = {
        "pdf": (glob, PyPDFLoader),
        "text": (glob, TextLoader),
        "markdown": (glob, TextLoader),
    }
    if loader_type in loaders_map:
        pattern, loader_cls = loaders_map[loader_type]
        loader = DirectoryLoader(str(path), glob=pattern, loader_cls=loader_cls)
        return loader.load()

    # Auto: load PDFs and text/md
    all_docs: list[Document] = []
    for pattern, loader_cls in [("**/*.pdf", PyPDFLoader), ("**/*.txt", TextLoader), ("**/*.md", TextLoader)]:
        try:
            loader = DirectoryLoader(str(path), glob=pattern, loader_cls=loader_cls)
            all_docs.extend(loader.load())
        except Exception:
            continue
    return all_docs


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def embed_and_format_for_qdrant(
    chunks: list[Document],
    embeddings: OpenAIEmbeddings,
) -> list[PointStruct]:
    """
    Embed chunk texts and format as Qdrant points (vector + payload).
    """
    texts = [d.page_content for d in chunks]
    vectors = embeddings.embed_documents(texts)

    points: list[PointStruct] = []
    for i, (doc, vec) in enumerate(zip(chunks, vectors)):
        source = doc.metadata.get("source", "unknown")
        document_name = Path(source).name if source else "unknown"
        metadata_tag = str(doc.metadata.get("metadata", ""))[:500]
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "content": doc.page_content,
                    "source": source,
                    "document_name": document_name,
                    "chunk_index": i,
                    "metadata": metadata_tag or "none",
                },
            )
        )
    return points


def ensure_collection(client: QdrantClient, config: RAGConfig) -> None:
    """Create the RAG collection if it does not exist."""
    if not client.collection_exists(config.collection_name):
        client.create_collection(
            collection_name=config.collection_name,
            vectors_config=VectorParams(
                size=config.embedding_dimensions,
                distance=Distance.COSINE,
            ),
        )


def ingest(
    path: str | Path,
    config: RAGConfig | None = None,
    *,
    glob: str = "**/*.pdf",
    loader_type: str = "auto",
    qdrant_client: QdrantClient | None = None,
) -> list[str]:
    """
    Ingest documents from path: load → chunk → embed → index into Qdrant.

    Args:
        path: Directory or file path to load from.
        config: RAG config; uses RAGConfig.from_env() if None.
        glob: Glob for directory loading.
        loader_type: "auto", "pdf", "text", or "markdown".
        qdrant_client: Optional Qdrant client; otherwise uses config.qdrant_url.

    Returns:
        List of Qdrant point IDs written.
    """
    config = config or RAGConfig.from_env()
    client = get_qdrant_client(config, qdrant_client)
    # Replace index: clear collection so only current ingest remains
    if client.collection_exists(config.collection_name):
        client.delete_collection(config.collection_name)
    ensure_collection(client, config)

    documents = load_documents(path, glob=glob, loader_type=loader_type)
    if not documents:
        return []

    chunks = chunk_documents(
        documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    embeddings = OpenAIEmbeddings(model=config.embedding_model)
    points = embed_and_format_for_qdrant(chunks, embeddings)

    client.upsert(collection_name=config.collection_name, points=points)
    return [str(p.id) for p in points]


def list_documents(
    config: RAGConfig | None = None,
    qdrant_client: QdrantClient | None = None,
) -> list[dict[str, str]]:
    """
    Return list of documents in the RAG collection (unique source + document_name from payloads).

    Returns:
        List of {"source": str, "document_name": str} for the UI document selector.
    """
    config = config or RAGConfig.from_env()
    client = get_qdrant_client(config, qdrant_client)
    if not client.collection_exists(config.collection_name):
        return []
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, str]] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=config.collection_name,
            with_payload=True,
            with_vectors=False,
            limit=100,
            offset=offset,
        )
        for rec in records:
            payload = rec.payload or {}
            source = payload.get("source") or ""
            doc_name = payload.get("document_name") or (Path(source).name if source else "")
            key = (source, doc_name)
            if key not in seen and source:
                seen.add(key)
                result.append({"source": source, "document_name": doc_name})
        if offset is None:
            break
    return result
