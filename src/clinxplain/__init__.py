"""ClinXplain: RAG pipeline, agentic RAG, and self-evolving supervisor for medical Q&A."""

from .rag import RAGConfig, RAGPipeline
from .agentic import create_rag_agent
from .supervisor import (
    create_supervisor,
    query_medical_system,
    ContextStrategy,
    ConversationTurn,
)

__all__ = [
    "RAGConfig",
    "RAGPipeline",
    "create_rag_agent",
    "create_supervisor",
    "query_medical_system",
    "ContextStrategy",
    "ConversationTurn",
]

__version__ = "0.1.0"
