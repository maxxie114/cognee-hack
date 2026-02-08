"""Agentic RAG: ReAct agent with retrieval and optional analyze/reformulate tools.

Exposes create_rag_agent(config) returning a message-based runnable so the
supervisor (or any caller) can ainvoke({"messages": [HumanMessage(...)]})
and read result["messages"][-1].content as the RAG response.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool
import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from .rag.config import RAGConfig
from .rag.retrieval import get_retriever

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable


def _make_retrieve_documents_tool(config: RAGConfig):
    """Build retrieve_documents tool bound to the given config."""

    @tool
    def retrieve_documents(
        query: str,
        top_k: int = 5,
        filter_expression: str | None = None,
    ) -> str:
        """Retrieve relevant documents from the vector store (Qdrant).
        Use this to get factual information from the indexed documents.
        """
        retriever = get_retriever(config)
        docs = retriever(
            query,
            config=config,
            top_k=top_k,
            filter_expression=filter_expression or None,
        )
        if not docs:
            return "No relevant documents found for this query."
        parts = []
        for i, doc in enumerate(docs, 1):
            score = (doc.metadata.get("score") if doc.metadata else None) or "N/A"
            content = (doc.page_content or "")[:500]
            if len(doc.page_content or "") > 500:
                content += "..."
            parts.append(f"{i}. [Relevance: {score}] {content}")
        return "\n\n".join(parts)

    return retrieve_documents


def _make_analyze_retrieved_context_tool():
    """Build analyze_retrieved_context tool (rule-based)."""

    @tool
    def analyze_retrieved_context(context: str) -> str:
        """Analyze retrieved context to identify key facts, medications, diagnoses, and missing information."""
        lines = context.split("\n")
        key_points = []
        keywords = ["diagnosed", "medication", "treatment", "lab", "test", "symptom", "history"]
        for line in lines:
            if any(kw in line.lower() for kw in keywords):
                key_points.append(line.strip())
        if key_points:
            return "Key points extracted:\n" + "\n".join(key_points[:10])
        return "No specific key points identified in context."

    return analyze_retrieved_context


def _make_reformulate_query_tool():
    """Build reformulate_query tool (rule-based)."""

    @tool
    def reformulate_query(original_query: str, conversation_context: str = "") -> str:
        """Reformulate the query to improve retrieval based on conversation context.
        Example: 'What about his heart?' -> 'What is the patient's cardiac history?'
        """
        query = original_query.strip()
        medical_terms = ["medication", "diagnosis", "treatment", "symptoms", "history", "condition"]
        if not any(term in query.lower() for term in medical_terms):
            if conversation_context.strip():
                query = f"Context: {conversation_context[:200]}\nQuery: {query}"
            else:
                query = f"Patient: {query}"
        return f"Reformulated: {query}"

    return reformulate_query


RAG_AGENT_SYSTEM_PROMPT = """You are a retrieval-augmented specialist. Your job is to:

1. RETRIEVE relevant documents using the retrieve_documents tool when you need factual information.
2. ANALYZE the retrieved context using analyze_retrieved_context to extract key points.
3. REFORMULATE the query using reformulate_query if the question is vague or needs context.
4. SYNTHESIZE a clear, accurate response based on retrieved context.

IMPORTANT:
- Base your answers on retrieved documents when the question asks for factual information.
- If information is missing or not found, say so clearly.
- Use the tools as needed; you may retrieve multiple times with different queries if useful.
- Consider conversation context when interpreting ambiguous questions."""


def create_rag_agent(config: RAGConfig | None = None) -> Runnable:
    """Create an agentic RAG agent (ReAct) that the supervisor can call with messages.

    The returned runnable supports:
    - .invoke({"messages": [HumanMessage(content=...)]]})
    - .ainvoke({"messages": [HumanMessage(content=...)]]})

    The supervisor should read result["messages"][-1].content as the RAG response.

    Args:
        config: RAG config for retrieval (Qdrant collection, top_k, etc.). Uses RAGConfig.from_env() if None.

    Returns:
        Compiled LangGraph agent with message state (messages in, messages out).
    """
    cfg = config or RAGConfig.from_env()

    tools = [
        _make_retrieve_documents_tool(cfg),
        _make_analyze_retrieved_context_tool(),
        _make_reformulate_query_tool(),
    ]

    _ollama_base = os.getenv("OLLAMA_LLM_BASE")
    llm = ChatOpenAI(
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
        **({"base_url": _ollama_base} if _ollama_base else {}),
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=RAG_AGENT_SYSTEM_PROMPT,
    )

    return agent
