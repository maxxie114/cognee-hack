"""LangGraph RAG pipeline: retrieve → generate."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .config import RAGConfig
from .retrieval import get_retriever, lookup_cache, save_to_cache


class RAGState(TypedDict):
    """State for the RAG graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    context: list[Document]
    question: str
    answer: str | None
    cache_hit: bool


def _get_question(state: RAGState) -> str:
    """Extract the latest user question from messages."""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage) and m.content:
            return m.content if isinstance(m.content, str) else str(m.content)
    return state.get("question", "")


def check_cache_node(state: RAGState, config: RunnableConfig) -> dict:
    """Check if query is in cache."""
    question = _get_question(state)
    if not question:
        return {"cache_hit": False}

    rag_config = RAGConfig.from_env()
    cached_answer = lookup_cache(question, config=rag_config)
    
    if cached_answer:
        return {
            "cache_hit": True,
            "answer": cached_answer,
            "messages": [SystemMessage(content=f"Cache Hit: {cached_answer}")] # Start with system message or just return answer
        }
    
    return {"cache_hit": False, "question": question}


def retrieve_node(state: RAGState, config: RunnableConfig) -> dict:
    """Retrieve relevant documents and attach to state."""
    question = state.get("question") or _get_question(state)
    if not question:
        return {"context": [], "question": question}

    rag_config = RAGConfig.from_env()
    retriever = get_retriever(rag_config)
    docs = retriever(question, config=rag_config, top_k=rag_config.top_k)
    return {"context": docs, "question": question}


def generate_node(state: RAGState, config: RunnableConfig) -> dict:
    """Build context from retrieved docs and generate answer with LLM."""
    context_docs = state.get("context") or []
    question = state.get("question") or _get_question(state)

    context_str = "\n\n---\n\n".join(
        d.page_content for d in context_docs
    ) or "No relevant context found."

    rag_config = RAGConfig.from_env()
    llm = ChatOpenAI(
        model=rag_config.llm_model,
        temperature=rag_config.llm_temperature,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You answer questions based only on the following context. If the context does not contain enough information, say so. Do not make up information.\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"context": context_str, "question": question})

    return {"messages": [response], "answer": response.content}


def write_cache_node(state: RAGState, config: RunnableConfig) -> dict:
    """Write generated answer to cache."""
    question = state.get("question")
    answer = state.get("answer")
    
    if question and answer:
        rag_config = RAGConfig.from_env()
        save_to_cache(question, str(answer), config=rag_config)
    
    return {}


def route_cache(state: RAGState) -> str:
    """Route based on cache hit."""
    if state.get("cache_hit"):
        return END
    return "retrieve"


def build_rag_graph() -> StateGraph:
    """Build the LangGraph RAG pipeline: check_cache -> (retrieve -> generate -> write_cache)."""
    graph = StateGraph(RAGState)

    graph.add_node("check_cache", check_cache_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("write_cache", write_cache_node)

    graph.set_entry_point("check_cache")
    
    graph.add_conditional_edges(
        "check_cache",
        route_cache,
        {
            END: END,
            "retrieve": "retrieve"
        }
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "write_cache")
    graph.add_edge("write_cache", END)

    return graph


def compile_rag_graph():
    """Compile and return the runnable RAG graph."""
    return build_rag_graph().compile()

# Expose the graph for LangGraph Studio
graph = compile_rag_graph()
