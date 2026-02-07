"""Self-evolving supervisor multi-agent system for medical Q&A.

Based on self_evolving_medical_qa.ipynb: supervisor that manages context
selection strategy and delegates to the agentic RAG sub-agent. Flow:
  START → synthesize_context → delegate_to_rag → check_continuation
       → [evolve] evolve_strategy → (loop to synthesize_context)
       → [finalize] finalize_response → END
"""

from __future__ import annotations

import asyncio
import json
import operator
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .rag import RAGConfig, RAGPipeline
from .agentic import create_rag_agent


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------


class ContextStrategy(BaseModel):
    """Evolving context selection strategy (RAG vs memory weights)."""

    version: int
    rag_weight: float = Field(default=0.70, ge=0.0, le=1.0)
    memory_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    retrieval_depth: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    max_iterations: int = Field(default=3, ge=1, le=5)
    iteration_count: int = 0
    last_response_quality: float = 0.0

    def validate_weights(self) -> None:
        total = self.rag_weight + self.memory_weight
        if not 0.99 <= total <= 1.01:
            self.rag_weight /= total
            self.memory_weight /= total


class RetrievedDocument(BaseModel):
    """A document retrieved from the vector store (Qdrant)."""

    content: str
    source: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationTurn(BaseModel):
    """A single turn in the conversation."""

    turn_number: int
    query: str
    response: str
    timestamp: datetime = Field(default_factory=datetime.now)
    context_used: str = ""


class AgentState(TypedDict, total=False):
    """Main graph state for the self-evolving supervisor."""

    query: str
    patient_id: str | None
    document_source: str | None
    rag_results: Annotated[list[RetrievedDocument], operator.add]
    conversation_history: Annotated[list[ConversationTurn], operator.add]
    context_strategy: ContextStrategy
    combined_context: str
    rag_response: str
    reformulated_query: str
    sources_used: list[str]
    current_iteration: int
    should_continue: bool
    final_response: str
    strategy_history: list[ContextStrategy]


# -----------------------------------------------------------------------------
# Vector store adapter (RAG pipeline → search interface for supervisor)
# -----------------------------------------------------------------------------


class RedisVLAdapter:
    """Adapter: RAG pipeline (Qdrant) exposing async search(patient_id, query, k, threshold)."""

    def __init__(self, pipeline: RAGPipeline, qdrant_client: Any = None) -> None:
        self.pipeline = pipeline
        self.qdrant_client = qdrant_client

    async def search(
        self,
        patient_id: str,
        query: str,
        k: int = 5,
        threshold: float = 0.75,
        source_filter: str | None = None,
    ) -> list[RetrievedDocument]:
        """Vector search via RAG pipeline; map to RetrievedDocument and filter by threshold."""
        effective_query = f"{query}" if not patient_id else f"Patient {patient_id}: {query}"
        docs = await asyncio.to_thread(
            self.pipeline.retrieve,
            effective_query,
            top_k=k,
            source_filter=source_filter,
            qdrant_client=self.qdrant_client,
        )
        results: list[RetrievedDocument] = []
        for d in docs:
            score = float(d.metadata.get("score") or 0.0)
            if score >= threshold:
                results.append(
                    RetrievedDocument(
                        content=d.page_content or "",
                        source=d.metadata.get("source", ""),
                        score=score,
                        metadata=dict(d.metadata),
                    )
                )
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]


# -----------------------------------------------------------------------------
# Short-term memory (Qdrant-backed)
# -----------------------------------------------------------------------------

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False
    QdrantClient = None  # type: ignore[misc, assignment]


MEMORY_COLLECTION = "agent_short_term_memory"


class ShortTermMemory:
    """Qdrant-backed short-term memory: add turns, get recent, or search by similarity."""

    def __init__(self, client: Any, collection: str, embeddings: OpenAIEmbeddings, dims: int) -> None:
        if not _QDRANT_AVAILABLE:
            raise ImportError("ShortTermMemory requires qdrant-client.")
        self.client = client
        self.collection = collection
        self.embeddings = embeddings
        self.dims = dims
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dims, distance=Distance.COSINE),
            )

    def _user_filter(self, user_id: str) -> Filter:
        return Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])

    def add(
        self,
        user_id: str,
        query: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a conversation turn. Returns memory_id."""
        memory_id = str(uuid.uuid4())
        content = f"Q: {query}\nA: {response}"
        created_at = datetime.utcnow().isoformat() + "Z"
        meta_str = json.dumps(metadata or {})
        vector = self.embeddings.embed_query(content)
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=memory_id,
                    vector=vector,
                    payload={
                        "content": content,
                        "memory_type": "short_term",
                        "metadata": meta_str,
                        "created_at": created_at,
                        "user_id": user_id,
                        "memory_id": memory_id,
                    },
                )
            ],
        )
        return memory_id

    def get_recent(self, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Fetch recent memories for user (by vector search then sort by created_at)."""
        query_vec = self.embeddings.embed_query("recent conversation memory")
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vec,
            query_filter=self._user_filter(user_id),
            limit=limit * 2,
            with_payload=True,
        )
        results = []
        for h in hits:
            p = h.payload or {}
            results.append({
                "content": p.get("content", ""),
                "created_at": p.get("created_at", ""),
                "metadata": p.get("metadata", ""),
            })
        results.sort(key=lambda x: (x.get("created_at") or ""), reverse=True)
        return results[:limit]

    def search(self, user_id: str, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Semantic search over user's short-term memories."""
        query_vec = self.embeddings.embed_query(query)
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vec,
            query_filter=self._user_filter(user_id),
            limit=k,
            with_payload=True,
        )
        return [
            {
                "content": (h.payload or {}).get("content", ""),
                "created_at": (h.payload or {}).get("created_at", ""),
                "metadata": (h.payload or {}).get("metadata", ""),
            }
            for h in hits
        ]


def create_short_term_memory(
    qdrant_client: Any,
    rag_config: RAGConfig | None = None,
) -> ShortTermMemory | None:
    """Create ShortTermMemory with Qdrant collection. Returns None on failure."""
    if not _QDRANT_AVAILABLE or qdrant_client is None:
        return None
    try:
        config = rag_config or RAGConfig.from_env()
        dims = getattr(config, "embedding_dimensions", 1536)
        return ShortTermMemory(
            qdrant_client,
            MEMORY_COLLECTION,
            OpenAIEmbeddings(model=config.embedding_model),
            dims,
        )
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Context manager (synthesize context + evolve strategy)
# -----------------------------------------------------------------------------


class ContextManager:
    """Manages context synthesis and strategy evolution."""

    def __init__(self, db: RedisVLAdapter, short_term_memory: ShortTermMemory | None = None) -> None:
        self.db = db
        self.short_term_memory = short_term_memory
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    async def synthesize_context(
        self,
        query: str,
        patient_id: str | None,
        conversation_history: list[ConversationTurn],
        strategy: ContextStrategy,
        document_source: str | None = None,
    ) -> str:
        """Synthesize context from RAG (patient history or document) and conversation memory per strategy weights."""
        rag_context = ""
        if patient_id or document_source:
            rag_results = await self.db.search(
                patient_id=patient_id or "",
                query=query,
                k=strategy.retrieval_depth,
                threshold=strategy.similarity_threshold,
                source_filter=document_source,
            )
            if rag_results:
                rag_docs = [f"[Score: {r.score:.2f}] {r.content}" for r in rag_results]
                rag_context = "\n\n".join(rag_docs)

        user_id = patient_id or "default"
        memory_context = self._get_relevant_memory(conversation_history, query, user_id=user_id)

        synthesis_prompt = f"""
You are synthesizing medical context for a patient query.

WEIGHTS: Patient History (RAG): {strategy.rag_weight:.0%}, Conversation Memory: {strategy.memory_weight:.0%}

PATIENT HISTORY:
{rag_context if rag_context else "[No relevant patient history found]"}

CONVERSATION HISTORY:
{memory_context if memory_context else "[No previous conversation]"}

CURRENT QUERY: {query}

Synthesize the above into a coherent, clinically relevant context summary. Prioritize by weights.
"""
        response = await self.llm.ainvoke(synthesis_prompt)
        return response.content if hasattr(response, "content") else str(response)

    def _get_relevant_memory(
        self,
        history: list[ConversationTurn],
        current_query: str,
        limit: int = 3,
        user_id: str | None = None,
    ) -> str:
        memory_parts = []
        if history:
            for turn in history[-limit:]:
                memory_parts.append(f"Q: {turn.query}\nA: {turn.response}")
        if self.short_term_memory and user_id:
            try:
                redis_memories = self.short_term_memory.get_recent(user_id, limit=limit)
                for m in redis_memories:
                    content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
                    if content and content not in "\n\n---\n\n".join(memory_parts):
                        memory_parts.append(content)
            except Exception:
                pass
        return "\n\n---\n\n".join(memory_parts) if memory_parts else ""

    def evolve_strategy(
        self,
        current_strategy: ContextStrategy,
        iteration: int,
        response_quality: float,
    ) -> ContextStrategy:
        new_strategy = ContextStrategy(
            version=current_strategy.version + 1,
            rag_weight=current_strategy.rag_weight,
            memory_weight=current_strategy.memory_weight,
            retrieval_depth=current_strategy.retrieval_depth,
            similarity_threshold=current_strategy.similarity_threshold,
            max_iterations=current_strategy.max_iterations,
            iteration_count=iteration,
            last_response_quality=response_quality,
        )
        if iteration == 2 and response_quality < 0.7:
            new_strategy.rag_weight = min(0.85, current_strategy.rag_weight + 0.15)
            new_strategy.memory_weight = 1.0 - new_strategy.rag_weight
            new_strategy.retrieval_depth = min(10, current_strategy.retrieval_depth + 2)
        elif iteration >= 3:
            new_strategy.rag_weight = 0.90
            new_strategy.memory_weight = 0.10
            new_strategy.retrieval_depth = min(15, current_strategy.retrieval_depth + 3)
            new_strategy.similarity_threshold = max(0.6, current_strategy.similarity_threshold - 0.1)
        new_strategy.validate_weights()
        return new_strategy

    def should_continue(
        self,
        iteration: int,
        max_iterations: int,
        response_quality: float = 0.0,
    ) -> bool:
        if iteration >= max_iterations:
            return False
        if response_quality >= 0.85:
            return False
        return True


# -----------------------------------------------------------------------------
# Supervisor graph and factory
# -----------------------------------------------------------------------------


def _build_supervisor_nodes(
    context_manager: ContextManager,
    rag_agent: Any,
    short_term_memory: ShortTermMemory | None,
):
    """Build node functions that close over context_manager, rag_agent, short_term_memory."""

    async def synthesize_context_node(state: AgentState) -> dict[str, Any]:
        combined_context = await context_manager.synthesize_context(
            query=state["query"],
            patient_id=state.get("patient_id"),
            conversation_history=state.get("conversation_history") or [],
            strategy=state["context_strategy"],
            document_source=state.get("document_source"),
        )
        return {"combined_context": combined_context}

    async def delegate_to_rag_node(state: AgentState) -> dict[str, Any]:
        conv = state.get("conversation_history") or []
        history_block = ""
        if conv:
            history_block = "\nCONVERSATION HISTORY (this session):\n" + "\n".join(
                f"Q: {t.query}\nA: {t.response}" for t in conv
            ) + "\n"
        rag_input = f"""
PATIENT ID: {state.get('patient_id', 'Not specified')}

SYNTHESIZED CONTEXT:
{state.get('combined_context', '')}
{history_block}

CURRENT QUERY: {state['query']}

Please retrieve patient information and answer the query based on the patient's history and the conversation above. If the user asks for chat history or what was discussed, summarize or list the conversation history from the CONVERSATION HISTORY section.
"""
        result = await rag_agent.ainvoke({"messages": [HumanMessage(content=rag_input)]})
        rag_response = result["messages"][-1].content if result.get("messages") else ""
        quality_score = 0.7 + (0.1 * state.get("current_iteration", 1))
        strategy = state["context_strategy"]
        updated_strategy = strategy.model_copy(update={"last_response_quality": quality_score})
        return {
            "rag_response": rag_response,
            "sources_used": ["patient_history_db"],
            "context_strategy": updated_strategy,
        }

    def check_continuation_node(state: AgentState) -> Literal["evolve", "finalize"]:
        strategy = state["context_strategy"]
        current_iter = state.get("current_iteration", 1)
        quality = strategy.last_response_quality
        if context_manager.should_continue(
            iteration=current_iter,
            max_iterations=strategy.max_iterations,
            response_quality=quality,
        ):
            return "evolve"
        return "finalize"

    def evolve_strategy_node(state: AgentState) -> dict[str, Any]:
        new_strategy = context_manager.evolve_strategy(
            current_strategy=state["context_strategy"],
            iteration=state.get("current_iteration", 1),
            response_quality=state["context_strategy"].last_response_quality,
        )
        strategy_history = list(state.get("strategy_history") or [])
        strategy_history.append(state["context_strategy"])
        return {
            "context_strategy": new_strategy,
            "current_iteration": state.get("current_iteration", 1) + 1,
            "strategy_history": strategy_history,
        }

    def finalize_response_node(state: AgentState) -> dict[str, Any]:
        new_turn = ConversationTurn(
            turn_number=len(state.get("conversation_history") or []) + 1,
            query=state["query"],
            response=state.get("rag_response", ""),
            context_used=(state.get("combined_context") or "")[:500] + "...",
        )
        if short_term_memory:
            try:
                user_id = state.get("patient_id") or "default"
                short_term_memory.add(
                    user_id=user_id,
                    query=state["query"],
                    response=state.get("rag_response", ""),
                    metadata={"patient_id": state.get("patient_id"), "iteration": state.get("current_iteration")},
                )
            except Exception:
                pass
        final_output = f"""
## MEDICAL Q&A RESPONSE

**Query:** {state['query']}

**Patient ID:** {state.get('patient_id', 'Not specified')}

**Response:**
{state.get('rag_response', '')}

---

**Evolution Statistics:**
- Iterations: {state.get('current_iteration', 1)}
- Final Strategy: v{state['context_strategy'].version}
- Final Weights: RAG={state['context_strategy'].rag_weight:.0%}, Memory={state['context_strategy'].memory_weight:.0%}
- Estimated Quality: {state['context_strategy'].last_response_quality:.2f}
"""
        return {
            "final_response": final_output,
            "conversation_history": (state.get("conversation_history") or []) + [new_turn],
            "should_continue": False,
        }

    return (
        synthesize_context_node,
        delegate_to_rag_node,
        check_continuation_node,
        evolve_strategy_node,
        finalize_response_node,
    )


def create_supervisor(
    rag_pipeline: RAGPipeline,
    rag_agent: Any = None,
    qdrant_client: Any = None,
    short_term_memory: ShortTermMemory | None = None,
    rag_config: RAGConfig | None = None,
):
    """Create the self-evolving supervisor graph (compiled).

    Args:
        rag_pipeline: RAG pipeline (Qdrant) for retrieval.
        rag_agent: Message-based RAG agent (e.g. from create_rag_agent()). If None, uses create_rag_agent(rag_config).
        qdrant_client: Optional shared Qdrant client for pipeline and memory.
        short_term_memory: Optional Qdrant-backed short-term memory. If None, only in-memory history is used.
        rag_config: Used for create_rag_agent if rag_agent is None. Defaults to RAGConfig.from_env().

    Returns:
        Compiled LangGraph (invoke/ainvoke with AgentState).
    """
    config = rag_config or RAGConfig.from_env()
    agent = rag_agent or create_rag_agent(config)
    redis_vl = RedisVLAdapter(rag_pipeline, qdrant_client=qdrant_client)
    context_manager = ContextManager(redis_vl, short_term_memory=short_term_memory)

    (
        synthesize_context_node,
        delegate_to_rag_node,
        check_continuation_node,
        evolve_strategy_node,
        finalize_response_node,
    ) = _build_supervisor_nodes(context_manager, agent, short_term_memory)

    workflow = StateGraph(AgentState)
    workflow.add_node("synthesize_context", synthesize_context_node)
    workflow.add_node("delegate_to_rag", delegate_to_rag_node)
    workflow.add_node("evolve_strategy", evolve_strategy_node)
    workflow.add_node("finalize_response", finalize_response_node)

    workflow.add_edge(START, "synthesize_context")
    workflow.add_edge("synthesize_context", "delegate_to_rag")
    workflow.add_conditional_edges(
        "delegate_to_rag",
        check_continuation_node,
        {"evolve": "evolve_strategy", "finalize": "finalize_response"},
    )
    workflow.add_edge("evolve_strategy", "synthesize_context")
    workflow.add_edge("finalize_response", END)

    return workflow.compile()


async def query_medical_system(
    query: str,
    patient_id: str | None = None,
    document_source: str | None = None,
    conversation_history: list[ConversationTurn] | None = None,
    supervisor_graph: Any = None,
    rag_pipeline: RAGPipeline | None = None,
    rag_config: RAGConfig | None = None,
) -> dict[str, Any]:
    """Run a medical query through the self-evolving supervisor.

    Args:
        query: User question.
        patient_id: Optional patient id for scoped retrieval.
        conversation_history: Optional previous turns (for multi-turn).
        supervisor_graph: Pre-built compiled graph. If None, builds one from rag_pipeline and rag_config.
        rag_pipeline: Used to build supervisor if supervisor_graph is None. Defaults to RAGPipeline(rag_config).
        rag_config: Used to build pipeline/agent if not provided.

    Returns:
        Final AgentState (includes final_response, conversation_history, etc.).
    """
    if supervisor_graph is None:
        config = rag_config or RAGConfig.from_env()
        pipeline = rag_pipeline or RAGPipeline(config)
        supervisor_graph = create_supervisor(pipeline, rag_config=config, short_term_memory=None)

    initial_strategy = ContextStrategy(
        version=1,
        rag_weight=0.70,
        memory_weight=0.30,
        retrieval_depth=5,
        max_iterations=3,
    )
    initial_state: AgentState = {
        "query": query,
        "patient_id": patient_id,
        "document_source": document_source,
        "rag_results": [],
        "conversation_history": conversation_history or [],
        "context_strategy": initial_strategy,
        "combined_context": "",
        "rag_response": "",
        "sources_used": [],
        "current_iteration": 1,
        "should_continue": True,
        "final_response": "",
        "strategy_history": [],
    }
    result = await supervisor_graph.ainvoke(initial_state)
    return result
