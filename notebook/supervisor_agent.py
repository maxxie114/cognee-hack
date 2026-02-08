import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Self-Evolving Supervisor Multi-Agent Medical Q&A System

    This notebook implements a self-evolving supervisor that manages context selection strategy while delegating to an Agentic RAG sub-agent.
    ## Summary

    This notebook implements a complete **Self-Evolving Supervisor Multi-Agent System** for medical Q&A with the following characteristics:

    ### Architecture:
    - **Self-Evolving Supervisor**: Manages context selection strategy that evolves across iterations
    - **Agentic RAG Sub-Agent**: Retrieves patient history with query reformulation
    - **Redis VL Integration**: Real RAG pipeline (RedisVL) for patient/document retrieval

    ### Evolution Mechanism:
    - Iteration 1: Balanced approach (70% RAG, 30% Memory)
    - Iteration 2: Increases RAG weight if quality is low
    - Iteration 3: Maximizes RAG (90%), minimizes memory (10%)

    ### Context Sources:
    - Patient history from Redis VL (RAG)
    - Conversation memory from current session
    - Synthesized context based on strategy weights

    ### Features:
    - Single session memory
    - No web search
    - No complex evaluation (simplified quality estimation)
    - Configurable iteration limits
    - Strategy history tracking

    Ready for production integration with actual Redis VL!
    """)
    return


@app.cell
def _():
    import os
    import sys
    import json
    import operator
    from typing import Annotated, TypedDict, List, Dict, Any, Optional, Literal
    from datetime import datetime
    from pathlib import Path
    from dataclasses import dataclass, field

    from dotenv import load_dotenv
    from pydantic import BaseModel, Field
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import HumanMessage, AIMessage
    from langgraph.graph import StateGraph, START, END
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    # Load environment variables
    load_dotenv()

    # Add project src to path for RAG pipeline (real RedisVL)
    project_root = Path(os.getcwd())
    if not (project_root / "src").exists() and (project_root.parent / "src").exists():
        project_root = project_root.parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path and src_path.exists():
        sys.path.insert(0, str(src_path))

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    _llm_backend = "OpenAI"

    # Verify setup
    print("✅ Dependencies loaded")
    print(f"LLM backend: {_llm_backend}")
    print(f"API key set: {bool(os.getenv('OPENAI_API_KEY'))}")
    return (
        Annotated,
        Any,
        BaseModel,
        ChatOpenAI,
        Dict,
        END,
        Field,
        HumanMessage,
        List,
        Literal,
        Optional,
        START,
        StateGraph,
        TypedDict,
        create_agent,
        datetime,
        operator,
        tool,
    )


@app.cell
def _(
    Annotated,
    Any,
    BaseModel,
    Dict,
    Field,
    List,
    Optional,
    TypedDict,
    datetime,
    operator,
):
    class ContextStrategy(BaseModel):
        """The supervisor's evolving context selection strategy"""
        version: int

        # Weight parameters (should sum to 1.0)
        rag_weight: float = Field(default=0.70, ge=0.0, le=1.0)
        memory_weight: float = Field(default=0.30, ge=0.0, le=1.0)

        # Retrieval parameters
        retrieval_depth: int = Field(default=5, ge=1, le=20)
        similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

        # Iteration control
        max_iterations: int = Field(default=3, ge=1, le=5)

        # Strategy performance tracking
        iteration_count: int = 0
        last_response_quality: float = 0.0

        def validate_weights(self):
            """Ensure weights sum to approximately 1.0"""
            total = self.rag_weight + self.memory_weight
            if not 0.99 <= total <= 1.01:
                # Normalize
                self.rag_weight /= total
                self.memory_weight /= total


    class RetrievedDocument(BaseModel):
        """A document retrieved from Redis VL"""
        content: str
        source: str
        score: float
        metadata: Dict[str, Any] = Field(default_factory=dict)


    class ConversationTurn(BaseModel):
        """A single turn in the conversation"""
        turn_number: int
        query: str
        response: str
        timestamp: datetime = Field(default_factory=datetime.now)
        context_used: str = ""


    class AgentState(TypedDict):
        """
        Main graph state for the self-evolving supervisor system
        """
        # User Input
        query: str

        # Context Sources
        patient_id: Optional[str]  # For patient-specific RAG
        rag_results: Annotated[List[RetrievedDocument], operator.add]
        conversation_history: Annotated[List[ConversationTurn], operator.add]

        # Supervisor's Evolving Context
        context_strategy: ContextStrategy
        combined_context: str

        # RAG Agent Output
        rag_response: str
        reformulated_query: str
        sources_used: List[str]

        # Evolution State
        current_iteration: int
        should_continue: bool

        # Final Output
        final_response: str
        strategy_history: List[ContextStrategy]


    print("✅ Data models defined")
    print(f"ContextStrategy fields: {list(ContextStrategy.model_fields.keys())}")
    print(f"AgentState keys: {[k for k in AgentState.__annotations__.keys()]}")
    return AgentState, ContextStrategy, ConversationTurn, RetrievedDocument


@app.cell
def _(mo):
    mo.md(r"""
    ## Redis VL (RAG Pipeline)

    Real Redis VL via the project's RAG pipeline (RedisVL + LangChain). An adapter exposes the same interface (search) for the supervisor.
    """)
    return


@app.cell
def _(List, RetrievedDocument):
    import asyncio
    from RAG import RAGPipeline, RAGConfig


    class RedisVLAdapter:
        """Adapter: RAG pipeline (RedisVL) exposing search(patient_id, query, k, threshold) → List[RetrievedDocument]."""

        def __init__(self, pipeline: RAGPipeline):
            self.pipeline = pipeline

        async def search(
            self,
            patient_id: str,
            query: str,
            k: int = 5,
            threshold: float = 0.75
        ) -> List[RetrievedDocument]:
            """Vector search via RAG pipeline; map to RetrievedDocument and filter by threshold."""
            # Optional: bias query with patient_id for patient-specific docs if index stores patient_id
            effective_query = f"{query}" if not patient_id else f"Patient {patient_id}: {query}"
            docs = await asyncio.to_thread(
                self.pipeline.retrieve,
                effective_query,
                top_k=k
            )
            results = []
            for d in docs:
                score = float(d.metadata.get("score") or 0.0)
                if score >= threshold:
                    results.append(RetrievedDocument(
                        content=d.page_content,
                        source=d.metadata.get("source", ""),
                        score=score,
                        metadata=dict(d.metadata)
                    ))
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:k]


    # Initialize real RAG pipeline and adapter
    rag_config = RAGConfig.from_env()
    rag_pipeline = RAGPipeline(rag_config)
    redis_vl = RedisVLAdapter(rag_pipeline)
    print("✅ Redis VL (RAG pipeline) initialized")
    print(f"Index: {rag_config.index_name}, Top-K: {rag_config.top_k}")
    return rag_config, redis_vl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Short-term Memory (Redis VL)

    Redis-backed short-term memory per the [Redis cookbook](https://redis.io/docs/): index for agent conversation turns (query/response) with vector search. Used to augment in-session context with recent memories from Redis.
    """)
    return


@app.cell
def _(rag_config):
    import json
    import uuid
    from datetime import datetime
    from typing import Any, Dict, List, Optional
    from redis import Redis
    from redisvl.index import SearchIndex
    from redisvl.schema.schema import IndexSchema
    from redisvl.query import VectorQuery
    from langchain_openai import OpenAIEmbeddings

    # Redis client (same connection as RAG pipeline)
    redis_client = Redis.from_url(rag_config.redis_url, decode_responses=False)

    # Schema for short-term memory (OpenAI embedding dims = 1536)
    MEMORY_EMBED_DIMS = getattr(rag_config, "embedding_dimensions", 1536)
    memory_schema = IndexSchema.from_dict({
        "index": {
            "name": "agent_short_term_memory",
            "prefix": "memory:",
            "key_separator": ":",
            "storage_type": "json",
        },
        "fields": [
            {"name": "content", "type": "text"},
            {"name": "memory_type", "type": "tag"},
            {"name": "metadata", "type": "text"},
            {"name": "created_at", "type": "text"},
            {"name": "user_id", "type": "tag"},
            {"name": "memory_id", "type": "tag"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "flat",
                    "dims": MEMORY_EMBED_DIMS,
                    "distance_metric": "cosine",
                    "datatype": "float32",
                },
            },
        ],
    })

    # Create short-term memory index (skip if already exists)
    short_term_memory_index = SearchIndex(
        schema=memory_schema,
        redis_client=redis_client,
        overwrite=False,
    )
    try:
        if not short_term_memory_index.exists():
            short_term_memory_index.create()
            print("✅ Short-term memory index created")
        else:
            print("✅ Short-term memory index ready")
    except Exception as e:
        print(f"⚠️ Short-term memory index: {e}")


    class ShortTermMemory:
        """Redis VL–backed short-term memory: add turns, get recent, or search by similarity."""

        def __init__(self, index: SearchIndex, embeddings: OpenAIEmbeddings):
            self.index = index
            self.embeddings = embeddings

        def add(
            self,
            user_id: str,
            query: str,
            response: str,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> str:
            """Store a conversation turn. Returns memory_id."""
            memory_id = str(uuid.uuid4())
            content = f"Q: {query}\nA: {response}"
            created_at = datetime.utcnow().isoformat() + "Z"
            meta_str = json.dumps(metadata or {})
            vector = self.embeddings.embed_query(content)
            record = {
                "content": content,
                "memory_type": "short_term",
                "metadata": meta_str,
                "created_at": created_at,
                "user_id": user_id,
                "memory_id": memory_id,
                "embedding": vector,
            }
            self.index.upsert([record])
            return memory_id

        def get_recent(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
            """Fetch recent memories for user (by vector search then sort by created_at)."""
            if not self.index.exists():
                return []
            query_vec = self.embeddings.embed_query("recent conversation memory")
            q = VectorQuery(
                vector=query_vec,
                vector_field_name="embedding",
                return_fields=["content", "created_at", "metadata"],
                num_results=limit * 2,
                filter_expression=f"@user_id:{{{user_id}}}",
            )
            raw = self.index.query(q)
            results = raw if isinstance(raw, list) else getattr(raw, "results", [raw]) or []
            for r in results:
                if isinstance(r, dict) and "created_at" not in r:
                    r["created_at"] = ""
            results.sort(key=lambda x: (x.get("created_at") or ""), reverse=True)
            return results[:limit]

        def search(self, user_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
            """Semantic search over user's short-term memories."""
            if not self.index.exists():
                return []
            query_vec = self.embeddings.embed_query(query)
            q = VectorQuery(
                vector=query_vec,
                vector_field_name="embedding",
                return_fields=["content", "created_at", "metadata"],
                num_results=k,
                filter_expression=f"@user_id:{{{user_id}}}",
            )
            raw = self.index.query(q)
            return raw if isinstance(raw, list) else getattr(raw, "results", [raw]) or []


    short_term_memory = ShortTermMemory(
        short_term_memory_index,
        OpenAIEmbeddings(model=rag_config.embedding_model),
    )
    print("✅ ShortTermMemory ready (add / get_recent / search)")
    return Any, Dict, List, Optional, datetime, short_term_memory


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Context Manager

    Manages the context selection strategy and synthesizes context from multiple sources.
    """)
    return


@app.cell
def _(
    ChatOpenAI,
    ContextStrategy,
    ConversationTurn,
    List,
    Optional,
    redis_vl,
    short_term_memory,
):
    class ContextManager:
        """Manages context synthesis and strategy evolution"""

        def __init__(self, db, short_term_memory=None):
            """db: RedisVLAdapter with async search(patient_id, query, k, threshold). short_term_memory: optional ShortTermMemory (Redis VL)."""
            self.db = db
            self.short_term_memory = short_term_memory
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        async def synthesize_context(
            self, 
            query: str, 
            patient_id: Optional[str],
            conversation_history: List[ConversationTurn],
            strategy: ContextStrategy
        ) -> str:
            """
            Synthesize context from RAG (patient history) and conversation memory
            according to the current strategy weights
            """

            # 1. Retrieve from RAG (patient history) - weighted
            rag_context = ""
            if patient_id:
                rag_results = await self.db.search(
                    patient_id=patient_id,
                    query=query,
                    k=strategy.retrieval_depth,
                    threshold=strategy.similarity_threshold
                )
                if rag_results:
                    rag_docs = [f"[Score: {r.score:.2f}] {r.content}" for r in rag_results]
                    rag_context = "\n\n".join(rag_docs)

            # 2. Get relevant conversation memory (in-memory + Redis short-term) - weighted
            user_id = patient_id or "default"
            memory_context = self._get_relevant_memory(conversation_history, query, user_id=user_id)

            # 3. Synthesize with strategy weights
            synthesis_prompt = f"""
            You are synthesizing medical context for a patient query.

            WEIGHTS (determines importance):
            - Patient History (RAG): {strategy.rag_weight:.0%}
            - Conversation Memory: {strategy.memory_weight:.0%}

            PATIENT HISTORY (from records):
            {rag_context if rag_context else "[No relevant patient history found]"}

            CONVERSATION HISTORY (from this session):
            {memory_context if memory_context else "[No previous conversation]"}

            CURRENT QUERY: {query}

            INSTRUCTIONS:
            1. Synthesize the above information into a coherent context
            2. Prioritize patient history according to its weight ({strategy.rag_weight:.0%})
            3. Include relevant conversation context according to its weight ({strategy.memory_weight:.0%})
            4. Highlight any contradictions or gaps
            5. Keep synthesis focused and clinically relevant

            Provide a concise, medically-focused context summary.
            """

            response = await self.llm.ainvoke(synthesis_prompt)
            return response.content

        def _get_relevant_memory(
            self, 
            history: List[ConversationTurn], 
            current_query: str,
            limit: int = 3,
            user_id: Optional[str] = None,
        ) -> str:
            """Extract relevant conversation turns from in-memory history and Redis short-term memory."""
            memory_parts = []

            # In-memory: last N turns
            if history:
                recent_turns = history[-limit:]
                for turn in recent_turns:
                    memory_parts.append(f"Q: {turn.query}\nA: {turn.response}")

            # Redis short-term: recent memories for this user
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
            response_quality: float
        ) -> ContextStrategy:
            """
            Evolve the context selection strategy based on iteration and quality
            This is the self-evolution mechanism
            """

            new_strategy = ContextStrategy(
                version=current_strategy.version + 1,
                rag_weight=current_strategy.rag_weight,
                memory_weight=current_strategy.memory_weight,
                retrieval_depth=current_strategy.retrieval_depth,
                similarity_threshold=current_strategy.similarity_threshold,
                max_iterations=current_strategy.max_iterations,
                iteration_count=iteration,
                last_response_quality=response_quality
            )

            # Evolution logic based on iteration
            if iteration == 1:
                # First iteration: Standard balanced approach
                pass
            elif iteration == 2:
                # Second iteration: If quality is low, adjust weights
                if response_quality < 0.7:
                    # Increase RAG weight for more factual grounding
                    new_strategy.rag_weight = min(0.85, current_strategy.rag_weight + 0.15)
                    new_strategy.memory_weight = 1.0 - new_strategy.rag_weight
                    # Also increase retrieval depth
                    new_strategy.retrieval_depth = min(10, current_strategy.retrieval_depth + 2)
            else:
                # Third iteration: Maximize RAG, minimize memory
                new_strategy.rag_weight = 0.90
                new_strategy.memory_weight = 0.10
                new_strategy.retrieval_depth = min(15, current_strategy.retrieval_depth + 3)
                # Lower threshold to get more results
                new_strategy.similarity_threshold = max(0.6, current_strategy.similarity_threshold - 0.1)

            new_strategy.validate_weights()
            return new_strategy

        def should_continue(
            self, 
            iteration: int, 
            max_iterations: int,
            response_quality: float = 0.0
        ) -> bool:
            """Determine if we should continue iterating"""
            if iteration >= max_iterations:
                return False

            # If quality is good enough, stop early
            if response_quality >= 0.85:
                return False

            return True

    # Initialize context manager (redis_vl from section 3, short_term_memory from section 4)
    context_manager = ContextManager(redis_vl, short_term_memory=short_term_memory)
    print("✅ Context Manager initialized")
    return (ContextManager,)


@app.cell
def _(ChatOpenAI, create_agent, redis_vl, tool):
    # Tools for Agentic RAG

    @tool
    async def retrieve_patient_history(
        patient_id: str, 
        query: str, 
        k: int = 5
    ) -> str:
        """
        Retrieve relevant patient history from Redis VL vector database (RAG pipeline).
        Use this to get factual patient information.
        """
        results = await redis_vl.search(patient_id, query, k=k)

        if not results:
            return "No relevant patient records found for this query."

        formatted_results = []
        for i, doc in enumerate(results, 1):
            formatted_results.append(
                f"{i}. [Relevance: {doc.score:.2f}] {doc.content[:200]}..."
            )

        return "\n".join(formatted_results)

    @tool
    def analyze_retrieved_context(context: str) -> str:
        """
        Analyze retrieved patient context to identify:
        - Key medical facts
        - Current medications
        - Recent diagnoses
        - Missing information
        """
        # This would use LLM in production
        lines = context.split("\n")
        key_points = []

        for line in lines:
            if any(keyword in line.lower() for keyword in ["diagnosed", "medication", "treatment", "lab", "test"]):
                key_points.append(line.strip())

        if key_points:
            return "Key points extracted:\n" + "\n".join(key_points[:5])
        return "No specific medical key points identified in context."

    @tool
    def reformulate_query(original_query: str, conversation_context: str) -> str:
        """
        Reformulate the query to improve retrieval based on conversation context.
        Example: "What about his heart?" → "What is the patient's cardiac history?"
        """
        # Simple reformulation rules
        pronouns = ["he", "she", "his", "her", "him", "patient"]

        query = original_query

        # Check if query is vague (contains pronouns without clear reference)
        if any(p in query.lower() for p in pronouns):
            # Try to extract patient reference from conversation context
            if "patient" in conversation_context.lower():
                # Keep as is, assume patient context is clear
                pass

        # Add medical domain context if missing
        medical_terms = ["medication", "diagnosis", "treatment", "symptoms", "history"]
        if not any(term in query.lower() for term in medical_terms):
            # Query might be too vague
            query = f"Patient {query}"

        return f"Reformulated: {query}"

    # Create the Agentic RAG agent
    rag_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    rag_agent = create_agent(
        model=rag_llm,
        tools=[retrieve_patient_history, analyze_retrieved_context, reformulate_query],
        system_prompt="""
        You are a medical information retrieval specialist. Your job is to:

        1. RETRIEVE relevant patient history using the retrieve_patient_history tool
        2. ANALYZE the retrieved context for key medical facts
        3. REFORMULATE queries if needed for better retrieval
        4. SYNTHESIZE a comprehensive, accurate medical response

        IMPORTANT:
        - Always base your answers on retrieved patient records
        - If information is missing, clearly state what is not known
        - Be precise with medical facts, dates, and medication names
        - Consider conversation context when interpreting ambiguous queries

        You have access to:
        - retrieve_patient_history: Get patient records from database
        - analyze_retrieved_context: Extract key medical information
        - reformulate_query: Improve query clarity for better retrieval
        """,
    )

    print("✅ Agentic RAG sub-agent created")
    print("Tools: retrieve_patient_history, analyze_retrieved_context, reformulate_query")
    return (rag_agent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Supervisor Graph Nodes

    Implement the self-evolving supervisor workflow.
    """)
    return


@app.cell
def _(
    AgentState,
    ChatOpenAI,
    ContextManager,
    ConversationTurn,
    Dict,
    HumanMessage,
    Literal,
    rag_agent,
    redis_vl,
    short_term_memory,
):
    # Initialize components (redis_vl from section 3, short_term_memory from section 4)
    context_manager = ContextManager(redis_vl, short_term_memory=short_term_memory)
    supervisor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    async def synthesize_context_node(state: AgentState) -> Dict:
        """
        Node 1: Synthesize context from RAG and conversation memory
        according to the current strategy
        """
        print(f"\n🔄 ITERATION {state['current_iteration']} - Context Synthesis")
        print(f"   Strategy v{state['context_strategy'].version}: RAG={state['context_strategy'].rag_weight:.0%}, Memory={state['context_strategy'].memory_weight:.0%}")

        combined_context = await context_manager.synthesize_context(
            query=state["query"],
            patient_id=state.get("patient_id"),
            conversation_history=state["conversation_history"],
            strategy=state["context_strategy"]
        )

        print(f"   ✓ Context synthesized ({len(combined_context)} chars)")

        return {"combined_context": combined_context}


    async def delegate_to_rag_node(state: AgentState) -> Dict:
        """
        Node 2: Delegate to Agentic RAG with synthesized context
        """
        print(f"\n📤 DELEGATING to Agentic RAG")

        # Prepare message for RAG agent with context
        rag_input = f"""
        PATIENT ID: {state.get('patient_id', 'Not specified')}

        SYNTHESIZED CONTEXT:
        {state['combined_context']}

        CURRENT QUERY: {state['query']}

        Please retrieve patient information and answer the query based on the patient's history.
        """

        # Invoke RAG agent
        result = await rag_agent.ainvoke({"messages": [HumanMessage(content=rag_input)]})

        # Extract response
        rag_response = result["messages"][-1].content

        # Estimate quality (simplified - in production would use proper evaluation)
        quality_score = 0.7 + (0.1 * state["current_iteration"])  # Increases with iteration

        print(f"   ✓ RAG response received ({len(rag_response)} chars)")
        print(f"   ✓ Estimated quality: {quality_score:.2f}")

        return {
            "rag_response": rag_response,
            "sources_used": ["patient_history_db"],
            "context_strategy": state["context_strategy"].copy(update={
                "last_response_quality": quality_score
            })
        }


    def check_continuation_node(state: AgentState) -> Literal["evolve", "finalize"]:
        """
        Node 3: Check if we should continue iterating or finalize
        """
        strategy = state["context_strategy"]
        current_iter = state["current_iteration"]
        quality = strategy.last_response_quality

        should_continue = context_manager.should_continue(
            iteration=current_iter,
            max_iterations=strategy.max_iterations,
            response_quality=quality
        )

        if should_continue:
            print(f"\n⚙️  CONTINUING to iteration {current_iter + 1}")
            return "evolve"
        else:
            print(f"\n✅ FINALIZING after {current_iter} iteration(s)")
            return "finalize"


    def evolve_strategy_node(state: AgentState) -> Dict:
        """
        Node 4: Evolve the context strategy for next iteration
        This is the SELF-EVOLUTION mechanism
        """
        print(f"\n🧬 EVOLVING Strategy")

        new_strategy = context_manager.evolve_strategy(
            current_strategy=state["context_strategy"],
            iteration=state["current_iteration"],
            response_quality=state["context_strategy"].last_response_quality
        )

        # Track strategy history
        strategy_history = state.get("strategy_history", [])
        strategy_history.append(state["context_strategy"])

        print(f"   ✓ Strategy evolved to v{new_strategy.version}")
        print(f"   ✓ New weights: RAG={new_strategy.rag_weight:.0%}, Memory={new_strategy.memory_weight:.0%}")
        print(f"   ✓ Retrieval depth: {new_strategy.retrieval_depth}")

        return {
            "context_strategy": new_strategy,
            "current_iteration": state["current_iteration"] + 1,
            "strategy_history": strategy_history
        }


    def finalize_response_node(state: AgentState) -> Dict:
        """
        Node 5: Finalize and format the response
        """
        print(f"\n📝 FINALIZING Response")

        # Add to conversation history
        new_turn = ConversationTurn(
            turn_number=len(state["conversation_history"]) + 1,
            query=state["query"],
            response=state["rag_response"],
            context_used=state["combined_context"][:500] + "..."
        )

        # Persist to Redis short-term memory (section 4)
        try:
            user_id = state.get("patient_id") or "default"
            short_term_memory.add(
                user_id=user_id,
                query=state["query"],
                response=state["rag_response"],
                metadata={"patient_id": state.get("patient_id"), "iteration": state["current_iteration"]},
            )
        except Exception:
            pass

        final_output = f"""
    ## MEDICAL Q&A RESPONSE

    **Query:** {state['query']}

    **Patient ID:** {state.get('patient_id', 'Not specified')}

    **Response:**
    {state['rag_response']}

    ---

    **Evolution Statistics:**
    - Iterations: {state['current_iteration']}
    - Final Strategy: v{state['context_strategy'].version}
    - Final Weights: RAG={state['context_strategy'].rag_weight:.0%}, Memory={state['context_strategy'].memory_weight:.0%}
    - Estimated Quality: {state['context_strategy'].last_response_quality:.2f}
    """

        print(f"   ✓ Response finalized")

        return {
            "final_response": final_output,
            "conversation_history": state["conversation_history"] + [new_turn],
            "should_continue": False
        }

    print("✅ Supervisor nodes defined")
    print(f"Nodes: synthesize_context → delegate_to_rag → check_continuation → [evolve|finalize]")
    return (
        check_continuation_node,
        delegate_to_rag_node,
        evolve_strategy_node,
        finalize_response_node,
        synthesize_context_node,
    )


@app.cell
def _(
    AgentState,
    END,
    START,
    StateGraph,
    check_continuation_node,
    delegate_to_rag_node,
    evolve_strategy_node,
    finalize_response_node,
    synthesize_context_node,
):
    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("synthesize_context", synthesize_context_node)
    workflow.add_node("delegate_to_rag", delegate_to_rag_node)
    workflow.add_node("evolve_strategy", evolve_strategy_node)
    workflow.add_node("finalize_response", finalize_response_node)

    # Define edges
    workflow.add_edge(START, "synthesize_context")
    workflow.add_edge("synthesize_context", "delegate_to_rag")

    # Conditional edge after RAG
    workflow.add_conditional_edges(
        "delegate_to_rag",
        check_continuation_node,
        {
            "evolve": "evolve_strategy",
            "finalize": "finalize_response"
        }
    )

    # Loop back for evolution
    workflow.add_edge("evolve_strategy", "synthesize_context")

    # End after finalization
    workflow.add_edge("finalize_response", END)

    # Compile
    self_evolving_system = workflow.compile()

    print("✅ Self-Evolving Supervisor Multi-Agent System compiled!")
    print("\nGraph Structure:")
    print("  START → synthesize_context → delegate_to_rag → check_continuation")
    print("                                          ↓")
    print("                              [evolve] → evolve_strategy → (loop back)")
    print("                                          ↓")
    print("                              [finalize] → finalize_response → END")
    return (self_evolving_system,)


@app.cell
async def _(
    ContextStrategy,
    ConversationTurn,
    List,
    Optional,
    self_evolving_system,
):
    async def query_medical_system(
        query: str, 
        patient_id: Optional[str] = None,
        conversation_history: List[ConversationTurn] = None
    ):
        """Run a medical query through the self-evolving system"""

        # Initialize state
        initial_strategy = ContextStrategy(
            version=1,
            rag_weight=0.70,
            memory_weight=0.30,
            retrieval_depth=5,
            max_iterations=3
        )

        initial_state = {
            "query": query,
            "patient_id": patient_id,
            "rag_results": [],
            "conversation_history": conversation_history or [],
            "context_strategy": initial_strategy,
            "combined_context": "",
            "rag_response": "",
            "sources_used": [],
            "current_iteration": 1,
            "should_continue": True,
            "final_response": "",
            "strategy_history": []
        }

        print("="*70)
        print("🚀 SELF-EVOLVING SUPERVISOR MULTI-AGENT SYSTEM")
        print("="*70)
        print(f"\n📋 QUERY: {query}")
        print(f"👤 PATIENT: {patient_id or 'Not specified'}")
        print(f"💬 HISTORY: {len(conversation_history or [])} previous turns")

        # Run the graph
        result = await self_evolving_system.ainvoke(initial_state)

        print("\n" + "="*70)
        print(result["final_response"])

        return result

    # Test 1: Simple query with patient history
    print("\n" + "🧪 TEST 1: Basic Patient Query" + "\n" + "="*70)
    result1 = await query_medical_system(
        query="What medications is this patient currently taking?",
        patient_id="patient_001"
    )
    return query_medical_system, result1


@app.cell
async def _(query_medical_system, result1):
    # Test 2: Follow-up question (tests memory integration)
    print("\n" + "🧪 TEST 2: Follow-up with Memory" + "\n" + "="*70)
    result2 = await query_medical_system(
        query="What about his heart condition?",
        patient_id="patient_001",
        conversation_history=result1["conversation_history"]
    )
    return (result2,)


@app.cell
async def _(query_medical_system):
    # Test 3: Different patient (diabetes case)
    print("\n" + "🧪 TEST 3: Diabetes Patient" + "\n" + "="*70)
    result3 = await query_medical_system(
        query="What is the patient's current diabetes management plan?",
        patient_id="patient_002"
    )
    return (result3,)


@app.cell
async def _(query_medical_system):
    # Test 4: Complex query requiring multiple iterations
    print("\n" + "🧪 TEST 4: Complex Query (will likely evolve)" + "\n" + "="*70)
    result4 = await query_medical_system(
        query="Summarize the complete medical history and provide treatment recommendations",
        patient_id="patient_003"
    )
    return (result4,)


@app.cell
def _(Dict, List, result1, result2, result3, result4):
    def analyze_strategy_evolution(results: List[Dict]):
        """Analyze how strategies evolved across test cases"""

        print("\n" + "="*70)
        print("📊 STRATEGY EVOLUTION ANALYSIS")
        print("="*70 + "\n")

        for i, result in enumerate(results, 1):
            print(f"Test Case {i}:")
            print(f"  Query: {result['query'][:60]}...")
            print(f"  Iterations: {result['current_iteration']}")
            print(f"  Final Strategy v{result['context_strategy'].version}")
            print(f"  - RAG Weight: {result['context_strategy'].rag_weight:.0%}")
            print(f"  - Memory Weight: {result['context_strategy'].memory_weight:.0%}")
            print(f"  - Retrieval Depth: {result['context_strategy'].retrieval_depth}")
            print(f"  - Quality Score: {result['context_strategy'].last_response_quality:.2f}")

            if result.get("strategy_history"):
                print("  Evolution History:")
                for strat in result["strategy_history"]:
                    print(f"    v{strat.version}: RAG={strat.rag_weight:.0%}, Mem={strat.memory_weight:.0%}, Depth={strat.retrieval_depth}")
            print()

    # Analyze all test results
    all_results = [result1, result2, result3, result4]
    analyze_strategy_evolution(all_results)
    return


@app.cell
def _(self_evolving_system):
    from IPython.display import display, Image
    # Generate and display directly
    png_bytes = self_evolving_system.get_graph().draw_mermaid_png()
    display(Image(data=png_bytes))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
