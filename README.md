# ClinXplain

**RAG pipeline, agentic RAG, and self-evolving supervisor for medical Q&A.** Uses **Qdrant** as the only vector store (no Redis). Supports **document-scoped chat** (select a file to chat with, no cross-document data leak) and **replace-on-ingest** (each ingest replaces the index). Built with **LangChain**, **LangGraph**, and **OpenAI**; optional **Cognee** backend for retrieval.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [System Architecture Diagram](#system-architecture-diagram)
- [Agentic Workflow Diagram](#agentic-workflow-diagram)
- [RAG Pipeline Diagram](#rag-pipeline-diagram)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Supervisor Graph Flow](#supervisor-graph-flow)
- [RAG Pipeline Flow](#rag-pipeline-flow)
- [Package Structure](#package-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [RAG Evaluation](#rag-evaluation)

---

## Architecture Overview

ClinXplain is a multi-layer medical Q&A system built on three pillars:

1. **RAG Pipeline** — Document ingestion (replace collection on each ingest), vector retrieval (Qdrant, optional Cognee), semantic caching, and answer generation (LangGraph). Retrieval can be **document-scoped** via `source_filter` so only the selected file is searched.
2. **Agentic RAG** — A ReAct agent with retrieval, analysis, and query-reformulation tools; used by the supervisor to answer from synthesized context.
3. **Self-Evolving Supervisor** — A LangGraph orchestrator that synthesizes context (RAG + conversation memory), delegates to the agentic RAG sub-agent, and evolves strategy (RAG vs. memory weights) based on response quality.

**Storage:** Only **Qdrant** is used: collection `rag_docs` for RAG chunks (with `source` and `document_name` in payload), and optional short-term memory collection. No Redis.

**Access:**
- **CLI** — `rag` (ingest/query/retrieve/eval, optional `cognee-ingest`) and `supervisor` commands
- **REST/WebSocket API** — FastAPI: GET `/documents`, POST `/chat`, POST `/ingest`, WebSocket `/ws/chat`; all support optional `document_source` for scoped retrieval
- **Streamlit UI** — Document selector, upload & ingest, chat with selected document or all documents
- **Python API** — Direct imports for programmatic use

---

## System Architecture Diagram

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        CLI["CLI (rag, supervisor, api)"]
        REST["REST API (GET /documents, POST /chat, POST /ingest)"]
        WS["WebSocket (/ws/chat)"]
        Streamlit["Streamlit UI\n(Document selector + Chat)"]
    end

    subgraph API["API Layer (FastAPI)"]
        DocsHandler["GET /documents"]
        ChatHandler["POST /chat Handler"]
        IngestHandler["POST /ingest Handler"]
        WSHandler["/ws/chat Handler"]
    end

    subgraph Core["Core Orchestration"]
        Supervisor["Self-Evolving Supervisor\n(LangGraph)"]
        AgenticRAG["Agentic RAG\n(ReAct Agent)"]
        RAGPipeline["RAG Pipeline"]
    end

    subgraph RAG["RAG Module"]
        Ingestion["Ingestion\n(Replace collection → Chunk → Embed)"]
        Retrieval["Retrieval\n(Qdrant / Cognee, source_filter)"]
        Graph["RAG Graph\n(Cache → Retrieve → Generate)"]
    end

    subgraph Storage["Storage Layer"]
        Qdrant["Qdrant\n(rag_docs + short-term memory)"]
    end

    subgraph External["External Services"]
        OpenAI["OpenAI\n(Embeddings + LLM)"]
    end

    CLI --> Supervisor
    CLI --> RAGPipeline
    Streamlit --> WS
    REST --> ChatHandler
    REST --> IngestHandler
    REST --> DocsHandler
    WS --> WSHandler

    ChatHandler --> Supervisor
    WSHandler --> RAGPipeline
    IngestHandler --> RAGPipeline
    DocsHandler --> RAGPipeline

    Supervisor --> AgenticRAG
    Supervisor --> RAGPipeline
    AgenticRAG --> RAGPipeline

    RAGPipeline --> Ingestion
    RAGPipeline --> Retrieval
    RAGPipeline --> Graph

    Ingestion --> Qdrant
    Retrieval --> Qdrant
    Graph --> Retrieval

    Ingestion --> OpenAI
    Retrieval --> OpenAI
    Graph --> OpenAI
    AgenticRAG --> OpenAI
    Supervisor --> OpenAI
```

---

## Agentic Workflow Diagram

End-to-end flow when using the **self-evolving supervisor** (e.g. POST `/chat`): context is synthesized from RAG (optionally scoped to one document) and conversation history, then the ReAct RAG agent answers; the supervisor may evolve strategy and loop until quality is sufficient.

```mermaid
flowchart TB
    subgraph Input["Input"]
        Q["User query"]
        PID["patient_id (optional)"]
        DS["document_source (optional)"]
        CH["conversation_history"]
    end

    subgraph Supervisor["Supervisor (LangGraph)"]
        S1["synthesize_context\n(RAG + memory, source_filter)"]
        S2["delegate_to_rag\n(ReAct agent with context)"]
        S3["check_continuation"]
        S4["evolve_strategy"]
        S5["finalize_response"]
    end

    subgraph RAGAgent["Agentic RAG"]
        T1["retrieve_documents"]
        T2["analyze_retrieved_context"]
        T3["reformulate_query"]
        T4["LLM synthesize"]
    end

    subgraph Store["Qdrant"]
        V["rag_docs\n(source, document_name)"]
    end

    Q --> S1
    PID --> S1
    DS --> S1
    CH --> S1
    S1 --> S2
    S2 --> T1
    S2 --> T2
    S2 --> T3
    T1 --> V
    T1 --> T4
    T2 --> T4
    T3 --> T4
    T4 --> S3
    S3 -->|continue| S4
    S4 --> S1
    S3 -->|finalize| S5
    S5 --> OUT["final_response\n+ conversation_history"]
```

---

## RAG Pipeline Diagram

RAG path used by **WebSocket /ws/chat** (streaming) and by the supervisor’s retrieval: ingest replaces the collection; retrieve supports optional `source_filter` so only one document is searched.

```mermaid
flowchart LR
    subgraph Ingest["Ingest (replace-on)"]
        I1["Upload / path"]
        I2["Delete collection\n(if exists)"]
        I3["Load → Chunk → Embed"]
        I4["Upsert to Qdrant\n(payload: source, document_name)"]
    end

    subgraph Retrieve["Retrieve (document-scoped)"]
        R1["Query + optional source_filter"]
        R2["Qdrant filter\nsource = source_filter"]
        R3["Top-k chunks"]
        R4["Context string"]
    end

    subgraph Generate["Generate"]
        G1["Context + conversation_history"]
        G2["LLM stream (WebSocket)\nor single reply"]
    end

    I1 --> I2 --> I3 --> I4
    I4 --> Q[(Qdrant\nrag_docs)]
    Q --> R1 --> R2 --> R3 --> R4 --> G1 --> G2
```

---

## Component Architecture

```mermaid
flowchart LR
    subgraph src_clinxplain["src/clinxplain"]
        api["api.py\nFastAPI app"]
        agentic["agentic.py\nReAct Agent"]
        supervisor["supervisor.py\nLangGraph Supervisor"]

        subgraph rag["rag/"]
            config["config.py\nRAGConfig"]
            ingestion["ingestion.py\nReplace collection, Chunk, Embed"]
            retrieval["retrieval.py\nQdrant/Cognee, source_filter"]
            graph["graph.py\nRAG LangGraph"]
            pipeline["pipeline.py\nRAGPipeline"]
            weave_eval["weave_eval.py\nWeave Eval"]
        end

        subgraph cli["cli/"]
            rag_cli["rag.py\ningest|query|retrieve|eval"]
            supervisor_cli["supervisor.py\n<query>"]
            api_cli["api.py\nuvicorn server"]
        end
    end

    api --> supervisor
    api --> pipeline
    supervisor --> agentic
    supervisor --> pipeline
    agentic --> retrieval
    pipeline --> ingestion
    pipeline --> retrieval
    pipeline --> graph
```

---

## Data Flow

### Document Ingestion Flow (replace-on-ingest)

Each ingest **replaces** the Qdrant collection so previous documents are not retained. Chunk payloads include `source` and `document_name` for document-scoped retrieval.

```mermaid
flowchart LR
    A[Documents\nPDF / TXT / MD] --> B[Load (PyPDF / TextLoader)]
    B --> C[RecursiveCharacterTextSplitter\nChunk 1000, Overlap 200]
    C --> D[OpenAI Embeddings\ntext-embedding-3-small]
    D --> E[Delete existing collection]
    E --> F[Upsert to Qdrant\npayload: source, document_name]
    F --> G[(rag_docs)]
```

### Query Flow (RAG with optional document scope)

When `document_source` (or `source_filter`) is set, only chunks from that document are returned (no cross-document leak).

```mermaid
flowchart LR
    Q[User Question\n+ optional source_filter] --> C1{Check Cache?}
    C1 -->|Yes| CacheHit["Return Cached Answer"]
    C1 -->|No| R[Embed Query]
    R --> S[Vector Search Qdrant\ntop_k, filter source if set]
    S --> T[Retrieved Chunks]
    T --> U[Build Context]
    U --> V[LLM Generate\n+ conversation_history]
    V --> A1[Answer]
    V --> W[Save to Cache]
    A1 --> OUT[Answer]
```

### Supervisor Query Flow

```mermaid
flowchart TB
    Q[User Query + patient_id / document_source] --> S1[Synthesize Context]
    S1 --> S2[Delegate to RAG Agent]
    S2 --> S3[Check Continuation]
    S3 -->|quality < threshold & iter < max| E[Evolve Strategy]
    S3 -->|done| F[Finalize Response]
    E --> S1
    F --> OUT[Final Response]
    F --> ShortMem[(Short-Term Memory\nQdrant)]

    subgraph Evolution["Strategy Evolution"]
        I1["Iter 1: 70% RAG, 30% memory"]
        I2["Iter 2: 85% RAG, 15% memory (if quality < 0.7)"]
        I3["Iter 3+: 90% RAG, 10% memory, lower threshold"]
    end
```

---

## Supervisor Graph Flow

The self-evolving supervisor uses a LangGraph `StateGraph` with conditional edges:

```mermaid
stateDiagram-v2
    [*] --> synthesize_context
    synthesize_context --> delegate_to_rag
    delegate_to_rag --> check_continuation

    check_continuation --> evolve_strategy: should_continue
    check_continuation --> finalize_response: finalize

    evolve_strategy --> synthesize_context: loop
    finalize_response --> [*]

    note right of synthesize_context
        ContextManager combines:
        - RAG retrieval (patient_id or document_source;
          source_filter for document scope)
        - Conversation memory (in-memory + optional Qdrant)
        Strategy weights: rag_weight, memory_weight
    end note

    note right of delegate_to_rag
        Agentic RAG (ReAct agent)
        Tools: retrieve_documents,
        analyze_retrieved_context,
        reformulate_query
    end note

    note right of evolve_strategy
        ContextStrategy evolves:
        - Increase rag_weight if quality < 0.7
        - Increase retrieval_depth
        - Lower similarity_threshold
    end note
```

### Supervisor State (`AgentState`)

| Field | Type | Description |
|-------|------|-------------|
| `query` | str | User question |
| `patient_id` | str \| None | Optional patient scope for retrieval |
| `document_source` | str \| None | Optional document path; when set, retrieval is scoped to this file only (no cross-document leak) |
| `rag_results` | list[RetrievedDocument] | Documents from RAG |
| `conversation_history` | list[ConversationTurn] | Prior turns (included in context for “chat history” queries) |
| `context_strategy` | ContextStrategy | RAG vs. memory weights, retrieval depth |
| `combined_context` | str | Synthesized context for RAG agent |
| `rag_response` | str | Response from agentic RAG |
| `sources_used` | list[str] | Sources cited |
| `current_iteration` | int | Iteration counter |
| `should_continue` | bool | Whether to evolve or finalize |
| `final_response` | str | Final output |
| `strategy_history` | list[ContextStrategy] | Evolution history |

---

## RAG Pipeline Flow

The RAG module uses a LangGraph `StateGraph` with caching:

```mermaid
flowchart TB
    START --> CheckCache
    CheckCache -->|Cache Hit| END
    CheckCache -->|Cache Miss| retrieve

    retrieve --> generate
    generate --> WriteCache
    WriteCache --> END

    subgraph retrieve
        R1[Extract Question]
        R2[Vector Search Qdrant]
        R3[Attach context to state]
    end

    subgraph generate
        G1[Build context string]
        G2[ChatOpenAI + Prompt]
        G3[Return messages]
    end

    subgraph cache
        CheckCache["Semantic Cache Lookup\n(Distance < 0.80)"]
        WriteCache["Save Q&A to Cache"]
    end
```

### RAG State (`RAGState`)

| Field | Type | Description |
|-------|------|-------------|
| `messages` | list[BaseMessage] | Chat history (add_messages reducer) |
| `context` | list[Document] | Retrieved documents |
| `question` | str | Current question |

---

## Agentic RAG Tools

The ReAct agent exposes three tools (used by the supervisor when delegating to RAG):

```mermaid
flowchart TB
    subgraph Tools["Agentic RAG Tools"]
        T1["retrieve_documents\n(query, top_k, filter)"]
        T2["analyze_retrieved_context\n(context)"]
        T3["reformulate_query\n(original_query, conversation_context)"]
    end

    T1 --> Qdrant["Qdrant Vector Store\n(rag_docs)"]
    T2 --> RuleBased["Rule-based extraction\n(keywords: medication, diagnosis, etc.)"]
    T3 --> RuleBased2["Rule-based reformulation\n(medical terms)"]
```

| Tool | Purpose |
|------|---------|
| `retrieve_documents` | Vector search over Qdrant (RAG pipeline); returns formatted chunk excerpts |
| `analyze_retrieved_context` | Extracts key facts (medications, diagnoses, labs) from context |
| `reformulate_query` | Reformulates vague queries with medical terms and context |

---

## Package Structure

```
src/clinxplain/
├── __init__.py          # RAGConfig, RAGPipeline, create_rag_agent, create_supervisor, query_medical_system
├── __main__.py          # Entry: python -m clinxplain → runs API
├── api.py               # FastAPI: /health, GET /documents, POST /chat, POST /ingest, WS /ws/chat
├── agentic.py           # ReAct agent with retrieve/analyze/reformulate tools
├── supervisor.py        # Self-evolving supervisor (LangGraph)
├── cli/
│   ├── rag.py           # rag ingest | query | retrieve | eval
│   ├── supervisor.py    # supervisor <query> [--patient-id ID]
│   └── api.py           # api [--host 0.0.0.0] [--port 8000]
└── rag/
    ├── config.py        # RAGConfig (Qdrant, embedding, chunking, retrieval)
    ├── ingestion.py     # Replace collection, load, chunk, embed, index (Qdrant)
    ├── retrieval.py     # Vector search (Qdrant or Cognee), source_filter
    ├── graph.py         # LangGraph RAG: retrieve → generate
    ├── pipeline.py      # RAGPipeline high-level API
    ├── weave_eval.py    # Weave tracing + evaluation
    └── schema.yaml      # RAG index schema reference
```

---

## Quick Start

### 1. Environment

```bash
cp .env.example .env
# Edit .env: QDRANT_URL, OPENAI_API_KEY
```

### 2. Install

```bash
uv sync
```

### 3. CLI

```bash
# RAG: ingest, query, retrieve, or evaluate
uv run rag ingest docs/ [--glob "**/*.pdf"]
uv run rag query "What medications is this patient taking?" [--top-k 5]
uv run rag retrieve "search query" [--top-k 5]
uv run rag eval [--project your-team/rag-eval --parallelism 3]

# Supervisor: run medical query through self-evolving system
uv run supervisor "What medications is this patient taking?" [--patient-id patient_001]

# API server (see "How to run the API" below)
uv run api [--host 0.0.0.0] [--port 8000] [--reload]

# Chatbot UI (Streamlit): upload docs + chat via WebSocket
uv run streamlit run streamlit_app.py [--server.port 8501]
```

#### How to run the API

1. **Prerequisites:** Copy `.env.example` to `.env`, set `QDRANT_URL` and `OPENAI_API_KEY`, then run `uv sync`. Start Qdrant (e.g. `docker run -p 6333:6333 qdrant/qdrant`).
2. **Start the server:**
   ```bash
   uv run api
   ```
   Default: listens on `http://0.0.0.0:8000` (all interfaces, port 8000).
3. **Optional flags:**
   - `--host <host>` — Bind address (default: `0.0.0.0`).
   - `--port <port>` — Port (default: `8000`).
   - `--reload` — Auto-reload on code changes (development only).
   ```bash
   uv run api --host 127.0.0.1 --port 8000 --reload
   ```
4. **Verify:** Open `http://localhost:8000/health` — should return `{"status":"ok"}`. Interactive API docs: `http://localhost:8000/docs` (Swagger UI).

### 4. Python API

```python
from clinxplain import RAGConfig, RAGPipeline, create_rag_agent, create_supervisor, query_medical_system

config = RAGConfig.from_env()
pipeline = RAGPipeline(config)
pipeline.ingest("docs/")
docs = pipeline.retrieve("search query", top_k=5)
answer = pipeline.query("Your question?")

# Agentic RAG (standalone or used by supervisor)
rag_agent = create_rag_agent(config)

# Self-evolving supervisor (optional document_source for scoped retrieval)
result = await query_medical_system(
    "Medical question?",
    patient_id="patient_001",
    document_source="/path/to/patient_doc.pdf",  # optional: scope to one document
)
print(result["final_response"])
```

---

## Configuration

| Variable | Description |
|----------|-------------|
| `QDRANT_URL` | Qdrant connection URL (default: `http://localhost:6333`) |
| `OPENAI_API_KEY` | OpenAI API key (embeddings + LLM) |
| `QDRANT_COLLECTION` | Qdrant collection name (default: `rag_docs`) |
| `RAG_TOP_K` | Chunks to retrieve (default: 5) |
| `RAG_CACHE_DISTANCE` | Cache distance threshold (default: 0.80) |
| `RAG_BACKEND` | `qdrant` (default) or `cognee` for retrieval backend |
| `WEAVE_PROJECT` | Weave project for eval (e.g. `your-username/rag-eval`) |

See `.env.example` for full list.

---

## API Reference

Base URL when running locally: `http://localhost:8000`. Interactive docs: **http://localhost:8000/docs**.

| Method / Type | Path | Description |
|---------------|------|-------------|
| GET | `/health` | Health check; returns `{"status":"ok"}` |
| GET | `/documents` | List indexed documents for selector; returns `[{ "source", "document_name" }]` |
| POST | `/chat` | Chat (supervisor + RAG); JSON: `message`, optional `patient_id`, `document_source`, `conversation_history` |
| POST | `/ingest` | Ingest documents (replace index); multipart form `files` (PDF, .txt, .md) |
| WebSocket | `/ws/chat` | Streaming chat; RAG retrieve (optional `document_source`) + token-by-token LLM stream |

### REST Endpoints

#### GET `/health`

Health check for load balancers and readiness probes.

- **Response:** `200 OK`
- **Body:** `{ "status": "ok" }`

---

#### GET `/documents`

List documents currently in the RAG index (for UI document selector). Each ingest replaces the index, so this reflects the latest ingest only.

- **Response:** `200 OK`, JSON array of `{ "source": "<path>", "document_name": "<name>" }`.

---

#### POST `/chat`

Chat with the medical Q&A system (self-evolving supervisor + RAG). Returns the assistant reply and updated conversation history.

- **Request:** JSON body
  - `message` (string, required) — User question.
  - `patient_id` (string, optional) — Patient ID for scoped retrieval.
  - `document_source` (string, optional) — When set, retrieval is scoped to this document only (no cross-document leak).
  - `conversation_history` (array, optional) — Previous turns: `[{ "query": "...", "response": "..." }]`.
- **Response:** `200 OK`, JSON
  - `response` (string) — Assistant reply.
  - `conversation_history` (array) — Updated list of `{ "query", "response" }`.
  - `iterations` (number) — Supervisor iteration count.
  - `strategy_version` (number) — Context strategy version.
- **Errors:** `500` — Server error (e.g. missing env, Qdrant/LLM failure).

**Example:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What medications is this patient on?", "document_source": "/path/to/patient_notes.pdf", "conversation_history": []}'
```

---

#### POST `/ingest`

Ingest documents into the RAG index (Qdrant). **Replaces** the existing collection on each run. Accepts multipart form-data with one or more files.

- **Request:** `multipart/form-data`
  - `files` (required) — One or more files; each part named `files`. Allowed types: `.pdf`, `.txt`, `.md`.
- **Response:** `200 OK`, JSON
  - `chunks_ingested` (number) — Number of chunks indexed.
  - `files_received` (number) — Number of files received.
- **Errors:**
  - `400` — No files or no valid file types.
  - `500` — Server error (e.g. ingest failure).

**Example:**

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@document1.pdf" \
  -F "files=@notes.txt"
```

### WebSocket `/ws/chat`

Real-time chat with **token-by-token LLM streaming** over a single WebSocket connection.

| Aspect | Details |
|--------|---------|
| **Path** | `ws://<host>:<port>/ws/chat` (e.g. `ws://localhost:8000/ws/chat`) |
| **Flow** | RAG retrieve (same config as pipeline) → build context → stream LLM tokens over the socket |
| **Note** | Uses a simplified RAG + streaming LLM path (not the full self-evolving supervisor) so responses can stream token-by-token. For full supervisor logic (strategy evolution, multi-iteration), use REST `POST /chat`. |

**Send (JSON, first message after connect):**

```json
{
  "message": "What medications is this patient on?",
  "patient_id": "patient_001",
  "document_source": "/path/to/patient_notes.pdf",
  "conversation_history": [
    { "query": "Previous question", "response": "Previous answer" }
  ]
}
```

- `message` (required): User question. Aliased as `query` if `message` is missing.
- `patient_id` (optional): Scopes retrieval (e.g. "Patient patient_001: ...").
- `document_source` (optional): When set, only chunks from this document are retrieved (document-scoped chat, no data leak).
- `conversation_history` (optional): Array of `{ "query", "response" }` for multi-turn context and “chat history” answers; included in the prompt.

**Receive (JSON messages):**

| `type` | Payload | When |
|--------|---------|------|
| `token` | `{ "type": "token", "content": "<one token or chunk>" }` | Each streamed LLM token/chunk |
| `done` | `{ "type": "done", "response": "<full text>", "conversation_history": [...] }` | After streaming finishes; includes full response and updated history |
| `error` | `{ "type": "error", "detail": "<message>" }` | On validation or server error (e.g. missing `message`, invalid JSON) |

**Minimal client example (browser):**

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/chat");
ws.onopen = () => {
  ws.send(JSON.stringify({
    message: "What are the main findings?",
    patient_id: null,
    conversation_history: []
  }));
};
ws.onmessage = (e) => {
  const data = JSON.parse(e.data);
  if (data.type === "token") process.stdout.write(data.content);
  if (data.type === "done") console.log("\nFull response:", data.response);
  if (data.type === "error") console.error("Error:", data.detail);
};
```

Document ingestion remains REST only: use `POST /ingest` for uploading files.

---

## RAG Evaluation (Weave)

- **Tracing:** Retrieval wrapped with `weave.op()` for observability.
- **RAGModel:** Weave `Model` with `predict(question)` → `{answer, context}`.
- **LLM Judge:** Context precision (RAGAS-style) scorer.
- **Optional RAGAS:** Faithfulness, answer relevancy (install `clinxplain[ragas]`).

```bash
uv run rag eval [--project your-team/rag-eval --parallelism 3]
```

```python
from clinxplain.rag.weave_eval import init_weave, RAGModel, run_evaluation

run_evaluation(project="your-team/rag-eval", parallelism=3)
```

---

## Qdrant MCP

If using Qdrant MCP in Cursor, the RAG pipeline uses the same Qdrant instance via `QDRANT_URL`.

---

## Document Paths

- **`docs/`** — Default folder for CLI ingest (created if missing).
- **`sample_patients/`** — Sample patient documents for testing document-scoped chat.
- **Streamlit** — Documents are uploaded via the UI; each ingest replaces the index. Use **Chat with document** to scope answers to one file.

---

## Dependencies

- **Core:** qdrant-client, langchain, langgraph, langchain-openai, pypdf, python-dotenv
- **API:** fastapi, uvicorn, python-multipart
- **Eval:** weave
- **Optional:** ragas (for RAGAS metrics)
