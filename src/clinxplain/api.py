"""FastAPI app: WebSocket chat (streaming), REST ingest, optional REST chat."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .rag import RAGConfig, RAGPipeline
from .rag.ingestion import list_documents
from .supervisor import ConversationTurn, create_supervisor, query_medical_system

# -----------------------------------------------------------------------------
# App and shared state (lazy init)
# -----------------------------------------------------------------------------

app = FastAPI(
    title="ClinXplain API",
    description="Chat (supervisor) and document ingestion for medical Q&A",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router with /api prefix so both POST /ingest and POST /api/ingest work
api_router = APIRouter(prefix="/api", tags=["api"])

_pipeline: RAGPipeline | None = None
_supervisor_graph: Any = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        config = RAGConfig.from_env()
        _pipeline = RAGPipeline(config)
    return _pipeline


def _get_supervisor():
    global _supervisor_graph
    if _supervisor_graph is None:
        pipeline = _get_pipeline()
        config = RAGConfig.from_env()
        _supervisor_graph = create_supervisor(pipeline, rag_config=config, short_term_memory=None)
    return _supervisor_graph


# -----------------------------------------------------------------------------
# Request / response models
# -----------------------------------------------------------------------------


class ChatTurn(BaseModel):
    """A single turn in conversation history (for multi-turn chat)."""

    query: str
    response: str


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    message: str = Field(..., description="User message / question")
    patient_id: str | None = Field(None, description="Optional patient ID for scoped retrieval")
    document_source: str | None = Field(
        None,
        description="Optional document source path to scope retrieval (no cross-document leak)",
    )
    conversation_history: list[ChatTurn] | None = Field(
        None,
        description="Previous turns for multi-turn context",
    )


class ChatResponse(BaseModel):
    """Response for POST /chat."""

    response: str = Field(..., description="Assistant reply (final_response text)")
    conversation_history: list[ChatTurn] = Field(default_factory=list)
    iterations: int = 0
    strategy_version: int = 0


class IngestResponse(BaseModel):
    """Response for POST /ingest."""

    chunks_ingested: int = Field(..., description="Number of chunks indexed")
    points_count: int = Field(..., description="Number of Qdrant points written")
    files_received: int = Field(..., description="Number of files uploaded")


class DocumentListItem(BaseModel):
    """One document in the RAG index (for document selector)."""

    source: str = Field(..., description="Unique source path (filter key)")
    document_name: str = Field(..., description="Display name (e.g. filename)")


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@app.get("/")
async def root() -> JSONResponse:
    """API info and available endpoints (avoids 404 when hitting base URL)."""
    return JSONResponse({
        "name": "ClinXplain API",
        "docs": "/docs",
        "health": "/health",
        "documents": "GET /documents (list for selector)",
        "ingest": "POST /ingest (multipart form: files)",
        "chat": "POST /chat",
        "ws_chat": "GET /ws/chat",
    })


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check for load balancers / readiness."""
    return {"status": "ok"}


@app.get("/documents", response_model=list[DocumentListItem])
async def get_documents() -> list[DocumentListItem]:
    """List documents in the RAG index (unique source + document_name) for the document selector."""
    config = RAGConfig.from_env()
    items = list_documents(config=config)
    return [DocumentListItem(source=it["source"], document_name=it["document_name"]) for it in items]


def _extract_final_response(final_response: str) -> str:
    """Strip markdown header from supervisor final_response for plain reply."""
    if not final_response or not final_response.startswith("## MEDICAL Q&A RESPONSE"):
        return final_response
    lines = final_response.split("\n")
    in_body = False
    body_lines = []
    for line in lines:
        if line.strip() == "**Response:**":
            in_body = True
            continue
        if in_body and line.strip() == "---":
            break
        if in_body:
            body_lines.append(line)
    return "\n".join(body_lines).strip() if body_lines else final_response


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """
    WebSocket chat with streaming LLM tokens.

    Send JSON: { "message": "...", "patient_id": "optional", "conversation_history": [{ "query", "response" }] }.
    Receives: { "type": "token", "content": "..." } for each token, then { "type": "done", "response", "conversation_history" }.
    Uses RAG retrieve + streaming LLM (same config as pipeline).
    """
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        data = json.loads(raw)
        message = data.get("message") or data.get("query", "")
        if not message:
            await websocket.send_json({"type": "error", "detail": "message is required"})
            return
        patient_id = data.get("patient_id")
        document_source = data.get("document_source")
        conversation_history = data.get("conversation_history") or []

        config = RAGConfig.from_env()
        pipeline = _get_pipeline()
        effective_query = f"{message}" if not patient_id else f"Patient {patient_id}: {message}"
        docs = pipeline.retrieve(
            effective_query,
            top_k=config.top_k,
            source_filter=document_source,
        )
        context_str = "\n\n---\n\n".join(
            (d.page_content or "").strip() for d in docs
        ) if docs else "No relevant documents found."
        system = (
            "Answer based on the following context and the conversation history when provided. "
            "If the user asks for chat history or what was discussed, use the conversation history below. "
            "If the context does not contain enough information, say so. Do not make up information."
        )
        history_block = ""
        if conversation_history:
            history_block = "Conversation history (this session):\n" + "\n".join(
                f"User: {t.get('query', '')}\nAssistant: {t.get('response', '')}"
                for t in conversation_history
            ) + "\n\n"
        user_content = f"{history_block}Context:\n{context_str}\n\nQuestion: {message}\n\nAnswer:"
        llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            streaming=True,
        )
        messages = [SystemMessage(content=system), HumanMessage(content=user_content)]
        streamed_content: list[str] = []
        async for chunk in llm.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                content = chunk.content if isinstance(chunk.content, str) else ""
                if content:
                    streamed_content.append(content)
                    await websocket.send_json({"type": "token", "content": content})

        full_response = "".join(streamed_content)
        new_history = list(conversation_history)
        new_history.append({"query": message, "response": full_response})

        await websocket.send_json({
            "type": "done",
            "response": full_response,
            "conversation_history": new_history,
        })
    except WebSocketDisconnect:
        pass
    except json.JSONDecodeError as e:
        try:
            await websocket.send_json({"type": "error", "detail": f"Invalid JSON: {e}"})
        except Exception:
            pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat with the medical Q&A system (self-evolving supervisor + RAG).

    Send a message and optional patient_id / conversation_history.
    Returns the assistant response and updated conversation history.
    """
    try:
        history: list[ConversationTurn] | None = None
        if request.conversation_history:
            history = [
                ConversationTurn(turn_number=i + 1, query=t.query, response=t.response)
                for i, t in enumerate(request.conversation_history)
            ]

        supervisor = _get_supervisor()
        result = await query_medical_system(
            query=request.message,
            patient_id=request.patient_id,
            document_source=request.document_source,
            conversation_history=history,
            supervisor_graph=supervisor,
        )

        final_response = result.get("final_response", "")
        # Strip markdown header if frontend prefers plain text; optional
        if final_response.startswith("## MEDICAL Q&A RESPONSE"):
            lines = final_response.split("\n")
            in_body = False
            body_lines = []
            for line in lines:
                if line.strip() == "**Response:**":
                    in_body = True
                    continue
                if in_body and line.strip() == "---":
                    break
                if in_body:
                    body_lines.append(line)
            if body_lines:
                final_response = "\n".join(body_lines).strip()

        raw_history = result.get("conversation_history", [])
        new_history = [
            ChatTurn(
                query=getattr(t, "query", t.get("query", "")),
                response=getattr(t, "response", t.get("response", "")),
            )
            for t in raw_history
        ]

        return ChatResponse(
            response=final_response,
            conversation_history=new_history,
            iterations=result.get("current_iteration", 0),
            strategy_version=result.get("context_strategy") and getattr(
                result["context_strategy"], "version", 0
            ) or 0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _ingest_impl(
    files: list[UploadFile],
) -> IngestResponse:
    """Shared ingest logic for /ingest and /api/ingest."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    allowed_suffixes = {".pdf", ".txt", ".md"}
    tmpdir = tempfile.mkdtemp(prefix="clinxplain_ingest_")
    try:
        written = 0
        for uf in files:
            fn = (uf.filename or "upload").lstrip("/")
            path = Path(tmpdir) / fn
            suffix = path.suffix.lower()
            if suffix not in allowed_suffixes:
                continue
            content = await uf.read()
            path.write_bytes(content)
            written += 1

        if written == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No valid files (allowed: {', '.join(allowed_suffixes)})",
            )

        pipeline = _get_pipeline()
        keys = pipeline.ingest(tmpdir, glob="*", loader_type="auto")
        return IngestResponse(
            chunks_ingested=len(keys),
            points_count=len(keys),
            files_received=written,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    files: list[UploadFile] = File(..., description="PDF or text files to ingest"),
) -> IngestResponse:
    """
    Ingest uploaded documents into the RAG index (Qdrant).

    Accepts multipart/form-data with one or more files (PDF, .txt, .md).
    Saves to a temp directory, runs the pipeline ingest, then cleans up.
    """
    return await _ingest_impl(files)


@api_router.post("/ingest", response_model=IngestResponse)
async def ingest_documents_api(
    files: list[UploadFile] = File(..., description="PDF or text files to ingest"),
) -> IngestResponse:
    """Same as POST /ingest; use when base URL includes /api."""
    return await _ingest_impl(files)


@api_router.get("/documents", response_model=list[DocumentListItem])
async def get_documents_api() -> list[DocumentListItem]:
    """Same as GET /documents; list documents for the selector."""
    return await get_documents()


app.include_router(api_router)


# -----------------------------------------------------------------------------
# Run with: uvicorn clinxplain.api:app --reload
# -----------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Return the FastAPI app (for testing or custom ASGI server)."""
    return app
