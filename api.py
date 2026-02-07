"""
Cognee Q&A FastAPI Server
=========================
POST /query  → {"question": "..."} → {"answer": "...", "status": "ok"}
GET  /health → {"status": "ok", "models": {...}}
"""

import os
import asyncio
import pathlib
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load env & configure BEFORE any cognee imports
load_dotenv()
os.environ["VECTOR_DB_PROVIDER"] = "qdrant"
os.environ.setdefault("VECTOR_DB_URL", os.getenv("QDRANT_URL", ""))
os.environ.setdefault("VECTOR_DB_KEY", os.getenv("QDRANT_API_KEY", ""))

_ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

os.environ.setdefault("LLM_API_KEY", ".")
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("LLM_MODEL", "cognee-distillabs-model-gguf-quantized")
os.environ.setdefault("LLM_ENDPOINT", f"{_ollama_host}/v1")
os.environ.setdefault("LLM_MAX_TOKENS", "16384")

os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
os.environ.setdefault("EMBEDDING_MODEL", "nomic-embed-text:latest")
os.environ.setdefault("EMBEDDING_ENDPOINT", f"{_ollama_host}/api/embed")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "768")
os.environ.setdefault("HUGGINGFACE_TOKENIZER", "nomic-ai/nomic-embed-text-v1.5")

# Register Qdrant adapter BEFORE importing cognee
import cognee_community_vector_adapter_qdrant.register  # noqa: F401

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from custom_retriever import GraphCompletionRetrieverWithUserPrompt

# Global retriever instance
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        system_prompt_path = str(
            pathlib.Path(
                os.path.join(pathlib.Path(__file__).parent, "prompts/system_prompt.txt")
            ).resolve()
        )
        _retriever = GraphCompletionRetrieverWithUserPrompt(
            user_prompt_filename="user_prompt.txt",
            system_prompt_path=system_prompt_path,
            top_k=10,
        )
    return _retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up retriever on startup
    _get_retriever()
    yield


app = FastAPI(
    title="Cognee Q&A API",
    description="Query the cognee knowledge graph via REST API",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins so teammates can call from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 10


class QueryResponse(BaseModel):
    answer: str
    status: str = "ok"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm_model": os.environ.get("LLM_MODEL"),
        "embedding_model": os.environ.get("EMBEDDING_MODEL"),
        "vector_db": os.environ.get("VECTOR_DB_PROVIDER"),
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    try:
        retriever = _get_retriever()
        result = await retriever.get_completion(query=req.question)
        answer = result[0] if isinstance(result, (list, tuple)) else str(result)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
