#!/bin/bash
set -e

echo "=== Cognee App Starting ==="

# ── Configure cognee env vars ──
# Qdrant Cloud (from .env)
export VECTOR_DB_PROVIDER="qdrant"
export VECTOR_DB_URL="${QDRANT_URL}"
export VECTOR_DB_KEY="${QDRANT_API_KEY}"

# LLM via Ollama (in sibling container)
OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"
export LLM_API_KEY="${LLM_API_KEY:-.}"
export LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
export LLM_MODEL="${LLM_MODEL:-cognee-distillabs-model-gguf-quantized}"
export LLM_ENDPOINT="${LLM_ENDPOINT:-${OLLAMA_URL}/v1}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-16384}"

# Embeddings via Ollama
export EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-ollama}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text:latest}"
export EMBEDDING_ENDPOINT="${EMBEDDING_ENDPOINT:-${OLLAMA_URL}/api/embed}"
export EMBEDDING_DIMENSIONS="${EMBEDDING_DIMENSIONS:-768}"
export HUGGINGFACE_TOKENIZER="${HUGGINGFACE_TOKENIZER:-nomic-ai/nomic-embed-text-v1.5}"

# Wait for Ollama to be reachable
echo "Waiting for Ollama at ${OLLAMA_URL}..."
until curl -sf "$OLLAMA_URL/api/tags" > /dev/null 2>&1; do
  sleep 2
done
echo "Ollama is ready."

# Import cognee data if not already done
MARKER="/app/.cognee_imported"
if [ ! -f "$MARKER" ]; then
  echo "Running initial cognee data import (setup.py)..."
  python setup.py
  touch "$MARKER"
  echo "Import complete."
else
  echo "Cognee data already imported."
fi

echo "Starting FastAPI on port 8000..."
uvicorn api:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit on port 8501..."
exec streamlit run streamlit_app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false
