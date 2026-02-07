#!/bin/bash
# prepare-deploy.sh — Validates the project is ready for Docker deployment
# Run this ONCE before `docker compose -f docker-compose.deploy.yml up --build`
#
# Prerequisites (manually upload these — they're gitignored):
#   1. models/                    (~2.7GB) — GGUF model files for Ollama
#   2. cognee_export/             (~462MB) — pre-built knowledge graph + data
#   3. .env                       — Qdrant Cloud credentials
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Validating deployment readiness ==="
ERRORS=0

# 1. Check cognee_export
if [ -L "cognee_export" ]; then
  echo "WARNING: cognee_export/ is a symlink. Resolving to real directory..."
  TARGET=$(readlink -f cognee_export)
  rm cognee_export
  cp -r "$TARGET" cognee_export
  echo "  Copied from $TARGET → cognee_export/"
elif [ -d "cognee_export" ]; then
  echo "OK: cognee_export/ exists"
else
  echo "MISSING: cognee_export/ — upload it (contains system_databases/ + data_storage/)"
  ERRORS=$((ERRORS + 1))
fi

# 2. Check models
if [ -d "models/cognee-distillabs-model-gguf-quantized" ] && [ -d "models/nomic-embed-text" ]; then
  echo "OK: models/ directory found"
else
  echo "MISSING: models/ — upload it with cognee-distillabs-model-gguf-quantized/ and nomic-embed-text/"
  ERRORS=$((ERRORS + 1))
fi

# 3. Check .env
if [ -f ".env" ]; then
  if grep -q "your-qdrant-api-key\|your-cluster-id" .env; then
    echo "WARNING: .env has placeholder values — edit with real Qdrant Cloud credentials"
  else
    echo "OK: .env exists"
  fi
else
  if [ -f ".env.example.deploy" ]; then
    cp .env.example.deploy .env
    echo "CREATED: .env from template — EDIT IT with your Qdrant Cloud credentials!"
  else
    echo "MISSING: .env — create it with QDRANT_URL and QDRANT_API_KEY"
    ERRORS=$((ERRORS + 1))
  fi
fi

# 4. Validate required code files
REQUIRED_FILES=(
  "Dockerfile.app"
  "docker-compose.deploy.yml"
  "scripts/register-models.sh"
  "scripts/docker-entrypoint.sh"
  "setup.py"
  "streamlit_app.py"
  "custom_retriever.py"
  "custom_generate_completion.py"
  "pyproject.toml"
  "prompts/system_prompt.txt"
  "prompts/user_prompt.txt"
)

for f in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "MISSING: $f"
    ERRORS=$((ERRORS + 1))
  fi
done

if [ "$ERRORS" -gt 0 ]; then
  echo ""
  echo "=== $ERRORS issue(s) found. Fix them before deploying. ==="
  exit 1
fi

echo ""
echo "=== All checks passed! ==="
echo ""
echo "Deploy with:"
echo "  docker compose -f docker-compose.deploy.yml up --build -d"
echo ""
echo "Streamlit UI will be at http://<host>:8501"
