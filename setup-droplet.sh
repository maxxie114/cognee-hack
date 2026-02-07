#!/bin/bash
# ============================================================
# DO GPU Droplet Setup Script
# ============================================================
# Run this on a fresh DO GPU droplet (Ubuntu + NVIDIA GPU).
#
# BEFORE running this script, SFTP upload these to ~/hackathon/:
#   1. models/                  (~2.7GB) - GGUF model files
#   2. cognee_export/           (~462MB) - pre-built knowledge graph
#   3. ai-memory-hackathon/     (the git repo)
#
# This script will:
#   1. Install Docker + NVIDIA Container Toolkit
#   2. Start Ollama with GPU in Docker, register the models
#   3. Start Cognee Q&A + Streamlit in Docker
#   4. Expose Ollama on port 11434 and Streamlit on port 8501
# ============================================================

set -e

WORK_DIR="${HOME}/hackathon"
MODELS_DIR="${WORK_DIR}/models"
EXPORT_DIR="${WORK_DIR}/cognee_export"
PROJECT_DIR="${WORK_DIR}/ai-memory-hackathon"

echo "============================================"
echo "  DO GPU Droplet Setup"
echo "============================================"

# ── 0. Validate uploaded files ──
echo ""
echo "[0/5] Checking prerequisites..."

if [ ! -d "$MODELS_DIR/cognee-distillabs-model-gguf-quantized" ]; then
  echo "ERROR: Missing $MODELS_DIR/cognee-distillabs-model-gguf-quantized/"
  echo "  Upload models/ to ~/hackathon/models/ via SFTP first."
  exit 1
fi
if [ ! -d "$MODELS_DIR/nomic-embed-text" ]; then
  echo "ERROR: Missing $MODELS_DIR/nomic-embed-text/"
  exit 1
fi
if [ ! -d "$EXPORT_DIR" ]; then
  echo "ERROR: Missing $EXPORT_DIR/"
  echo "  Upload cognee_export/ to ~/hackathon/cognee_export/ via SFTP first."
  exit 1
fi
if [ ! -d "$PROJECT_DIR" ]; then
  echo "ERROR: Missing $PROJECT_DIR/"
  echo "  Clone or upload the ai-memory-hackathon repo to ~/hackathon/ first."
  exit 1
fi
echo "  All files present."

# ── 1. Install Docker (if not present) ──
echo ""
echo "[1/5] Installing Docker..."
if command -v docker &>/dev/null; then
  echo "  Docker already installed: $(docker --version)"
else
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER"
  echo "  Docker installed. You may need to re-login for group changes."
fi

# ── 2. Install NVIDIA Container Toolkit (if not present) ──
echo ""
echo "[2/5] Installing NVIDIA Container Toolkit..."
if dpkg -l | grep -q nvidia-container-toolkit 2>/dev/null; then
  echo "  NVIDIA Container Toolkit already installed."
else
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update -qq
  sudo apt-get install -y -qq nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  echo "  NVIDIA Container Toolkit installed."
fi

# Verify GPU
echo "  GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  WARNING: nvidia-smi failed"

# ── 3. Start Ollama with GPU ──
echo ""
echo "[3/5] Starting Ollama..."

# Stop existing container if any
docker rm -f ollama 2>/dev/null || true

docker run -d \
  --name ollama \
  --gpus all \
  --restart unless-stopped \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  -v "${MODELS_DIR}:/models:ro" \
  ollama/ollama:latest

echo "  Waiting for Ollama to start..."
until curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; do
  sleep 2
done
echo "  Ollama is running."

# ── 4. Register models ──
echo ""
echo "[4/5] Registering models with Ollama..."

# Check if already registered
TAGS=$(curl -sf http://localhost:11434/api/tags)
if echo "$TAGS" | grep -q "nomic-embed-text"; then
  echo "  nomic-embed-text already registered."
else
  echo "  Registering nomic-embed-text..."
  curl -sf -X POST http://localhost:11434/api/create \
    -H "Content-Type: application/json" \
    -d '{"name": "nomic-embed-text", "modelfile": "FROM /models/nomic-embed-text/nomic-embed-text-v1.5.f16.gguf"}'
  echo ""
  echo "  Done."
fi

if echo "$TAGS" | grep -q "cognee-distillabs"; then
  echo "  cognee-distillabs-model-gguf-quantized already registered."
else
  echo "  Registering cognee-distillabs-model-gguf-quantized..."
  # Build modelfile with absolute container path
  MODELFILE=$(sed 's|FROM model-quantized.gguf|FROM /models/cognee-distillabs-model-gguf-quantized/model-quantized.gguf|' \
    "${MODELS_DIR}/cognee-distillabs-model-gguf-quantized/Modelfile")
  # Escape for JSON
  ESCAPED=$(printf '%s' "$MODELFILE" | sed 's/\\/\\\\/g; s/"/\\"/g' | awk '{printf "%s\\n", $0}')
  curl -sf -X POST http://localhost:11434/api/create \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"cognee-distillabs-model-gguf-quantized\", \"modelfile\": \"$ESCAPED\"}"
  echo ""
  echo "  Done."
fi

echo "  Registered models:"
curl -sf http://localhost:11434/api/tags | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    print(f\"    - {m['name']} ({m.get('size', 0) / 1e9:.1f}GB)\")
" 2>/dev/null || curl -sf http://localhost:11434/api/tags

# ── 5. Start Cognee + Streamlit ──
echo ""
echo "[5/5] Starting Cognee + Streamlit app..."

# Make sure cognee_export is in the project dir (copy if not)
if [ ! -d "${PROJECT_DIR}/cognee_export" ]; then
  echo "  Copying cognee_export into project..."
  cp -r "$EXPORT_DIR" "${PROJECT_DIR}/cognee_export"
fi

# Create .env if not exists
if [ ! -f "${PROJECT_DIR}/.env" ]; then
  cat > "${PROJECT_DIR}/.env" << 'ENVEOF'
QDRANT_URL=https://bd71c2f8-8918-44fa-8512-cda641e88d9c.us-east4-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.hFXY86eHTALmRobepNhZWkcU4Lb76aIEWhO32i2Fer4
ENVEOF
  echo "  Created .env with Qdrant Cloud credentials."
fi

# Build and run the app container
docker rm -f cognee-app 2>/dev/null || true

cd "$PROJECT_DIR"
docker build -t cognee-app -f Dockerfile.app .

# Get the host IP for Ollama (docker bridge)
OLLAMA_HOST_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ollama)

docker run -d \
  --name cognee-app \
  --restart unless-stopped \
  -p 8501:8501 \
  --env-file .env \
  -e "OLLAMA_HOST=http://${OLLAMA_HOST_IP}:11434" \
  -e "LLM_ENDPOINT=http://${OLLAMA_HOST_IP}:11434/v1" \
  -e "EMBEDDING_ENDPOINT=http://${OLLAMA_HOST_IP}:11434/api/embed" \
  -e "LLM_API_KEY=." \
  -e "LLM_PROVIDER=ollama" \
  -e "LLM_MODEL=cognee-distillabs-model-gguf-quantized" \
  -e "LLM_MAX_TOKENS=16384" \
  -e "EMBEDDING_PROVIDER=ollama" \
  -e "EMBEDDING_MODEL=nomic-embed-text:latest" \
  -e "EMBEDDING_DIMENSIONS=768" \
  -e "HUGGINGFACE_TOKENIZER=nomic-ai/nomic-embed-text-v1.5" \
  -e "VECTOR_DB_PROVIDER=qdrant" \
  -e "VECTOR_DB_URL=https://bd71c2f8-8918-44fa-8512-cda641e88d9c.us-east4-0.gcp.cloud.qdrant.io" \
  -e "VECTOR_DB_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.hFXY86eHTALmRobepNhZWkcU4Lb76aIEWhO32i2Fer4" \
  cognee-app

echo ""
echo "============================================"
echo "  DEPLOYMENT COMPLETE"
echo "============================================"
echo ""
DROPLET_IP=$(curl -sf http://169.254.169.254/metadata/v1/interfaces/public/0/ipv4/address 2>/dev/null || hostname -I | awk '{print $1}')
echo "  Ollama API:  http://${DROPLET_IP}:11434"
echo "  Streamlit:   http://${DROPLET_IP}:8501"
echo ""
echo "  Share with teammates:"
echo "  ─────────────────────"
echo "  Ollama (OpenAI-compatible):"
echo "    Base URL:  http://${DROPLET_IP}:11434/v1"
echo "    Models:    cognee-distillabs-model-gguf-quantized, nomic-embed-text"
echo ""
echo "  Embeddings:"
echo "    curl http://${DROPLET_IP}:11434/api/embed -d '{\"model\":\"nomic-embed-text\",\"input\":\"hello\"}'"
echo ""
echo "  Chat:"
echo "    curl http://${DROPLET_IP}:11434/api/chat -d '{\"model\":\"cognee-distillabs-model-gguf-quantized\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
echo ""
echo "  Streamlit Q&A UI:"
echo "    http://${DROPLET_IP}:8501"
echo ""
echo "  Logs:"
echo "    docker logs -f ollama"
echo "    docker logs -f cognee-app"
echo "============================================"
