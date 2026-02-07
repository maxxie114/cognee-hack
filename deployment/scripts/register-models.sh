#!/bin/sh
# register-models.sh — waits for Ollama, then creates models from GGUF files
set -e

OLLAMA_URL="http://ollama:11434"

echo "Waiting for Ollama to be ready..."
until curl -sf "$OLLAMA_URL/api/tags" > /dev/null 2>&1; do
  sleep 2
done
echo "Ollama is ready."

# Check if models already registered (look for our model names)
TAGS=$(curl -sf "$OLLAMA_URL/api/tags")
if echo "$TAGS" | grep -q "nomic-embed-text" && echo "$TAGS" | grep -q "cognee-distillabs"; then
  echo "Models already registered. Skipping."
  exit 0
fi

echo "Registering nomic-embed-text..."
curl -sf -X POST "$OLLAMA_URL/api/create" \
  -H "Content-Type: application/json" \
  -d '{"name": "nomic-embed-text", "modelfile": "FROM /models/nomic-embed-text/nomic-embed-text-v1.5.f16.gguf"}'
echo ""
echo "nomic-embed-text registered."

echo "Registering cognee-distillabs-model-gguf-quantized..."
# Build the modelfile with absolute path — read template and replace FROM line
# Since we can't rely on python3 in this minimal image, use sed
MODELFILE=$(sed 's|FROM model-quantized.gguf|FROM /models/cognee-distillabs-model-gguf-quantized/model-quantized.gguf|' /models/cognee-distillabs-model-gguf-quantized/Modelfile)

# Escape for JSON: replace newlines with \n, escape quotes and backslashes
ESCAPED=$(printf '%s' "$MODELFILE" | sed 's/\\/\\\\/g; s/"/\\"/g' | awk '{printf "%s\\n", $0}')

curl -sf -X POST "$OLLAMA_URL/api/create" \
  -H "Content-Type: application/json" \
  -d "{\"name\": \"cognee-distillabs-model-gguf-quantized\", \"modelfile\": \"$ESCAPED\"}"
echo ""
echo "cognee-distillabs-model-gguf-quantized registered."

echo ""
echo "All models registered:"
curl -sf "$OLLAMA_URL/api/tags"
