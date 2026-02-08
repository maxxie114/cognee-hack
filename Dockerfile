# ClinXplain API — run with: docker build -t clinxplain . && docker run -p 8000:8000 clinxplain
# Requires Qdrant (e.g. docker run -p 6333:6333 qdrant/qdrant) and OPENAI_API_KEY at runtime.

FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock* ./

# Install dependencies (no dev); use --no-install-project so we don't need src yet
RUN uv sync --frozen --no-install-project --no-dev 2>/dev/null || uv sync --no-install-project --no-dev

# Copy package and app
COPY src ./src

# Install the project itself
RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

# Default: run the API on port 8000
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000

EXPOSE 8000

CMD ["uv", "run", "api", "--host", "0.0.0.0", "--port", "8000"]
