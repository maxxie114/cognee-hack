FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# ── Install Python dependencies ──
COPY pyproject.toml ./
RUN uv pip install --system \
    "cognee==0.4.1" \
    "transformers" \
    "qdrant-client" \
    "cognee-community-vector-adapter-qdrant" \
    "python-dotenv" \
    "requests>=2.31" \
    "boto3>=1.35" \
    "streamlit>=1.30" \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.27"

# ── Copy project code ──
COPY custom_retriever.py custom_generate_completion.py ./
COPY solution_q_and_a.py ./
COPY streamlit_app.py ./
COPY api.py ./
COPY helper_functions/ ./helper_functions/
COPY prompts/ ./prompts/
COPY shared/ ./shared/

# ── Copy cognee_export data (graph DB, system DB, data files) ──
# This contains the pre-built knowledge graph (Kuzu), LanceDB vectors, SQLite metadata,
# and 2000 text files. Must be manually placed here (it's gitignored, ~462MB).
COPY cognee_export/ ./cognee_export/

# ── Setup script: imports cognee_export into cognee's internal paths ──
COPY setup.py ./
COPY scripts/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 8501 8000

ENTRYPOINT ["/docker-entrypoint.sh"]
