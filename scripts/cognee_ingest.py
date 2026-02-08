#!/usr/bin/env python3
"""
One-time Cognee ingestion from existing data_storage directory.

Uses Cognee Add (local path) + Cognify; vectors are written to Qdrant when
VECTOR_DB_PROVIDER=qdrant. Does not replace POST /ingest or `rag ingest`.

Usage:
  python scripts/cognee_ingest.py [path_to_data_storage]
  # or: uv run python scripts/cognee_ingest.py ./data_storage
  # or: rag cognee-ingest [path]

Requires: pip install -e ".[cognee]"
Env: OPENAI_API_KEY, VECTOR_DB_PROVIDER=qdrant, VECTOR_DB_URL, etc. (see .env.example)
"""

from __future__ import annotations

import sys

# Ensure project root is on path when run as script
if __name__ == "__main__":
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from clinxplain.rag.cognee_ingest import run_cognee_ingest

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(run_cognee_ingest(path=path))
