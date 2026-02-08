"""One-time Cognee ingestion from existing data_storage (used by rag cognee-ingest CLI)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


def _project_root() -> Path:
    root = Path(__file__).resolve().parent.parent.parent
    if (root / "src" / "clinxplain").exists():
        return root
    return root.parent if root.parent else root


async def _run_ingest_async(data_path: Path, dataset_name: str) -> None:
    # Import runs registration (use_vector_adapter / use_dataset_database_handler); do not call register()
    import cognee_community_vector_adapter_qdrant.register  # noqa: F401
    import cognee

    path_str = str(data_path.resolve())
    print(f"Adding data from {path_str} to dataset '{dataset_name}'...")
    await cognee.add(path_str, dataset_name=dataset_name)
    print("Cognifying (chunk, embed, graph)...")
    await cognee.cognify(datasets=[dataset_name])
    print("Done. Set RAG_BACKEND=cognee to query via Cognee Search.")


def run_cognee_ingest(path: str | None = None) -> int:
    """
    Run Cognee add + cognify on the given path (or default data_storage).
    Returns 0 on success, non-zero on failure.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(_project_root() / ".env")
    except ImportError:
        pass

    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY.", file=sys.stderr)
        return 1

    root = _project_root()
    if path:
        data_path = Path(path).expanduser().resolve()
    else:
        data_path = root / "data_storage"

    if not data_path.exists() or not data_path.is_dir():
        print(f"Directory not found: {data_path}", file=sys.stderr)
        return 1

    dataset_name = os.getenv("COGNEE_DATASET_NAME", "clin_xplain_data")

    try:
        asyncio.run(_run_ingest_async(data_path, dataset_name))
    except ImportError as e:
        print("Install Cognee: uv pip install -e '.[cognee]'", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0
