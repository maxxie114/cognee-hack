#!/usr/bin/env python3
"""Run the ClinXplain FastAPI server (uvicorn)."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="api",
        description="Run the ClinXplain API server (chat + ingest).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("Install uvicorn: uv add uvicorn[standard]", file=sys.stderr)
        return 1

    uvicorn.run(
        "clinxplain.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
