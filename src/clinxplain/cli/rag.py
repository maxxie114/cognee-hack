#!/usr/bin/env python3
"""CLI for RAG pipeline: ingest, query, retrieve, eval."""

from __future__ import annotations

import argparse
import os
import sys

from clinxplain.rag import RAGConfig, RAGPipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="RAG pipeline: ingest documents and query.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest", help="Ingest documents from a path")
    ingest_parser.add_argument("path", type=str, help="Directory or file path")
    ingest_parser.add_argument("--glob", default="**/*.pdf", help="Glob for directory (default: **/*.pdf)")
    ingest_parser.add_argument("--loader", default="auto", choices=["auto", "pdf", "text", "markdown"])

    query_parser = sub.add_parser("query", help="Run a RAG query (retrieve + generate)")
    query_parser.add_argument("question", type=str, help="Question to answer")
    query_parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")

    retrieve_parser = sub.add_parser("retrieve", help="Retrieve chunks only (no LLM)")
    retrieve_parser.add_argument("question", type=str, help="Query for retrieval")
    retrieve_parser.add_argument("--top-k", type=int, default=None)
    retrieve_parser.add_argument("--no-print", action="store_true", help="Do not print chunk text")

    eval_parser = sub.add_parser("eval", help="Run RAG evaluation with Weave (tracing + LLM judge)")
    eval_parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Weave project (e.g. your-username/rag-eval). Default: WEAVE_PROJECT env",
    )
    eval_parser.add_argument(
        "--parallelism",
        type=int,
        default=None,
        help="Max parallel eval workers (e.g. 3 to avoid rate limits)",
    )

    cognee_ingest_parser = sub.add_parser(
        "cognee-ingest",
        help="One-time Cognee ingest from data_storage (requires pip install -e '.[cognee]')",
    )
    cognee_ingest_parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="Path to data_storage (default: ./data_storage)",
    )

    args = parser.parse_args()
    config = RAGConfig.from_env()
    pipeline = RAGPipeline(config)

    if args.command == "ingest":
        keys = pipeline.ingest(args.path, glob=args.glob, loader_type=args.loader)
        print(f"Ingested {len(keys)} chunks.")
        return 0

    if args.command == "query":
        if not os.getenv("OPENAI_API_KEY"):
            print("Set OPENAI_API_KEY for generation.", file=sys.stderr)
            return 1
        if args.top_k is not None:
            config.top_k = args.top_k
        answer = pipeline.query(args.question)
        print(answer)
        return 0

    if args.command == "retrieve":
        kwargs = {} if args.top_k is None else {"top_k": args.top_k}
        docs = pipeline.retrieve(args.question, **kwargs)
        print(f"Retrieved {len(docs)} chunks.")
        if not getattr(args, "no_print", False):
            for i, d in enumerate(docs):
                print(f"\n--- Chunk {i + 1} (source: {d.metadata.get('source', '')}) ---")
                print(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))
        return 0

    if args.command == "eval":
        try:
            from clinxplain.rag.weave_eval import run_evaluation
        except ImportError as e:
            print("Weave evaluation requires weave.", file=sys.stderr)
            print("Install with: uv add weave", file=sys.stderr)
            raise SystemExit(1) from e
        if run_evaluation is None:
            print("Weave is not installed. Install with: uv add weave", file=sys.stderr)
            return 1
        run_evaluation(
            project=args.project,
            parallelism=args.parallelism,
        )
        print("Evaluation complete. Check your Weave project for traces and scores.")
        return 0

    if args.command == "cognee-ingest":
        try:
            from clinxplain.rag.cognee_ingest import run_cognee_ingest
        except ImportError as e:
            print("Cognee ingest requires cognee. Install with: uv pip install -e '.[cognee]'", file=sys.stderr)
            raise SystemExit(1) from e
        return run_cognee_ingest(path=args.path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
