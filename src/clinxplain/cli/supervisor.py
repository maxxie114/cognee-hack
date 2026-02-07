#!/usr/bin/env python3
"""CLI for self-evolving supervisor: run a medical query."""

from __future__ import annotations

import argparse
import asyncio
import sys

from clinxplain import RAGConfig, RAGPipeline, query_medical_system


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="supervisor",
        description="Run a medical query through the self-evolving supervisor (RAG + context evolution).",
    )
    parser.add_argument("query", type=str, help="Medical question to answer")
    parser.add_argument("--patient-id", type=str, default=None, help="Patient ID for scoped retrieval")
    parser.add_argument("--no-print-response", action="store_true", help="Do not print final response to stdout")

    args = parser.parse_args()

    async def run() -> dict:
        config = RAGConfig.from_env()
        pipeline = RAGPipeline(config)
        result = await query_medical_system(
            query=args.query,
            patient_id=args.patient_id,
            supervisor_graph=None,
            rag_pipeline=pipeline,
            rag_config=config,
        )
        return result

    result = asyncio.run(run())

    if not args.no_print_response and result.get("final_response"):
        print(result["final_response"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
