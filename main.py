#!/usr/bin/env python3
"""Entry point for ClinXplain (optional). Use `rag` or `supervisor` CLI, or import clinxplain."""

from clinxplain import __version__


def main() -> None:
    print(f"ClinXplain {__version__}")
    print("CLI: rag (ingest | query | retrieve | eval), supervisor (query)")


if __name__ == "__main__":
    main()
