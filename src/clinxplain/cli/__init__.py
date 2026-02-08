"""CLI entry points for ClinXplain."""

from .api import main as api_main
from .rag import main as rag_main
from .supervisor import main as supervisor_main

__all__ = ["api_main", "rag_main", "supervisor_main"]
