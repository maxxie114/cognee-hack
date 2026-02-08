"""Allow running the API with: python -m clinxplain."""

from __future__ import annotations

from .cli.api import main

if __name__ == "__main__":
    raise SystemExit(main())
