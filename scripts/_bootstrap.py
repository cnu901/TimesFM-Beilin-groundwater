"""Shared bootstrap: make ``src`` importable and provide a CLI header."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def banner(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)
