"""Streaming partial JSON parser for tool call arguments."""

from __future__ import annotations

import json
from typing import Any


def parse_streaming_json(partial: str) -> dict[str, Any]:
    """Parse potentially incomplete JSON from streaming tool call args.

    Uses three-tier fallback:
    1. Standard json.loads() for complete JSON
    2. partial_json.loads() for incomplete JSON
    3. Empty dict fallback
    """
    if not partial or not partial.strip():
        return {}
    try:
        result = json.loads(partial)
        if isinstance(result, dict):
            return result
        return {}
    except json.JSONDecodeError:
        pass
    try:
        from partial_json_parser import loads as partial_loads

        result = partial_loads(partial)
        if isinstance(result, dict):
            return result
        return {}
    except Exception:
        return {}
