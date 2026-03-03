# src/llm_local/api_parts/sse.py
"""Server-Sent Events helpers."""

from __future__ import annotations

import json
from typing import Any


def sse(event: str | None, data: Any) -> str:
    """Format a Server-Sent Event message.

    The `data:` field is JSON-encoded so clients can parse reliably.

    Parameters
    ----------
    event
        Optional SSE event name (e.g., ``"delta"``, ``"done"``).
        If ``None``, no ``event:`` line is emitted.
    data
        JSON-serializable payload.

    Returns
    -------
    str
        A correctly formatted SSE message ending with a double newline.
    """
    payload = json.dumps(data, ensure_ascii=False)
    if event:
        return f"event: {event}\ndata: {payload}\n\n"
    return f"data: {payload}\n\n"