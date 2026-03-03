"""
Shared dependencies for the API layer.

This module contains:
- Backend configuration (environment variables)
- A singleton LocalLLM client instance
- In-memory session storage and helpers

Notes
-----
The in-memory session store is intended for local/dev usage. If you want
durability or multi-replica support, replace this with a database/redis.
"""

from __future__ import annotations

import os
import uuid

from fastapi import HTTPException

from llm_local import LocalLLM, LocalLLMConfig
from llm_local.api_parts.schemas import ChatSession

LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://ollama:11434")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2:3b")

llm = LocalLLM(
    config=LocalLLMConfig(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
    )
)

SESSIONS: dict[str, ChatSession] = {}


def get_session(session_id: str) -> ChatSession:
    """Fetch a chat session from in-memory storage.

    Parameters
    ----------
    session_id
        Unique session identifier.

    Returns
    -------
    ChatSession
        The stored session.

    Raises
    ------
    fastapi.HTTPException
        If the session does not exist (404).
    """
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess


def new_session_id() -> str:
    """Generate a new unique session identifier.

    Returns
    -------
    str
        Hex-encoded UUID.
    """
    return uuid.uuid4().hex
