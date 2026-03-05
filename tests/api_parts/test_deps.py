# tests/api_parts/test_deps.py

import re

from fastapi import HTTPException
import pytest

from llm_local.api_parts import deps
from llm_local.api_parts.schemas import ChatSession


@pytest.fixture(autouse=True)
def _clear_sessions_between_tests():
    # Ensure global in-memory state doesn't leak across tests
    deps.SESSIONS.clear()
    yield
    deps.SESSIONS.clear()


def test_get_session_returns_existing_session():
    sess = ChatSession(
        session_id="abc123",
        created_at=123.0,
        model="llama3.2:3b",
        system_prompt=None,
        messages=[],
    )
    deps.SESSIONS["abc123"] = sess

    out = deps.get_session("abc123")
    assert out is sess


def test_get_session_raises_404_when_missing():
    with pytest.raises(HTTPException) as excinfo:
        deps.get_session("does-not-exist")

    exc = excinfo.value
    assert exc.status_code == 404
    assert exc.detail == "Session not found"


def test_new_session_id_returns_hex_uuid_and_is_unique():
    a = deps.new_session_id()
    b = deps.new_session_id()

    assert a != b
    assert re.fullmatch(r"[0-9a-f]{32}", a) is not None
    assert re.fullmatch(r"[0-9a-f]{32}", b) is not None
