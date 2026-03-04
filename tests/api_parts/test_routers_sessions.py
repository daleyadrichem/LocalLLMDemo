# tests/api_parts/test_routers_sessions.py

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_local.api_parts.schemas import Message, Role


def _parse_sse(text: str) -> list[tuple[str | None, Any]]:
    """
    Parse a minimal subset of SSE produced by llm_local.api_parts.sse.sse().

    Returns a list of (event, data_obj).
    """
    events: list[tuple[str | None, Any]] = []
    for chunk in text.split("\n\n"):
        chunk = chunk.strip("\n")
        if not chunk.strip():
            continue

        event: str | None = None
        data_raw: str | None = None

        for line in chunk.splitlines():
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_raw = line[len("data:") :].strip()

        if data_raw is None:
            continue

        try:
            data_obj = json.loads(data_raw)
        except json.JSONDecodeError:
            data_obj = data_raw

        events.append((event, data_obj))
    return events


@dataclass
class DummyLLMWithConfig:
    """
    Sessions router uses llm.config.model when CreateSessionRequest.model is None.
    It also calls llm.chat and llm.chat_stream with model_override.
    """

    default_model: str = "default-model"
    chat_reply: str = "ASSISTANT"
    stream_chunks: list[str] = None  # type: ignore[assignment]
    raise_on_chat: bool = False
    raise_on_stream: bool = False

    # record calls
    chat_calls: list[dict[str, Any]] = None  # type: ignore[assignment]
    stream_calls: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.stream_chunks is None:
            self.stream_chunks = ["a", "b"]
        if self.chat_calls is None:
            self.chat_calls = []
        if self.stream_calls is None:
            self.stream_calls = []

    @property
    def config(self):
        # minimal config-like object with .model attribute
        class _Cfg:
            def __init__(self, model: str) -> None:
                self.model = model

        return _Cfg(self.default_model)

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        options: dict[str, Any] | None = None,
        model_override: str | None = None,
    ) -> str:
        self.chat_calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "options": options,
                "model_override": model_override,
            }
        )
        if self.raise_on_chat:
            raise RuntimeError("chat failed")
        return self.chat_reply

    def chat_stream(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        options: dict[str, Any] | None = None,
        model_override: str | None = None,
    ) -> Iterator[str]:
        self.stream_calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "options": options,
                "model_override": model_override,
            }
        )
        if self.raise_on_stream:
            raise RuntimeError("stream failed")
        for c in self.stream_chunks:
            yield c


@pytest.fixture()
def app_client_and_state(monkeypatch: pytest.MonkeyPatch):
    """
    Build a small FastAPI app with only the sessions router, and patch its module-level
    globals (SESSIONS, llm, get_session, new_session_id) to isolate state.
    """
    from llm_local.api_parts.routers import sessions as sessions_router

    dummy_llm = DummyLLMWithConfig(default_model="cfg-model")

    # Patch llm used by router
    monkeypatch.setattr(sessions_router, "llm", dummy_llm, raising=True)

    # Use an isolated sessions dict for this test module
    isolated_sessions: dict[str, Any] = {}
    monkeypatch.setattr(sessions_router, "SESSIONS", isolated_sessions, raising=True)

    # Deterministic session id + created_at for stable assertions
    monkeypatch.setattr(sessions_router, "new_session_id", lambda: "sess1", raising=True)
    monkeypatch.setattr(sessions_router.time, "time", lambda: 1000.0, raising=True)

    # get_session should read from our isolated dict and behave like real deps.get_session
    def _get_session(session_id: str):
        sess = isolated_sessions.get(session_id)
        if sess is None:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Session not found")
        return sess

    monkeypatch.setattr(sessions_router, "get_session", _get_session, raising=True)

    app = FastAPI()
    app.include_router(sessions_router.router)
    client = TestClient(app)

    return client, isolated_sessions, dummy_llm


# -----------------------------
# Create / info / history
# -----------------------------


def test_create_session_defaults_to_llm_config_model(app_client_and_state):
    client, store, _ = app_client_and_state

    r = client.post("/sessions", json={"system_prompt": None, "model": None})
    assert r.status_code == 200
    body = r.json()

    assert body["session_id"] == "sess1"
    assert body["model"] == "cfg-model"
    assert body["created_at"] == 1000.0

    assert "sess1" in store
    sess = store["sess1"]
    assert sess.session_id == "sess1"
    assert sess.model == "cfg-model"
    assert sess.system_prompt is None
    assert sess.created_at == 1000.0
    assert sess.messages == []


def test_create_session_with_system_prompt_adds_system_message(app_client_and_state):
    client, store, _ = app_client_and_state

    r = client.post("/sessions", json={"system_prompt": "You are helpful.", "model": "m1"})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "m1"

    sess = store["sess1"]
    assert sess.system_prompt == "You are helpful."
    assert len(sess.messages) == 1
    assert sess.messages[0].role == Role.system
    assert sess.messages[0].content == "You are helpful."


def test_get_session_info_ok(app_client_and_state):
    client, store, _ = app_client_and_state

    # create first
    client.post("/sessions", json={"system_prompt": "sys", "model": "m1"})
    r = client.get("/sessions/sess1")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "sess1"
    assert body["model"] == "m1"
    assert body["system_prompt"] == "sys"
    assert body["created_at"] == 1000.0
    assert body["message_count"] == 1


def test_get_session_history_ok(app_client_and_state):
    client, store, _ = app_client_and_state

    client.post("/sessions", json={"system_prompt": "sys", "model": "m1"})
    # add a message to history directly
    store["sess1"].messages.append(Message(role=Role.user, content="hi"))

    r = client.get("/sessions/sess1/history")
    assert r.status_code == 200
    body = r.json()
    assert "history" in body
    assert [m["role"] for m in body["history"]] == ["system", "user"]
    assert [m["content"] for m in body["history"]] == ["sys", "hi"]


def test_info_and_history_404_when_missing(app_client_and_state):
    client, _, _ = app_client_and_state

    r1 = client.get("/sessions/nope")
    assert r1.status_code == 404
    assert r1.json()["detail"] == "Session not found"

    r2 = client.get("/sessions/nope/history")
    assert r2.status_code == 404
    assert r2.json()["detail"] == "Session not found"


# -----------------------------
# Delete
# -----------------------------


def test_delete_session_ok(app_client_and_state):
    client, store, _ = app_client_and_state

    client.post("/sessions", json={"system_prompt": None, "model": None})
    assert "sess1" in store

    r = client.delete("/sessions/sess1")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    assert "sess1" not in store


def test_delete_session_404_when_missing(app_client_and_state):
    client, _, _ = app_client_and_state

    r = client.delete("/sessions/nope")
    assert r.status_code == 404
    assert r.json()["detail"] == "Session not found"


# -----------------------------
# Send message (non-streaming)
# -----------------------------


def test_send_session_message_appends_user_and_assistant(app_client_and_state):
    client, store, dummy = app_client_and_state

    client.post("/sessions", json={"system_prompt": "sys", "model": "m1"})

    r = client.post("/sessions/sess1/messages", json={"message": " hi "})
    assert r.status_code == 200
    assert r.json() == {"response": dummy.chat_reply}

    sess = store["sess1"]
    # system + user + assistant
    assert [m.role for m in sess.messages] == [Role.system, Role.user, Role.assistant]
    assert sess.messages[1].content == "hi"  # stripped by validator
    assert sess.messages[2].content == dummy.chat_reply

    # ensure llm.chat called with model_override session model
    assert dummy.chat_calls[-1]["model_override"] == "m1"
    # backend messages are dicts with role/content
    backend_msgs = dummy.chat_calls[-1]["messages"]
    assert backend_msgs[0]["role"] == "system"
    assert backend_msgs[1]["role"] == "user"


def test_send_session_message_500_on_backend_error(app_client_and_state):
    client, store, dummy = app_client_and_state
    dummy.raise_on_chat = True

    client.post("/sessions", json={"system_prompt": None, "model": "m1"})

    r = client.post("/sessions/sess1/messages", json={"message": "hi"})
    assert r.status_code == 500
    assert "chat failed" in r.json()["detail"]

    # user message should still have been appended before backend call
    sess = store["sess1"]
    assert len(sess.messages) == 1
    assert sess.messages[0].role == Role.user
    assert sess.messages[0].content == "hi"


# -----------------------------
# Streaming send message
# -----------------------------


def test_send_session_message_stream_appends_assistant_when_include_true(app_client_and_state):
    client, store, dummy = app_client_and_state
    dummy.stream_chunks = ["a", "", "b"]

    client.post("/sessions", json={"system_prompt": "sys", "model": "m1"})

    with client.stream(
        "POST",
        "/sessions/sess1/messages/stream?include_assistant_message=true",
        json={"message": "hi"},
    ) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())

    events = _parse_sse(body)
    assert events[0][0] == "meta"
    # only non-empty deltas are emitted by router
    assert ("delta", {"content": "a"}) in events
    assert ("delta", {"content": "b"}) in events
    assert events[-1][0] == "done"
    assert events[-1][1]["ok"] is True
    assert events[-1][1]["final_length"] == len("ab")

    sess = store["sess1"]
    # system + user + assistant(final)
    assert [m.role for m in sess.messages] == [Role.system, Role.user, Role.assistant]
    assert sess.messages[-1].content == "ab"

    # ensure llm.chat_stream called with model_override session model
    assert dummy.stream_calls[-1]["model_override"] == "m1"


def test_send_session_message_stream_does_not_append_assistant_when_include_false(app_client_and_state):
    client, store, dummy = app_client_and_state
    dummy.stream_chunks = ["a", "b"]

    client.post("/sessions", json={"system_prompt": None, "model": "m1"})

    with client.stream(
        "POST",
        "/sessions/sess1/messages/stream?include_assistant_message=false",
        json={"message": "hi"},
    ) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())

    events = _parse_sse(body)
    assert events[0][0] == "meta"
    assert events[-1][0] == "done"

    sess = store["sess1"]
    # only user is appended immediately; assistant is NOT appended
    assert [m.role for m in sess.messages] == [Role.user]


def test_send_session_message_stream_emits_error_event_on_exception(app_client_and_state):
    client, store, dummy = app_client_and_state
    dummy.raise_on_stream = True

    client.post("/sessions", json={"system_prompt": None, "model": "m1"})

    with client.stream(
        "POST",
        "/sessions/sess1/messages/stream",
        json={"message": "hi"},
    ) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())

    events = _parse_sse(body)
    assert events[0][0] == "meta"
    assert events[-1][0] == "error"
    assert events[-1][1]["ok"] is False
    assert "stream failed" in events[-1][1]["error"]

    # user message should be appended even if stream fails
    sess = store["sess1"]
    assert len(sess.messages) == 1
    assert sess.messages[0].role == Role.user