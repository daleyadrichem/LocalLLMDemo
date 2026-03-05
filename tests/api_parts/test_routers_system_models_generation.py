# tests/api_parts/test_routers_system_models_generation.py

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import json
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest


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
class DummyLLM:
    # Health
    backend_available: bool = True

    # Models
    models: list[str] = None  # type: ignore[assignment]
    show_details: dict[str, Any] = None  # type: ignore[assignment]
    raise_on_list: bool = False
    raise_on_pull: bool = False
    raise_on_delete: bool = False
    raise_on_show: bool = False

    pulled: list[str] = None  # type: ignore[assignment]
    deleted: list[str] = None  # type: ignore[assignment]
    shown: list[str] = None  # type: ignore[assignment]

    # Generation (non-stream)
    generate_text: str = "GEN"
    chat_text: str = "CHAT"
    raise_on_generate: bool = False
    raise_on_chat: bool = False

    # Streaming
    gen_stream_chunks: list[str] = None  # type: ignore[assignment]
    chat_stream_chunks: list[str] = None  # type: ignore[assignment]
    raise_on_generate_stream: bool = False
    raise_on_chat_stream: bool = False

    def __post_init__(self) -> None:
        if self.models is None:
            self.models = ["llama3.2:3b"]
        if self.show_details is None:
            self.show_details = {"ok": True}
        if self.pulled is None:
            self.pulled = []
        if self.deleted is None:
            self.deleted = []
        if self.shown is None:
            self.shown = []
        if self.gen_stream_chunks is None:
            self.gen_stream_chunks = ["a", "", "b"]
        if self.chat_stream_chunks is None:
            self.chat_stream_chunks = ["x", "", "y"]

    # -----------------
    # Health
    # -----------------
    def is_backend_available(self) -> bool:
        return self.backend_available

    # -----------------
    # Models
    # -----------------
    def list_models(self) -> list[str]:
        if self.raise_on_list:
            raise RuntimeError("list failed")
        return list(self.models)

    def pull_model(self, name: str) -> None:
        if self.raise_on_pull:
            raise RuntimeError("pull failed")
        self.pulled.append(name)

    def delete_model(self, name: str) -> None:
        if self.raise_on_delete:
            raise RuntimeError("delete failed")
        self.deleted.append(name)

    def show_model(self, name: str) -> dict[str, Any]:
        if self.raise_on_show:
            raise RuntimeError("show failed")
        self.shown.append(name)
        return dict(self.show_details)

    # -----------------
    # Generation
    # -----------------
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        if self.raise_on_generate:
            raise RuntimeError("generate failed")
        return self.generate_text

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        if self.raise_on_chat:
            raise RuntimeError("chat failed")
        return self.chat_text

    # -----------------
    # Streaming
    # -----------------
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        if self.raise_on_generate_stream:
            raise RuntimeError("generate stream failed")
        for c in self.gen_stream_chunks:
            yield from c

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        if self.raise_on_chat_stream:
            raise RuntimeError("chat stream failed")
        for c in self.chat_stream_chunks:
            yield from c


@pytest.fixture()
def app_and_client(monkeypatch: pytest.MonkeyPatch):
    # Import routers (they each bind a module-level `llm` from deps at import time)
    from llm_local.api_parts.routers import generation, models, system

    dummy = DummyLLM()
    # Patch module-level llm symbols used by router handlers
    monkeypatch.setattr(system, "llm", dummy, raising=True)
    monkeypatch.setattr(models, "llm", dummy, raising=True)
    monkeypatch.setattr(generation, "llm", dummy, raising=True)

    app = FastAPI()
    app.include_router(system.router)
    app.include_router(models.router)
    app.include_router(generation.router)

    client = TestClient(app)
    return app, client, dummy


# -----------------------------
# /health (system router)
# -----------------------------


def test_health_ok(app_and_client):
    _, client, dummy = app_and_client
    dummy.backend_available = True

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_health_503_when_backend_unavailable(app_and_client):
    _, client, dummy = app_and_client
    dummy.backend_available = False

    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["detail"] == "LLM backend not available"


# -----------------------------
# /models (models router)
# -----------------------------


def test_models_list_ok(app_and_client):
    _, client, dummy = app_and_client
    dummy.models = ["a", "b"]

    r = client.get("/models")
    assert r.status_code == 200
    assert r.json() == {"models": ["a", "b"]}


def test_models_list_500_on_runtimeerror(app_and_client):
    _, client, dummy = app_and_client
    dummy.raise_on_list = True

    r = client.get("/models")
    assert r.status_code == 500
    assert "list failed" in r.json()["detail"]


def test_models_pull_ok(app_and_client):
    _, client, dummy = app_and_client

    r = client.post("/models/pull", json={"name": "llama3.2:3b"})
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    assert dummy.pulled == ["llama3.2:3b"]


def test_models_pull_500_on_runtimeerror(app_and_client):
    _, client, dummy = app_and_client
    dummy.raise_on_pull = True

    r = client.post("/models/pull", json={"name": "llama3.2:3b"})
    assert r.status_code == 500
    assert "pull failed" in r.json()["detail"]


def test_models_delete_ok(app_and_client):
    _, client, dummy = app_and_client

    r = client.delete("/models/llama3.2:3b")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    assert dummy.deleted == ["llama3.2:3b"]


def test_models_delete_500_on_runtimeerror(app_and_client):
    _, client, dummy = app_and_client
    dummy.raise_on_delete = True

    r = client.delete("/models/llama3.2:3b")
    assert r.status_code == 500
    assert "delete failed" in r.json()["detail"]


def test_models_show_ok(app_and_client):
    _, client, dummy = app_and_client
    dummy.show_details = {"family": "llama"}

    r = client.get("/models/llama3.2:3b")
    assert r.status_code == 200
    assert r.json() == {"model": "llama3.2:3b", "details": {"family": "llama"}}
    assert dummy.shown == ["llama3.2:3b"]


def test_models_show_500_on_runtimeerror(app_and_client):
    _, client, dummy = app_and_client
    dummy.raise_on_show = True

    r = client.get("/models/llama3.2:3b")
    assert r.status_code == 500
    assert "show failed" in r.json()["detail"]


# -----------------------------
# /generate, /chat (generation router)
# -----------------------------


def test_generate_ok(app_and_client):
    _, client, dummy = app_and_client
    dummy.generate_text = "hello"

    r = client.post("/generate", json={"prompt": "hi"})
    assert r.status_code == 200
    assert r.json() == {"response": "hello"}


def test_generate_500_on_runtimeerror(app_and_client):
    _, client, dummy = app_and_client
    dummy.raise_on_generate = True

    r = client.post("/generate", json={"prompt": "hi"})
    assert r.status_code == 500
    assert "generate failed" in r.json()["detail"]


def test_chat_ok(app_and_client):
    _, client, dummy = app_and_client
    dummy.chat_text = "there"

    r = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json() == {"response": "there"}


def test_chat_500_on_runtimeerror(app_and_client):
    _, client, dummy = app_and_client
    dummy.raise_on_chat = True

    r = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 500
    assert "chat failed" in r.json()["detail"]


# -----------------------------
# Streaming SSE endpoints
# -----------------------------


def test_generate_stream_emits_meta_deltas_done(app_and_client):
    _, client, dummy = app_and_client
    dummy.gen_stream_chunks = ["a", "", "b"]

    with client.stream("POST", "/generate/stream", json={"prompt": "hi"}) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())

    events = _parse_sse(body)
    # meta, delta(a), delta(b), done
    assert events[0][0] == "meta"
    assert events[1] == ("delta", {"content": "a"})
    assert events[2] == ("delta", {"content": "b"})
    assert events[-1][0] == "done"
    assert events[-1][1]["ok"] is True


def test_generate_stream_emits_error_event_on_exception(app_and_client):
    _, client, dummy = app_and_client
    dummy.raise_on_generate_stream = True

    with client.stream("POST", "/generate/stream", json={"prompt": "hi"}) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())

    events = _parse_sse(body)
    assert events[0][0] == "meta"
    assert events[-1][0] == "error"
    assert events[-1][1]["ok"] is False
    assert "generate stream failed" in events[-1][1]["error"]


def test_chat_stream_emits_meta_deltas_done(app_and_client):
    _, client, dummy = app_and_client
    dummy.chat_stream_chunks = ["x", "", "y"]

    payload = {"messages": [{"role": "user", "content": "hi"}]}
    with client.stream("POST", "/chat/stream", json=payload) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())

    events = _parse_sse(body)
    assert events[0][0] == "meta"
    assert events[1] == ("delta", {"content": "x"})
    assert events[2] == ("delta", {"content": "y"})
    assert events[-1][0] == "done"
    assert events[-1][1]["ok"] is True


def test_chat_stream_emits_error_event_on_exception(app_and_client):
    _, client, dummy = app_and_client
    dummy.raise_on_chat_stream = True

    payload = {"messages": [{"role": "user", "content": "hi"}]}
    with client.stream("POST", "/chat/stream", json=payload) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())

    events = _parse_sse(body)
    assert events[0][0] == "meta"
    assert events[-1][0] == "error"
    assert events[-1][1]["ok"] is False
    assert "chat stream failed" in events[-1][1]["error"]
