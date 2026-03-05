# tests/test_ollama_http_client.py

from __future__ import annotations

from collections.abc import Iterator
import json
from typing import Any

import pytest
import requests

from llm_local.ollama_http_client import OllamaHTTPClient


class DummyResponse:
    def __init__(
        self,
        *,
        ok: bool = True,
        status_code: int = 200,
        text: str = "",
        json_data: Any = None,
        json_raises: Exception | None = None,
        iter_lines_data: list[str] | None = None,
        raise_for_status_exc: Exception | None = None,
    ):
        self.ok = ok
        self.status_code = status_code
        self.text = text
        self._json_data = json_data
        self._json_raises = json_raises
        self._iter_lines_data = iter_lines_data or []
        self._raise_for_status_exc = raise_for_status_exc

    def json(self) -> Any:
        if self._json_raises is not None:
            raise self._json_raises
        return self._json_data

    def raise_for_status(self) -> None:
        if self._raise_for_status_exc is not None:
            raise self._raise_for_status_exc
        # mimic requests behavior: raise HTTPError on non-OK if not provided
        if not self.ok:
            raise requests.HTTPError(f"{self.status_code} error")

    def iter_lines(self, decode_unicode: bool = True) -> Iterator[str]:
        for line in self._iter_lines_data:
            yield from line


class DummySession:
    def __init__(self):
        self.post_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.post_response: DummyResponse | Exception | None = None
        self.get_response: DummyResponse | Exception | None = None

    def post(self, url: str, json: dict[str, Any], timeout: int, stream: bool = False):
        self.post_calls.append({"url": url, "json": json, "timeout": timeout, "stream": stream})
        if isinstance(self.post_response, Exception):
            raise self.post_response
        assert self.post_response is not None, "DummySession.post_response must be set"
        return self.post_response

    def get(self, url: str, timeout: int):
        self.get_calls.append({"url": url, "timeout": timeout})
        if isinstance(self.get_response, Exception):
            raise self.get_response
        assert self.get_response is not None, "DummySession.get_response must be set"
        return self.get_response


def make_client(dummy_session: DummySession) -> OllamaHTTPClient:
    client = OllamaHTTPClient(base_url="http://example.com/", timeout_seconds=12)
    # replace the requests.Session() instance with our dummy
    client.session = dummy_session  # type: ignore[assignment]
    return client


# -----------------------------
# _post_json
# -----------------------------


def test_post_json_success():
    sess = DummySession()
    sess.post_response = DummyResponse(ok=True, json_data={"hello": "world"})
    client = make_client(sess)

    out = client._post_json("/api/test", {"a": 1})

    assert out == {"hello": "world"}
    assert sess.post_calls[0]["url"] == "http://example.com/api/test"
    assert sess.post_calls[0]["json"] == {"a": 1}
    assert sess.post_calls[0]["timeout"] == 12
    assert sess.post_calls[0]["stream"] is False


def test_post_json_http_error_maps_to_runtimeerror():
    sess = DummySession()
    sess.post_response = DummyResponse(
        ok=False,
        status_code=500,
        text="oops",
        raise_for_status_exc=requests.HTTPError("500 error"),
    )
    client = make_client(sess)

    with pytest.raises(RuntimeError, match=r"Failed to call local LLM backend:"):
        client._post_json("/api/test", {"a": 1})


def test_post_json_request_exception_maps_to_runtimeerror():
    sess = DummySession()
    sess.post_response = requests.Timeout("timeout")
    client = make_client(sess)

    with pytest.raises(RuntimeError, match=r"Failed to call local LLM backend:"):
        client._post_json("/api/test", {"a": 1})


def test_post_json_non_json_maps_to_runtimeerror():
    sess = DummySession()
    sess.post_response = DummyResponse(ok=True, json_raises=ValueError("no json"))
    client = make_client(sess)

    with pytest.raises(RuntimeError, match="Backend returned non-JSON response"):
        client._post_json("/api/test", {"a": 1})


# -----------------------------
# _get_json
# -----------------------------


def test_get_json_success():
    sess = DummySession()
    sess.get_response = DummyResponse(ok=True, json_data={"models": []})
    client = make_client(sess)

    out = client._get_json("/api/tags")

    assert out == {"models": []}
    assert sess.get_calls[0]["url"] == "http://example.com/api/tags"
    assert sess.get_calls[0]["timeout"] == 12


def test_get_json_http_error_maps_to_runtimeerror():
    sess = DummySession()
    sess.get_response = DummyResponse(
        ok=False,
        status_code=404,
        text="not found",
        raise_for_status_exc=requests.HTTPError("404 error"),
    )
    client = make_client(sess)

    with pytest.raises(RuntimeError, match=r"Failed to call local LLM backend:"):
        client._get_json("/api/tags")


# -----------------------------
# _post_stream
# -----------------------------


def test_post_stream_yields_only_valid_json_lines_and_skips_invalid():
    sess = DummySession()
    # includes: blank line, invalid JSON, then valid JSON objects
    sess.post_response = DummyResponse(
        ok=True,
        iter_lines_data=[
            "",
            "not-json",
            json.dumps({"response": "hi", "done": False}),
            json.dumps({"response": "!", "done": True}),
        ],
    )
    client = make_client(sess)

    chunks = list(client._post_stream("/api/generate", {"prompt": "x"}))

    assert chunks == [{"response": "hi", "done": False}, {"response": "!", "done": True}]
    assert sess.post_calls[0]["stream"] is True
    assert sess.post_calls[0]["url"] == "http://example.com/api/generate"


def test_post_stream_http_error_maps_to_runtimeerror():
    sess = DummySession()
    sess.post_response = DummyResponse(
        ok=False,
        status_code=500,
        text="server error",
        raise_for_status_exc=requests.HTTPError("500 error"),
        iter_lines_data=[],
    )
    client = make_client(sess)

    with pytest.raises(RuntimeError, match=r"Failed to stream from local LLM backend:"):
        list(client._post_stream("/api/chat", {"x": 1}))


def test_post_stream_request_exception_maps_to_runtimeerror():
    sess = DummySession()
    sess.post_response = requests.ConnectionError("boom")
    client = make_client(sess)

    with pytest.raises(RuntimeError, match=r"Failed to stream from local LLM backend:"):
        list(client._post_stream("/api/chat", {"x": 1}))


# -----------------------------
# Public endpoint wrappers
# -----------------------------


def test_endpoint_wrappers_call_expected_paths():
    sess = DummySession()
    client = make_client(sess)

    # tags -> GET /api/tags
    sess.get_response = DummyResponse(ok=True, json_data={"models": [{"name": "x"}]})
    assert client.tags() == {"models": [{"name": "x"}]}
    assert sess.get_calls[-1]["url"].endswith("/api/tags")

    # chat -> POST /api/chat
    sess.post_response = DummyResponse(ok=True, json_data={"message": {"content": "ok"}})
    assert client.chat({"model": "m"}) == {"message": {"content": "ok"}}
    assert sess.post_calls[-1]["url"].endswith("/api/chat")

    # generate -> POST /api/generate
    sess.post_response = DummyResponse(ok=True, json_data={"response": "ok"})
    assert client.generate({"model": "m"}) == {"response": "ok"}
    assert sess.post_calls[-1]["url"].endswith("/api/generate")

    # show -> POST /api/show
    sess.post_response = DummyResponse(ok=True, json_data={"details": {"a": 1}})
    assert client.show("m") == {"details": {"a": 1}}
    assert sess.post_calls[-1]["url"].endswith("/api/show")

    # delete -> POST /api/delete
    sess.post_response = DummyResponse(ok=True, json_data={"status": "ok"})
    client.delete("m")
    assert sess.post_calls[-1]["url"].endswith("/api/delete")

    # pull -> consumes stream from POST /api/pull
    sess.post_response = DummyResponse(
        ok=True,
        iter_lines_data=[json.dumps({"status": "downloading"}), json.dumps({"status": "done"})],
    )
    client.pull("m")
    assert sess.post_calls[-1]["url"].endswith("/api/pull")
    assert sess.post_calls[-1]["stream"] is True
