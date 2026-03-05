# tests/test_llm_client.py

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from llm_local.llm_client import LocalLLM
from llm_local.llm_config import LocalLLMConfig


class FakeHTTP:
    def __init__(self) -> None:
        self.tags_result: dict[str, Any] | None = {"models": []}
        self.tags_raises: Exception | None = None

        self.generate_result: dict[str, Any] | None = {"response": " ok "}
        self.generate_raises: Exception | None = None

        self.chat_result: dict[str, Any] | None = {"message": {"content": " hi "}}
        self.chat_raises: Exception | None = None

        self.generate_stream_items: list[dict[str, Any]] = []
        self.chat_stream_items: list[dict[str, Any]] = []

        self.pulled: list[str] = []
        self.deleted: list[str] = []
        self.shown: list[str] = []

        self.generate_payloads: list[dict[str, Any]] = []
        self.chat_payloads: list[dict[str, Any]] = []

    def tags(self) -> dict[str, Any]:
        if self.tags_raises is not None:
            raise self.tags_raises
        assert self.tags_result is not None
        return self.tags_result

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.generate_payloads.append(payload)
        if self.generate_raises is not None:
            raise self.generate_raises
        assert self.generate_result is not None
        return self.generate_result

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.chat_payloads.append(payload)
        if self.chat_raises is not None:
            raise self.chat_raises
        assert self.chat_result is not None
        return self.chat_result

    def generate_stream(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        self.generate_payloads.append(payload)
        for x in self.generate_stream_items:
            yield from x

    def chat_stream(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        self.chat_payloads.append(payload)
        for x in self.chat_stream_items:
            yield from x

    def pull(self, name: str) -> None:
        self.pulled.append(name)

    def delete(self, name: str) -> None:
        self.deleted.append(name)

    def show(self, name: str) -> dict[str, Any]:
        self.shown.append(name)
        return {"name": name, "details": {"ok": True}}


@dataclass
class FakeBuilder:
    generate_calls: list[dict[str, Any]]
    chat_calls: list[dict[str, Any]]

    def generate_payload(
        self,
        model: str,
        prompt: str,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
        options: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        call = {
            "model": model,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "options": options,
            "stream": stream,
        }
        self.generate_calls.append(call)
        # return a payload shape similar to real builder, but minimal
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": stream}
        if system_prompt:
            payload["system"] = system_prompt
        if options is not None:
            payload["options"] = options
        if temperature is not None:
            payload.setdefault("options", {})
            payload["options"]["temperature"] = temperature
        if max_tokens is not None:
            payload.setdefault("options", {})
            payload["options"]["num_predict"] = max_tokens
        return payload

    def chat_payload(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        options: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        call = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "options": options,
            "stream": stream,
        }
        self.chat_calls.append(call)
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if options is not None:
            payload["options"] = options
        if temperature is not None:
            payload.setdefault("options", {})
            payload["options"]["temperature"] = temperature
        if max_tokens is not None:
            payload.setdefault("options", {})
            payload["options"]["num_predict"] = max_tokens
        return payload


def make_llm() -> tuple[LocalLLM, FakeHTTP, FakeBuilder]:
    llm = LocalLLM(config=LocalLLMConfig(model="base-model", base_url="http://x"))
    fake_http = FakeHTTP()
    fake_builder = FakeBuilder(generate_calls=[], chat_calls=[])

    # Swap internal deps after __post_init__
    llm._http = fake_http  # type: ignore[attr-defined]
    llm._builder = fake_builder  # type: ignore[attr-defined]
    return llm, fake_http, fake_builder


# -----------------------------
# Health / metadata
# -----------------------------


def test_is_backend_available_true_when_tags_ok():
    llm, fake_http, _ = make_llm()
    fake_http.tags_result = {"models": []}

    assert llm.is_backend_available() is True


def test_is_backend_available_false_when_tags_raises():
    llm, fake_http, _ = make_llm()
    fake_http.tags_raises = RuntimeError("down")

    assert llm.is_backend_available() is False


def test_list_models_extracts_names():
    llm, fake_http, _ = make_llm()
    fake_http.tags_result = {"models": [{"name": "a"}, {"name": "b"}]}

    assert llm.list_models() == ["a", "b"]


def test_list_models_bad_format_raises():
    llm, fake_http, _ = make_llm()
    # missing "name" key in model dict triggers KeyError in list comprehension
    fake_http.tags_result = {"models": [{"nope": "x"}]}

    with pytest.raises(RuntimeError, match="Unexpected response format"):
        llm.list_models()


# -----------------------------
# Model lifecycle
# -----------------------------


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_model_lifecycle_requires_name(bad: str):
    llm, _, _ = make_llm()

    with pytest.raises(RuntimeError, match="Model name is required"):
        llm.pull_model(bad)

    with pytest.raises(RuntimeError, match="Model name is required"):
        llm.delete_model(bad)

    with pytest.raises(RuntimeError, match="Model name is required"):
        llm.show_model(bad)


def test_pull_delete_show_strip_name():
    llm, fake_http, _ = make_llm()

    llm.pull_model("  llama3  ")
    llm.delete_model("  llama3  ")
    out = llm.show_model("  llama3  ")

    assert fake_http.pulled == ["llama3"]
    assert fake_http.deleted == ["llama3"]
    assert fake_http.shown == ["llama3"]
    assert out["name"] == "llama3"


# -----------------------------
# Non-streaming generation
# -----------------------------


def test_generate_uses_config_model_when_no_override_and_strips_response():
    llm, fake_http, fake_builder = make_llm()
    fake_http.generate_result = {"response": "  hello  "}

    txt = llm.generate(prompt="p", system_prompt=None)

    assert txt == "hello"
    assert fake_builder.generate_calls[-1]["model"] == "base-model"
    assert fake_builder.generate_calls[-1]["stream"] is False
    assert fake_http.generate_payloads[-1]["model"] == "base-model"


def test_generate_model_override_passed_to_builder_and_http():
    llm, fake_http, fake_builder = make_llm()
    fake_http.generate_result = {"response": "ok"}

    llm.generate(prompt="p", model_override="other-model")

    assert fake_builder.generate_calls[-1]["model"] == "other-model"
    assert fake_http.generate_payloads[-1]["model"] == "other-model"


def test_generate_missing_response_returns_empty_string():
    llm, fake_http, _ = make_llm()
    fake_http.generate_result = {"something_else": 1}

    assert llm.generate(prompt="p") == ""


# -----------------------------
# Non-streaming chat
# -----------------------------


def test_chat_parses_message_content_and_strips():
    llm, fake_http, fake_builder = make_llm()
    fake_http.chat_result = {"message": {"content": "  hi  "}}

    txt = llm.chat(messages=[{"role": "user", "content": "x"}])

    assert txt == "hi"
    assert fake_builder.chat_calls[-1]["model"] == "base-model"
    assert fake_builder.chat_calls[-1]["stream"] is False


def test_chat_bad_format_raises_runtimeerror():
    llm, fake_http, _ = make_llm()
    fake_http.chat_result = {"nope": True}

    with pytest.raises(RuntimeError, match="Unexpected response format"):
        llm.chat(messages=[{"role": "user", "content": "x"}])


def test_chat_model_override_passed_to_builder():
    llm, fake_http, fake_builder = make_llm()
    fake_http.chat_result = {"message": {"content": "ok"}}

    llm.chat(messages=[{"role": "user", "content": "x"}], model_override="other-model")

    assert fake_builder.chat_calls[-1]["model"] == "other-model"


# -----------------------------
# Streaming generation
# -----------------------------


def test_generate_stream_yields_only_non_empty_string_chunks():
    llm, fake_http, fake_builder = make_llm()
    fake_http.generate_stream_items = [
        {"response": ""},  # ignored
        {"response": "a"},
        {"response": None},  # ignored
        {"not_response": "x"},  # ignored
        {"response": "b"},
    ]

    chunks = list(llm.generate_stream(prompt="p"))

    assert chunks == ["a", "b"]
    assert fake_builder.generate_calls[-1]["stream"] is True


def test_chat_stream_yields_only_non_empty_content_from_message_dict():
    llm, fake_http, fake_builder = make_llm()
    fake_http.chat_stream_items = [
        {"message": {"content": ""}},  # ignored
        {"message": {"content": "a"}},
        {"message": {}},  # ignored
        {"message": {"content": None}},  # ignored
        {"no_message": True},  # ignored
        {"message": {"content": "b"}},
    ]

    chunks = list(llm.chat_stream(messages=[{"role": "user", "content": "x"}]))

    assert chunks == ["a", "b"]
    assert fake_builder.chat_calls[-1]["stream"] is True
