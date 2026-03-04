# tests/api_parts/test_schemas.py

import pytest
from pydantic import ValidationError

from llm_local.api_parts.schemas import (
    ChatRequest,
    CreateSessionRequest,
    GenerateRequest,
    Message,
    Role,
    SessionMessageRequest,
)


def test_message_strips_whitespace():
    m = Message(role=Role.user, content="  hello  ")
    assert m.content == "hello"


def test_message_rejects_empty_after_strip():
    with pytest.raises(ValidationError):
        Message(role=Role.user, content="   ")


def test_message_to_backend():
    m = Message(role=Role.assistant, content="hi")
    assert m.to_backend() == {"role": "assistant", "content": "hi"}


def test_session_message_request_strips_whitespace():
    req = SessionMessageRequest(message="  ping  ")
    assert req.message == "ping"


def test_session_message_request_rejects_empty_after_strip():
    with pytest.raises(ValidationError):
        SessionMessageRequest(message=" \n\t ")


def test_generate_request_requires_non_empty_prompt():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="")

    # ok case
    req = GenerateRequest(prompt="hello", temperature=0.2, max_tokens=10, options={"top_p": 0.9})
    assert req.prompt == "hello"
    assert req.temperature == 0.2
    assert req.max_tokens == 10
    assert req.options == {"top_p": 0.9}


def test_generate_request_temperature_range_validation():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="x", temperature=-0.01)

    with pytest.raises(ValidationError):
        GenerateRequest(prompt="x", temperature=2.01)

    # boundaries are allowed
    assert GenerateRequest(prompt="x", temperature=0.0).temperature == 0.0
    assert GenerateRequest(prompt="x", temperature=2.0).temperature == 2.0


def test_generate_request_max_tokens_validation():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="x", max_tokens=0)

    assert GenerateRequest(prompt="x", max_tokens=1).max_tokens == 1


def test_chat_request_builds_with_messages():
    req = ChatRequest(messages=[Message(role=Role.user, content="hi")], temperature=0.3)
    assert len(req.messages) == 1
    assert req.messages[0].role == Role.user
    assert req.messages[0].content == "hi"
    assert req.temperature == 0.3


def test_create_session_request_allows_optional_fields():
    req = CreateSessionRequest(system_prompt=None, model=None)
    assert req.system_prompt is None
    assert req.model is None

    req2 = CreateSessionRequest(system_prompt="  system  ", model="llama3.2:3b")
    # note: CreateSessionRequest does not strip; only Message/SessionMessageRequest do
    assert req2.system_prompt == "  system  "
    assert req2.model == "llama3.2:3b"