"""Pydantic schemas and in-memory session types for the API."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Role(str, Enum):
    """Allowed chat message roles."""

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class Message(BaseModel):
    """A single chat message.

    Parameters
    ----------
    role
        Message role (e.g. ``"user"``).
    content
        Message text content. Leading/trailing whitespace is stripped.

    Notes
    -----
    This model is used both for validation (incoming requests) and for
    storing session histories. It serializes to the same JSON shape as
    a typical chat message dict: ``{"role": "...", "content": "..."}``.
    """

    role: Role
    content: str = Field(..., min_length=1)

    @field_validator("content")
    @classmethod
    def _strip_and_validate_content(cls, v: str) -> str:
        """Strip whitespace and ensure non-empty content."""
        v = v.strip()
        if not v:
            raise ValueError("content must not be empty")
        return v

    def to_backend(self) -> dict[str, str]:
        """Convert to the backend message dictionary.

        Returns
        -------
        dict[str, str]
            Message formatted as ``{"role": "<role>", "content": "<content>"}``.
        """
        return {"role": self.role.value, "content": self.content}


@dataclass
class ChatSession:
    """In-memory chat session.

    Parameters
    ----------
    session_id
        Unique session identifier.
    created_at
        Unix timestamp (seconds since epoch).
    model
        Model name used for this session.
    system_prompt
        Optional system prompt for session creation.
    messages
        Ordered list of messages in the conversation.
    """

    session_id: str
    created_at: float
    model: str
    system_prompt: str | None
    messages: list[Message]


class HealthResponse(BaseModel):
    status: str


class ModelsResponse(BaseModel):
    models: list[str]


class ModelPullRequest(BaseModel):
    name: str = Field(..., description="Model name to pull, e.g. llama3.2:3b")


class ModelPullResponse(BaseModel):
    status: str


class ModelDeleteResponse(BaseModel):
    status: str


class ModelShowResponse(BaseModel):
    model: str
    details: dict[str, Any]


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    system_prompt: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    options: dict[str, Any] | None = None


class GenerateResponse(BaseModel):
    response: str


class ChatRequest(BaseModel):
    messages: list[Message]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    options: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    response: str


class CreateSessionRequest(BaseModel):
    system_prompt: str | None = None
    model: str | None = None


class CreateSessionResponse(BaseModel):
    session_id: str
    model: str
    created_at: float


class SessionInfoResponse(BaseModel):
    session_id: str
    model: str
    system_prompt: str | None
    created_at: float
    message_count: int


class ChatHistoryResponse(BaseModel):
    history: list[Message]


class SessionMessageRequest(BaseModel):
    message: str = Field(..., min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    options: dict[str, Any] | None = None

    @field_validator("message")
    @classmethod
    def _strip_and_validate_message(cls, v: str) -> str:
        """Strip whitespace and ensure non-empty message."""
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        return v


class SessionMessageResponse(BaseModel):
    response: str


class DeleteSessionResponse(BaseModel):
    status: str
