from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LocalLLMConfig:
    """Configuration for connecting to a local LLM server.

    Attributes
    ----------
    model : str
        Model identifier exposed by the local LLM backend.
    base_url : str
        Base URL for the local LLM HTTP API.
    timeout_seconds : int
        Request timeout in seconds.
    default_temperature : float
        Default sampling temperature used for generation requests.
    default_max_tokens : int | None
        Default maximum number of generated tokens. If ``None``, the backend
        default is used.
    default_options : dict[str, Any]
        Additional backend-specific generation options merged into requests.
    """

    model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 360
    default_temperature: float = 0.2
    default_max_tokens: int | None = None
    default_options: dict[str, Any] = field(default_factory=dict)
