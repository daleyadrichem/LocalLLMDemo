from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LocalLLMConfig:
    """
    Configuration for connecting to a local LLM server (e.g. Ollama).
    """

    model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 360
    default_temperature: float = 0.2
    default_max_tokens: int | None = None
    default_options: dict[str, Any] = field(default_factory=dict)
