from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
import logging
from typing import Any

from llm_local.llm_config import LocalLLMConfig
from llm_local.ollama_http_client import OllamaHTTPClient
from llm_local.ollama_request_builder import OllamaRequestBuilder

logger = logging.getLogger(__name__)


@dataclass
class LocalLLM:
    """
    High-level client for interacting with a local LLM server (Ollama).
    Adds:
    - Streaming generators (generate_stream, chat_stream)
    - Model lifecycle (pull_model, delete_model, show_model)
    """

    config: LocalLLMConfig = field(default_factory=LocalLLMConfig)
    _http: OllamaHTTPClient = field(init=False, repr=False)
    _builder: OllamaRequestBuilder = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._http = OllamaHTTPClient(
            base_url=self.config.base_url,
            timeout_seconds=self.config.timeout_seconds,
        )
        self._builder = OllamaRequestBuilder(self.config)
        logger.debug(
            "Initialized LocalLLM model=%s base_url=%s", self.config.model, self.config.base_url
        )

    # ------------------------------------------------------------------
    # Health / metadata
    # ------------------------------------------------------------------

    def is_backend_available(self) -> bool:
        try:
            self._http.tags()
            return True
        except RuntimeError as exc:
            logger.warning("LLM backend not available: %s", exc)
            return False

    def list_models(self) -> list[str]:
        data = self._http.tags()
        try:
            return [m["name"] for m in data.get("models", [])]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response format: {data}") from exc

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def pull_model(self, name: str) -> None:
        if not name or not name.strip():
            raise RuntimeError("Model name is required")
        self._http.pull(name.strip())

    def delete_model(self, name: str) -> None:
        if not name or not name.strip():
            raise RuntimeError("Model name is required")
        self._http.delete(name.strip())

    def show_model(self, name: str) -> dict[str, Any]:
        if not name or not name.strip():
            raise RuntimeError("Model name is required")
        return self._http.show(name.strip())

    # ------------------------------------------------------------------
    # Non-streaming generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        options: dict[str, Any] | None = None,
        model_override: str | None = None,
    ) -> str:
        model = model_override or self.config.model
        payload = self._builder.generate_payload(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            options=options,
            stream=False,
        )
        data = self._http.generate(payload)

        # Ollama /api/generate typically returns { "response": "...", ... } when stream=false
        try:
            return str(data.get("response", "")).strip()
        except Exception as exc:
            raise RuntimeError(f"Unexpected response format: {data}") from exc

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        options: dict[str, Any] | None = None,
        model_override: str | None = None,
    ) -> str:
        model = model_override or self.config.model
        payload = self._builder.chat_payload(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            options=options,
            stream=False,
        )
        data = self._http.chat(payload)
        try:
            return data["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response format: {data}") from exc

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        options: dict[str, Any] | None = None,
        model_override: str | None = None,
    ) -> Iterator[str]:
        """
        Yields incremental text chunks for /api/generate stream mode.
        """
        model = model_override or self.config.model
        payload = self._builder.generate_payload(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            options=options,
            stream=True,
        )

        for obj in self._http.generate_stream(payload):
            # Typical stream object includes {"response": "...", "done": false/true, ...}
            chunk = obj.get("response")
            if isinstance(chunk, str) and chunk:
                yield chunk

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        options: dict[str, Any] | None = None,
        model_override: str | None = None,
    ) -> Iterator[str]:
        """
        Yields incremental text chunks for /api/chat stream mode.
        """
        model = model_override or self.config.model
        payload = self._builder.chat_payload(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            options=options,
            stream=True,
        )

        for obj in self._http.chat_stream(payload):
            # Typical stream object includes {"message": {"role":"assistant","content":"..."}, "done": false/true}
            msg = obj.get("message")
            if isinstance(msg, dict):
                chunk = msg.get("content")
                if isinstance(chunk, str) and chunk:
                    yield chunk
