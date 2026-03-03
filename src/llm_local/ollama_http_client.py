from __future__ import annotations

from collections.abc import Iterator
import json
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class OllamaHTTPClient:
    """
    Low-level HTTP client for the Ollama API.
    Keeps request/response parsing in one place.
    """

    def __init__(self, base_url: str, timeout_seconds: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    # -----------------------------
    # Utilities
    # -----------------------------

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout_seconds)
            if not resp.ok:
                logger.error("Ollama status: %s", resp.status_code)
                logger.error("Ollama body: %s", resp.text)
                resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to call local LLM backend: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError("Backend returned non-JSON response") from exc

    def _post_stream(self, path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """
        Ollama streaming endpoints return NDJSON (one JSON object per line).
        """
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout_seconds, stream=True)
            if not resp.ok:
                logger.error("Ollama status: %s", resp.status_code)
                logger.error("Ollama body: %s", resp.text)
                resp.raise_for_status()

            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Some proxies/runtime configs can yield non-JSON lines; ignore safely.
                    continue
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to stream from local LLM backend: {exc}") from exc

    def _get_json(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.get(url, timeout=self.timeout_seconds)
            if not resp.ok:
                logger.error("Ollama status: %s", resp.status_code)
                logger.error("Ollama body: %s", resp.text)
                resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to call local LLM backend: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError("Backend returned non-JSON response") from exc

    # -----------------------------
    # Ollama endpoints
    # -----------------------------

    def tags(self) -> dict[str, Any]:
        return self._get_json("/api/tags")

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json("/api/chat", payload)

    def chat_stream(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        return self._post_stream("/api/chat", payload)

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json("/api/generate", payload)

    def generate_stream(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        return self._post_stream("/api/generate", payload)

    def pull(self, name: str) -> None:
        # Ollama supports /api/pull (streaming). We'll call non-streaming-safe by consuming stream.
        payload = {"name": name, "stream": True}
        for _ in self._post_stream("/api/pull", payload):
            pass

    def delete(self, name: str) -> None:
        payload = {"name": name}
        self._post_json("/api/delete", payload)

    def show(self, name: str) -> dict[str, Any]:
        payload = {"name": name}
        return self._post_json("/api/show", payload)
