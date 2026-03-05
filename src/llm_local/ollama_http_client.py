from __future__ import annotations

from collections.abc import Iterator
import json
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class OllamaHTTPClient:
    """Low-level HTTP client for the Ollama API.

    This client centralizes HTTP request handling and JSON/NDJSON parsing for
    Ollama endpoints.

    Parameters
    ----------
    base_url : str
        Base URL for the Ollama server, for example ``http://localhost:11434``.
    timeout_seconds : int
        Request timeout in seconds for all outbound HTTP calls.
    """

    def __init__(self, base_url: str, timeout_seconds: int) -> None:
        """Initialize the Ollama HTTP client.

        Parameters
        ----------
        base_url : str
            Base URL for the Ollama server.
        timeout_seconds : int
            Per-request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    # -----------------------------
    # Utilities
    # -----------------------------

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON POST request and parse a JSON response body.

        Parameters
        ----------
        path : str
            API route appended to ``base_url``.
        payload : dict[str, Any]
            JSON payload sent in the POST body.

        Returns
        -------
        dict[str, Any]
            Parsed JSON response content.

        Raises
        ------
        RuntimeError
            If the HTTP request fails or the response body is not valid JSON.
        """
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
        """Send a streaming POST request and yield NDJSON objects.

        Ollama streaming endpoints return NDJSON (one JSON object per line).

        Parameters
        ----------
        path : str
            API route appended to ``base_url``.
        payload : dict[str, Any]
            JSON payload sent in the POST body.

        Yields
        ------
        dict[str, Any]
            Parsed JSON object for each valid line in the stream.

        Raises
        ------
        RuntimeError
            If the HTTP streaming request fails.
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
        """Send a GET request and parse a JSON response body.

        Parameters
        ----------
        path : str
            API route appended to ``base_url``.

        Returns
        -------
        dict[str, Any]
            Parsed JSON response content.

        Raises
        ------
        RuntimeError
            If the HTTP request fails or the response body is not valid JSON.
        """
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
        """Fetch available model tags from the Ollama server.

        Returns
        -------
        dict[str, Any]
            Response payload from the ``/api/tags`` endpoint.
        """
        return self._get_json("/api/tags")

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Call the Ollama chat endpoint with a JSON payload.

        Parameters
        ----------
        payload : dict[str, Any]
            Request body for ``/api/chat``.

        Returns
        -------
        dict[str, Any]
            Parsed JSON response payload.
        """
        return self._post_json("/api/chat", payload)

    def chat_stream(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Stream responses from the Ollama chat endpoint.

        Parameters
        ----------
        payload : dict[str, Any]
            Request body for ``/api/chat``.

        Yields
        ------
        dict[str, Any]
            Parsed NDJSON objects from the chat stream.
        """
        return self._post_stream("/api/chat", payload)

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Call the Ollama generate endpoint with a JSON payload.

        Parameters
        ----------
        payload : dict[str, Any]
            Request body for ``/api/generate``.

        Returns
        -------
        dict[str, Any]
            Parsed JSON response payload.
        """
        return self._post_json("/api/generate", payload)

    def generate_stream(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Stream responses from the Ollama generate endpoint.

        Parameters
        ----------
        payload : dict[str, Any]
            Request body for ``/api/generate``.

        Yields
        ------
        dict[str, Any]
            Parsed NDJSON objects from the generate stream.
        """
        return self._post_stream("/api/generate", payload)

    def pull(self, name: str) -> None:
        """Pull a model from Ollama by consuming the streaming pull endpoint.

        Parameters
        ----------
        name : str
            Model name to pull from Ollama.
        """
        # Ollama supports /api/pull (streaming). We'll call non-streaming-safe by consuming stream.
        payload = {"name": name, "stream": True}
        for _ in self._post_stream("/api/pull", payload):
            pass

    def delete(self, name: str) -> None:
        """Delete a local Ollama model by name.

        Parameters
        ----------
        name : str
            Model name to delete.
        """
        payload = {"name": name}
        self._post_json("/api/delete", payload)

    def show(self, name: str) -> dict[str, Any]:
        """Fetch metadata/details for a model.

        Parameters
        ----------
        name : str
            Model name to inspect.

        Returns
        -------
        dict[str, Any]
            Parsed JSON response payload from ``/api/show``.
        """
        payload = {"name": name}
        return self._post_json("/api/show", payload)
