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
    """High-level client for interacting with an Ollama backend.

    Parameters
    ----------
    config : LocalLLMConfig, optional
        Configuration containing the backend URL, default model name, timeout,
        and related request defaults.

    Attributes
    ----------
    config : LocalLLMConfig
        Active client configuration.
    _http : OllamaHTTPClient
        Low-level HTTP wrapper used for Ollama endpoints.
    _builder : OllamaRequestBuilder
        Payload builder for generate and chat requests.
    """

    config: LocalLLMConfig = field(default_factory=LocalLLMConfig)
    _http: OllamaHTTPClient = field(init=False, repr=False)
    _builder: OllamaRequestBuilder = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize internal HTTP and payload builder helpers."""
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
        """Check whether the Ollama backend is reachable.

        Returns
        -------
        bool
            ``True`` when the backend responds to the tags endpoint,
            otherwise ``False``.
        """
        try:
            self._http.tags()
            return True
        except RuntimeError as exc:
            logger.warning("LLM backend not available: %s", exc)
            return False

    def list_models(self) -> list[str]:
        """List available model names from the backend.

        Returns
        -------
        list of str
            Model names returned by Ollama.

        Raises
        ------
        RuntimeError
            If the response cannot be parsed as expected.
        """
        data = self._http.tags()
        try:
            return [m["name"] for m in data.get("models", [])]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response format: {data}") from exc

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def pull_model(self, name: str) -> None:
        """Download a model to the local Ollama store.

        Parameters
        ----------
        name : str
            Model identifier to pull.

        Raises
        ------
        RuntimeError
            If ``name`` is empty or whitespace-only.
        """
        if not name or not name.strip():
            raise RuntimeError("Model name is required")
        self._http.pull(name.strip())

    def delete_model(self, name: str) -> None:
        """Delete a model from the local Ollama store.

        Parameters
        ----------
        name : str
            Model identifier to delete.

        Raises
        ------
        RuntimeError
            If ``name`` is empty or whitespace-only.
        """
        if not name or not name.strip():
            raise RuntimeError("Model name is required")
        self._http.delete(name.strip())

    def show_model(self, name: str) -> dict[str, Any]:
        """Fetch metadata for a specific model.

        Parameters
        ----------
        name : str
            Model identifier to inspect.

        Returns
        -------
        dict of str to Any
            Model metadata as returned by Ollama.

        Raises
        ------
        RuntimeError
            If ``name`` is empty or whitespace-only.
        """
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
        """Generate a non-streamed completion from a single prompt.

        Parameters
        ----------
        prompt : str
            User prompt to send to the model.
        system_prompt : str or None, optional
            Optional system instruction prepended server-side.
        temperature : float or None, optional
            Sampling temperature override.
        max_tokens : int or None, optional
            Maximum number of tokens to generate.
        options : dict of str to Any or None, optional
            Additional Ollama request options.
        model_override : str or None, optional
            Model name to use instead of ``config.model``.

        Returns
        -------
        str
            Generated text response.

        Raises
        ------
        RuntimeError
            If the backend response format is unexpected.
        """
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
        """Generate a non-streamed chat completion.

        Parameters
        ----------
        messages : list of dict of str to str
            Chat messages, each containing at least ``role`` and ``content``.
        temperature : float or None, optional
            Sampling temperature override.
        max_tokens : int or None, optional
            Maximum number of tokens to generate.
        options : dict of str to Any or None, optional
            Additional Ollama request options.
        model_override : str or None, optional
            Model name to use instead of ``config.model``.

        Returns
        -------
        str
            Assistant message text from the response.

        Raises
        ------
        RuntimeError
            If the backend response format is unexpected.
        """
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
        """Stream completion chunks for a single prompt.

        Parameters
        ----------
        prompt : str
            User prompt to send to the model.
        system_prompt : str or None, optional
            Optional system instruction prepended server-side.
        temperature : float or None, optional
            Sampling temperature override.
        max_tokens : int or None, optional
            Maximum number of tokens to generate.
        options : dict of str to Any or None, optional
            Additional Ollama request options.
        model_override : str or None, optional
            Model name to use instead of ``config.model``.

        Yields
        ------
        str
            Incremental text chunks from the stream response.
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
        """Stream chat completion chunks.

        Parameters
        ----------
        messages : list of dict of str to str
            Chat messages, each containing at least ``role`` and ``content``.
        temperature : float or None, optional
            Sampling temperature override.
        max_tokens : int or None, optional
            Maximum number of tokens to generate.
        options : dict of str to Any or None, optional
            Additional Ollama request options.
        model_override : str or None, optional
            Model name to use instead of ``config.model``.

        Yields
        ------
        str
            Incremental assistant text chunks from the stream response.
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
