from __future__ import annotations

import logging
from typing import Any

from llm_local.llm_config import LocalLLMConfig

logger = logging.getLogger(__name__)


class OllamaRequestBuilder:
    """Build request payloads for Ollama endpoints.

    Parameters
    ----------
    config : LocalLLMConfig
        Shared configuration containing default generation options.
    """

    def __init__(self, config: LocalLLMConfig) -> None:
        """Initialize the request builder.

        Parameters
        ----------
        config : LocalLLMConfig
            Runtime configuration used to fill request defaults.
        """
        self.config = config

    def _merged_options(
        self,
        temperature: float | None,
        max_tokens: int | None,
        options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Merge request-level options with configured defaults.

        Parameters
        ----------
        temperature : float | None
            Temperature override for the request. When ``None``, the configured
            default temperature is used.
        max_tokens : int | None
            Token limit override for the request. When positive, this value is
            mapped to Ollama's ``num_predict`` option.
        options : dict[str, Any] | None
            Additional Ollama options that should override merged defaults.

        Returns
        -------
        dict[str, Any]
            Effective options dictionary to include in an Ollama request.
        """
        effective_temperature = (
            temperature if temperature is not None else self.config.default_temperature
        )
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self.config.default_max_tokens
        )

        request_options = dict(self.config.default_options)
        request_options["temperature"] = effective_temperature

        # Ollama uses num_predict for max tokens.
        if effective_max_tokens is not None and effective_max_tokens > 0:
            request_options["num_predict"] = effective_max_tokens

        if options:
            request_options.update(options)

        return request_options

    def chat_payload(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        options: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        """Build a payload for Ollama's chat endpoint.

        Parameters
        ----------
        model : str
            Model name to target for chat completion.
        messages : list[dict[str, str]]
            Chat messages represented as role/content dictionaries.
        temperature : float | None
            Optional temperature override for the request.
        max_tokens : int | None
            Optional token limit override for the request.
        options : dict[str, Any] | None
            Optional additional Ollama options.
        stream : bool
            Whether the server should stream partial responses.

        Returns
        -------
        dict[str, Any]
            JSON-serializable payload for the chat API.
        """
        return {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": self._merged_options(temperature, max_tokens, options),
        }

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
        """Build a payload for Ollama's generate endpoint.

        Parameters
        ----------
        model : str
            Model name to target for text generation.
        prompt : str
            User prompt passed to the model.
        system_prompt : str | None
            Optional system instruction included as the ``system`` field.
        temperature : float | None
            Optional temperature override for the request.
        max_tokens : int | None
            Optional token limit override for the request.
        options : dict[str, Any] | None
            Optional additional Ollama options.
        stream : bool
            Whether the server should stream partial responses.

        Returns
        -------
        dict[str, Any]
            JSON-serializable payload for the generate API.
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": self._merged_options(temperature, max_tokens, options),
        }
        if system_prompt:
            payload["system"] = system_prompt
        return payload
