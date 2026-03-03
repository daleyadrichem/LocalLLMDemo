from __future__ import annotations

import logging
from typing import Any

from llm_local.llm_config import LocalLLMConfig

logger = logging.getLogger(__name__)


class OllamaRequestBuilder:
    """
    Builds backend-specific payloads from high-level inputs.
    """

    def __init__(self, config: LocalLLMConfig) -> None:
        self.config = config

    def _merged_options(
        self,
        temperature: float | None,
        max_tokens: int | None,
        options: dict[str, Any] | None,
    ) -> dict[str, Any]:
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
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": self._merged_options(temperature, max_tokens, options),
        }
        if system_prompt:
            payload["system"] = system_prompt
        return payload
