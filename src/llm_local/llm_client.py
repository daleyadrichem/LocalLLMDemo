from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class LocalLLMConfig:
    """
    Configuration for connecting to a local LLM server (e.g. Ollama).

    Parameters
    ----------
    model:
        Name of the local model (as known by Ollama).
    base_url:
        Base URL of the LLM server.
    timeout_seconds:
        HTTP timeout in seconds for requests.
    default_temperature:
        Default sampling temperature used if none is provided per call.
    default_max_tokens:
        Optional default maximum number of tokens to generate if none is
        provided per call. This is mapped to the backend-specific field
        (e.g. 'num_predict' in Ollama).
    default_options:
        Additional backend-specific options to send on every request.
        For Ollama, this can include keys like 'num_ctx', 'top_p', etc.

    Notes
    -----
    GPU usage
    ---------
    This client does not explicitly toggle GPU on or off. Backends like
    Ollama automatically use the GPU if they are installed and configured
    with GPU support. To ensure the model runs on the GPU, configure your
    LLM runtime (e.g. Ollama) appropriately. This class simply sends
    requests to that runtime.
    """

    model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 360
    default_temperature: float = 0.2
    default_max_tokens: Optional[int] = None
    default_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalLLM:
    """
    Client for interacting with a local LLM server.

    This class is intentionally focused on a single responsibility:
    sending prompts (or chat messages) to a local LLM backend and
    returning the raw model outputs. Higher-level functionality such
    as summarization, code generation, or file manipulation should be
    implemented in separate, dedicated classes that depend on this one.

    Typical usage
    -------------
    >>> llm = LocalLLM()
    >>> text = llm.generate("Explain what a neural network is.")
    >>> print(text)

    For a chat-style interaction:
    >>> messages = [
    ...     {"role": "system", "content": "You are a helpful assistant."},
    ...     {"role": "user", "content": "Hi, who are you?"},
    ... ]
    >>> reply = llm.chat(messages)
    >>> print(reply)
    """

    config: LocalLLMConfig = field(default_factory=LocalLLMConfig)
    _session: requests.Session = field(init=False, repr=False)
    _chat_history: Optional[List[Dict[str, str]]] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """
        Initialize the underlying HTTP session.
        """
        self._session = requests.Session()
        logger.debug(
            "Initialized LocalLLM with model=%s, base_url=%s",
            self.config.model,
            self.config.base_url,
        )

    # ------------------------------------------------------------------
    # Public methods: health / metadata
    # ------------------------------------------------------------------

    def is_backend_available(self) -> bool:
        """
        Check whether the LLM backend is reachable.

        This performs a lightweight request to the backend. For Ollama,
        it queries the /api/tags endpoint.

        Returns
        -------
        bool
            True if the backend responds successfully, False otherwise.
        """
        try:
            response = self._session.get(
                f"{self.config.base_url}/api/tags",
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("LLM backend not available: %s", exc)
            return False

        return True

    def list_models(self) -> List[str]:
        """
        List the models available on the local LLM backend.

        For Ollama, this calls the /api/tags endpoint and returns the
        model names.

        Returns
        -------
        list of str
            A list of available model names.

        Raises
        ------
        RuntimeError
            If the request to the backend fails or the response format
            is not as expected.
        """
        try:
            response = self._session.get(
                f"{self.config.base_url}/api/tags",
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.exception("Failed to list models from LLM backend.")
            raise RuntimeError(f"Failed to list models: {exc}") from exc

        data = response.json()
        logger.debug("Received models metadata: %s", data)

        try:
            # Ollama returns a dict with a 'models' list; each item has a 'name'
            models = [m["name"] for m in data.get("models", [])]
        except (KeyError, TypeError) as exc:
            logger.exception("Unexpected response format when listing models.")
            raise RuntimeError(f"Unexpected response format: {data}") from exc

        return models

    # ------------------------------------------------------------------
    # Public methods: text generation and chat
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate text from the local LLM given a single user prompt.

        This is a convenience method that internally uses a chat-style
        API but exposes a simple prompt-based interface.

        Parameters
        ----------
        prompt:
            The user prompt to send to the model.
        system_prompt:
            Optional system-level instruction describing the assistant's role.
        temperature:
            Sampling temperature; higher values produce more diverse output.
            If None, the value from LocalLLMConfig.default_temperature is used.
        max_tokens:
            Optional maximum number of tokens to generate. If None, the value
            from LocalLLMConfig.default_max_tokens is used. This is mapped to
            the backend-specific field (e.g. 'num_predict' in Ollama).
        options:
            Optional backend-specific options to merge with the default options
            defined in LocalLLMConfig.default_options.

        Returns
        -------
        str
            The model's response text.

        Raises
        ------
        RuntimeError
            If the request to the LLM backend fails or returns an unexpected
            format.
        """
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            options=options,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send a list of chat messages to the local LLM and return the reply.

        Parameters
        ----------
        messages:
            A list of chat messages. Each message is a dict with keys:
            - 'role': typically 'system', 'user', or 'assistant'.
            - 'content': the message text.
            This matches the format used by many chat-based LLM APIs.
        temperature:
            Sampling temperature; higher values produce more diverse output.
            If None, the value from LocalLLMConfig.default_temperature is used.
        max_tokens:
            Optional maximum number of tokens to generate. If None, the value
            from LocalLLMConfig.default_max_tokens is used. This is mapped to
            the backend-specific field (e.g. 'num_predict' in Ollama).
        options:
            Optional backend-specific options to merge with the default options
            defined in LocalLLMConfig.default_options.

        Returns
        -------
        str
            The model's response text from the assistant.

        Raises
        ------
        RuntimeError
            If the request to the LLM backend fails or returns an unexpected
            format.
        """
        effective_temperature = (
            temperature if temperature is not None else self.config.default_temperature
        )
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self.config.default_max_tokens
        )

        request_options = dict(self.config.default_options)
        request_options["temperature"] = effective_temperature

        if effective_max_tokens is not None and effective_max_tokens > 0:
            request_options["num_predict"] = effective_max_tokens

        if options:
            request_options.update(options)

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": request_options,
        }

        try:
            response = self._session.post(
                f"{self.config.base_url}/api/chat",
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            if not response.ok:
                logger.error("Ollama status: %s", response.status_code)
                logger.error("Ollama body: %s", response.text)
                response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to call local LLM backend: {exc}") from exc

        data = response.json()
        try:
            return data["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response format: {data}") from exc
        
    # ------------------------------------------------------------------
    # NEW SECTION: PERSISTENT CHAT SUPPORT
    # ------------------------------------------------------------------

    def start_chat(self, system_prompt: Optional[str] = None) -> None:
        """
        Start a persistent chat session.
        Optionally pass a system prompt.
        """
        self._chat_history = []
        if system_prompt:
            self._chat_history.append({"role": "system", "content": system_prompt})

    def send_chat_message(
        self,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a message to the chat history, send the full conversation
        to the LLM, and store its response.

        Returns the assistant's reply.
        """
        if self._chat_history is None:
            raise RuntimeError(
                "No chat session active. Call start_chat() first."
            )

        # Append user message
        self._chat_history.append({"role": "user", "content": user_message})

        # Send full history
        reply = self.chat(
            messages=self._chat_history,
            temperature=temperature,
            max_tokens=max_tokens,
            options=options,
        )

        # Append assistant reply
        self._chat_history.append({"role": "assistant", "content": reply})

        return reply

    def get_history(self) -> List[Dict[str, str]]:
        """
        Return the current chat history.
        """
        return list(self._chat_history or [])

    def reset_chat(self) -> None:
        """
        Clear the chat history (equivalent to ending the session).
        """
        self._chat_history = None