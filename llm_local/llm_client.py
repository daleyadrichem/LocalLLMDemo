from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from llm_local.utils import chunk_text, build_summarization_prompt

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
    max_chunk_chars:
        Maximum characters per text chunk when summarizing long documents.
    chunk_overlap_chars:
        Overlap between consecutive chunks when summarizing long documents.
    """

    model: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 120
    max_chunk_chars: int = 4000
    chunk_overlap_chars: int = 200


@dataclass
class LocalLLM:
    """
    Client for interacting with a local LLM server.

    This implementation targets the Ollama HTTP API but is intentionally
    structured so you can swap in a different backend later if needed.

    Typical usage
    -------------
    >>> llm = LocalLLM()
    >>> result = llm.generate("Explain what a neural network is.")
    >>> print(result)

    For summarization of a long text:
    >>> summary = llm.summarize_text(long_text, max_words=200)
    """

    config: LocalLLMConfig = field(default_factory=LocalLLMConfig)
    _session: requests.Session = field(init=False, repr=False)

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
    # Public high-level methods
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text from the local LLM given a prompt.

        Parameters
        ----------
        prompt:
            The user prompt to send to the model.
        system_prompt:
            Optional system-level instruction describing the assistant's role.
        temperature:
            Sampling temperature; higher values produce more diverse output.
        max_tokens:
            Optional maximum number of tokens to generate.

        Returns
        -------
        str
            The model's response text.

        Raises
        ------
        RuntimeError
            If the request to the LLM backend fails.
        """
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        # Ollama uses 'num_predict' instead of 'max_tokens'
        if max_tokens is not None and max_tokens > 0:
            payload["options"]["num_predict"] = max_tokens

        logger.debug("Sending generate request: %s", payload)

        try:
            response = self._session.post(
                f"{self.config.base_url}/api/chat",
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.exception("Failed to call local LLM backend.")
            raise RuntimeError(f"Failed to call local LLM backend: {exc}") from exc

        data = response.json()
        logger.debug("Received response: %s", data)

        try:
            # Ollama chat API returns a single "message" with "content"
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            logger.exception("Unexpected response format from LLM backend.")
            raise RuntimeError(f"Unexpected response format: {data}") from exc

        return content.strip()

    def summarize_text(
        self,
        text: str,
        max_words: Optional[int] = 200,
        temperature: float = 0.0,
    ) -> str:
        """
        Summarize a potentially long text using the local LLM.

        The text is automatically split into overlapping chunks if it is
        longer than the configured chunk size. The summaries of the chunks
        are then combined into a final global summary.

        Parameters
        ----------
        text:
            The input text to summarize.
        max_words:
            Maximum number of words for the final summary. If None or
            non-positive, no explicit limit is requested (the model may
            still choose to be concise).
        temperature:
            Sampling temperature for the summarization calls.

        Returns
        -------
        str
            The summarized text.
        """
        text = text.strip()
        if not text:
            return ""

        chunks = chunk_text(
            text,
            max_chars=self.config.max_chunk_chars,
            overlap=self.config.chunk_overlap_chars,
        )

        logger.info("Summarizing text with %d chunk(s).", len(chunks))

        if len(chunks) == 1:
            prompt = build_summarization_prompt(chunks[0], max_words=max_words)
            return self.generate(prompt, temperature=temperature)

        # Step 1: summarize each chunk individually
        partial_summaries: List[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            logger.debug("Summarizing chunk %d/%d", idx, len(chunks))
            prompt = build_summarization_prompt(chunk, max_words=max_words)
            summary = self.generate(prompt, temperature=temperature)
            partial_summaries.append(summary)

        # Step 2: combine partial summaries into a final global summary
        combined_text = "\n\n".join(
            f"Chunk {i + 1} summary:\n{summary}"
            for i, summary in enumerate(partial_summaries)
        )

        combine_prompt = (
            "You are a helpful assistant that creates a single, coherent summary "
            "from multiple partial summaries.\n\n"
            "Read the following partial summaries and produce one clear, concise "
            "overall summary that captures the main points of the original document.\n\n"
            f"{combined_text}\n\n"
            "Overall summary"
        )

        if max_words is not None and max_words > 0:
            combine_prompt += f" (in at most {max_words} words)"

        combine_prompt += ":"

        logger.debug("Combining %d partial summaries into final summary.", len(partial_summaries))

        final_summary = self.generate(combine_prompt, temperature=temperature)
        return final_summary.strip()
