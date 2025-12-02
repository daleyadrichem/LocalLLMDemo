from __future__ import annotations

from pathlib import Path
from typing import List


def load_text_file(path: str | Path, encoding: str = "utf-8") -> str:
    """
    Load and return the contents of a text file.

    Parameters
    ----------
    path:
        Path to the text file.
    encoding:
        File encoding. Defaults to UTF-8.

    Returns
    -------
    str
        Contents of the file as a single string.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    UnicodeDecodeError
        If the file cannot be decoded using the provided encoding.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    return file_path.read_text(encoding=encoding)


def chunk_text(
    text: str,
    max_chars: int = 4000,
    overlap: int = 200,
) -> List[str]:
    """
    Split text into overlapping character-based chunks.

    This is a simple, model-agnostic chunking strategy that works reasonably
    well for many local LLM use cases without needing tokenization.

    Parameters
    ----------
    text:
        The input text to chunk.
    max_chars:
        Maximum number of characters per chunk.
    overlap:
        Number of overlapping characters between chunks. This helps give
        the model some context continuity between chunks.

    Returns
    -------
    list of str
        List of text chunks.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars.")

    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chars, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Move the window forward with overlap
        if end == text_len:
            break
        start = end - overlap

    return chunks


def build_summarization_prompt(chunk: str, max_words: int | None = None) -> str:
    """
    Build a summarization prompt for a given text chunk.

    Parameters
    ----------
    chunk:
        The text chunk to summarize.
    max_words:
        Optional maximum number of words to request in the summary.

    Returns
    -------
    str
        A prompt string suitable for sending to an LLM.
    """
    limit_instruction = (
        f"in at most {max_words} words "
        if max_words is not None and max_words > 0
        else ""
    )

    return (
        "You are a helpful assistant that summarizes documents.\n\n"
        "Summarize the following text "
        f"{limit_instruction}"
        "using clear, concise language that a non-expert can understand.\n\n"
        "Text:\n"
        "------\n"
        f"{chunk}\n"
        "------\n\n"
        "Summary:"
    )
