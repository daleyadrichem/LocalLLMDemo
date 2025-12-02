# **Local LLM Demo**

# llm-local

A small, reusable Python library for interacting with **local LLMs** (e.g. [Ollama](https://ollama.com/)),
with a focus on **document summarization**.

This is designed for workshops, demos, and real projects:
- Clean, library-style `LocalLLM` client
- Simple utilities for loading and chunking text
- A CLI script for summarizing `.txt` files
- Example notebooks for interactive exploration

---

## Features

- ✅ Local LLM client (`LocalLLM`) using the **Ollama** HTTP API
- ✅ Chunking of large documents into overlapping segments
- ✅ Hierarchical summarization (chunk summaries → global summary)
- ✅ Reusable utilities (`utils.py`) for text loading, chunking, prompts
- ✅ CLI demo for workshops: summarize a `.txt` document live
- ✅ Type hints, docstrings, and basic tooling (ruff, mypy)

---

## Installation

### 1. Install Ollama and pull a model

Follow the instructions on the Ollama website, then:

```bash
ollama pull llama3.2:3b
```

You can use another model if you prefer (e.g. `mistral`, `qwen`, etc.).

Start the Ollama server (if not already running):

```bash
ollama serve
```

Alternatively, to download and run this exact model, uv can be used with:

```bash
uv run pull
uv run serve
```

### 2. Install the package (using uv)

From the project root:

```bash
uv pip install -e .
```

This will install `llm-local` in editable mode.

If you prefer `pip`:

```bash
pip install -e .
```

> Make sure you have Python ≥ 3.10.

---

## Quickstart

### Summarize a text file (CLI demo)

From the project root:

```bash
python demo_summarize.py path/to/document.txt \
    --model llama3.2:3b \
    --max-words 200 \
    --temperature 0.0 \
    --verbose
```

Output will look something like:

```text
================================================================================
SUMMARY
================================================================================

<summary text here>

================================================================================
```

This is perfect for a **live workshop demo**:

* Show the original document
* Run the command
* Show the concise summary produced locally on your machine

---

## Library usage

You can also use `LocalLLM` directly in Python:

```python
from llm_local import LocalLLM, LocalLLMConfig

config = LocalLLMConfig(model="llama3.2:3b")
llm = LocalLLM(config=config)

# Simple generation
response = llm.generate("Explain what a neural network is in simple terms.")
print(response)

# Summarize a long text
long_text = "..."  # your document text
summary = llm.summarize_text(long_text, max_words=200)
print(summary)
```

---

## Project structure

```text
llm_local/
  __init__.py        # exports LocalLLM
  llm_client.py      # LocalLLM and LocalLLMConfig classes
  utils.py           # load_text_file, chunk_text, build_summarization_prompt

demo_summarize.py    # CLI demo for summarizing .txt documents

examples/
  01_quickstart_summarize.ipynb       # basic usage demo in a notebook
  02_experiment_with_prompts.ipynb    # playing with prompts and settings

pyproject.toml       # project metadata and build configuration (uv_build)
ruff.toml            # linting configuration (Ruff)
mypy.ini             # static typing configuration (mypy)
```

---

## Development

### Install dev dependencies

Using `uv` (recommended):

```bash
uv pip install -e ".[dev]"
```

Or using `pip`:

```bash
pip install -e ".[dev]"
```

### Run Ruff (lint)

```bash
ruff check .
```

Ruff config is in `ruff.toml`.

### Run mypy (type checking)

```bash
mypy .
```