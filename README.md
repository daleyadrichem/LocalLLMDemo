# **Local LLM Demo**

# `llm-local`

A small, reusable Python library and API service for interacting with **local LLMs** (e.g. Ollama), designed for:

* Workshops
* Demos
* Prototyping
* Internal tools
* Self-hosted AI services

This project provides:

* ✅ A clean, reusable `LocalLLM` Python client
* ✅ A FastAPI HTTP wrapper for serving the model
* ✅ Persistent chat session support
* ✅ Docker + Docker Compose setup (Ollama + API)
* ✅ CLI demo for document summarization
* ✅ Type hints, linting, and static typing

---

# ✨ Features

## 1️⃣ Reusable Python Client

`LocalLLM` (see `llm_client.py`)  provides:

* Simple `generate()` interface
* Chat-style `chat()` interface
* Persistent chat sessions:

  * `start_chat()`
  * `send_chat_message()`
  * `get_history()`
  * `reset_chat()`
* Backend health checks
* Model listing
* Configurable temperature, max tokens, and backend options

---

## 2️⃣ FastAPI HTTP API

The project includes a production-ready API layer (`api.py`) .

### Available Endpoints

### System

* `GET /health`
* `GET /models`

### Generation

* `POST /generate`
* `POST /chat`

### Persistent Chat

* `POST /chat/start`
* `POST /chat/send`
* `GET /chat/history`
* `POST /chat/reset`

The API wraps the `LocalLLM` client and exposes it as a clean HTTP service.

---

## 3️⃣ Dockerized Setup (Ollama + API)

A full `docker-compose.yml` is included .

It runs:

* `ollama` – LLM backend
* `ollama-init` – pulls the configured model automatically
* `app` – FastAPI service wrapping the LLM

---

# 🚀 Quick Start (Docker – Recommended)

## 1️⃣ Start Everything

From the project root:

```bash
docker compose up --build
```

This will:

* Start Ollama
* Pull `llama3.2:3b`
* Start the FastAPI service on:

```
http://localhost:8000
```

---

## 2️⃣ Test the API

### Health check

```bash
curl http://localhost:8000/health
```

### List models

```bash
curl http://localhost:8000/models
```

### Generate text

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain what a neural network is.",
    "temperature": 0.2
  }'
```

### Start persistent chat

```bash
curl -X POST "http://localhost:8000/chat/start?system_prompt=You%20are%20helpful"
```

### Send message

```bash
curl -X POST http://localhost:8000/chat/send \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi! Who are you?"}'
```

---

# 🐍 Local Development (Without Docker)

## 1️⃣ Install Ollama

Install Ollama and pull a model:

```bash
ollama pull llama3.2:3b
ollama serve
```

---

## 2️⃣ Install the package

Using `uv` (recommended):

```bash
uv pip install -e .
```

Or:

```bash
pip install -e .
```

Python ≥ 3.10 required.

---

## 3️⃣ Run the API locally

```bash
uvicorn api:app --reload
```

Default configuration:

* Base URL: `http://localhost:11434`
* Model: `llama3.2:3b`

You can override via environment variables:

```bash
export LLM_BASE_URL=http://localhost:11434
export LLM_MODEL=llama3.2:3b
```

---

# 📚 Using the Library Directly

```python
from llm_local import LocalLLM, LocalLLMConfig

config = LocalLLMConfig(
    model="llama3.2:3b",
    base_url="http://localhost:11434"
)

llm = LocalLLM(config=config)

# Simple generation
text = llm.generate("Explain transformers in simple terms.")
print(text)

# Chat
messages = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "What is reinforcement learning?"}
]

reply = llm.chat(messages)
print(reply)

# Persistent chat
llm.start_chat(system_prompt="You are a coding assistant.")
print(llm.send_chat_message("Write a Python function for factorial."))
print(llm.get_history())
```

---

# 🗂 Project Structure

```
.
├── api.py                  # FastAPI HTTP wrapper
├── demo_summarize.py       # CLI demo
├── docker-compose.yml      # Multi-container setup (Ollama + API)
├── Dockerfile              # App container build
├── llm_local/
│   ├── __init__.py
│   └── llm_client.py       # LocalLLM implementation
├── examples/
├── pyproject.toml
└── README.md
```

---

# 🔧 Development

## Install dev dependencies

```bash
uv pip install -e ".[dev]"
```

or

```bash
pip install -e ".[dev]"
```

---

## Lint

```bash
ruff check .
```

---

## Type checking

```bash
mypy .
```

---

# 🧠 Architecture Overview

```
Client (Python / curl / frontend)
            ↓
        FastAPI (api.py)
            ↓
        LocalLLM client
            ↓
        Ollama backend
            ↓
        Local model (llama3, mistral, etc.)
```

The `LocalLLM` class is intentionally minimal and backend-focused.
Higher-level logic (agents, RAG, tools, workflows) should be built **on top of it**, not inside it.

---

# 🎯 Intended Use Cases

* Internal AI tools
* Self-hosted assistants
* Workshop demos
* Prototyping before using cloud LLMs
* Teaching how LLM backends work
* Building agent systems on local models