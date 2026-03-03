# **Local LLM Demo**

# `llm-local`

A modular, production-ready Python library and FastAPI service for interacting with **local LLM backends** (e.g. Ollama).

Designed for:

* Workshops
* Prototyping
* Internal AI tools
* Self-hosted assistants
* Backend experimentation
* Teaching LLM infrastructure

---

# ✨ Features

## ✅ Clean Python Client

`LocalLLM` provides:

* `generate()` — single prompt completion
* `chat()` — message-based completion
* `generate_stream()` — streaming completion
* `chat_stream()` — streaming chat
* Model lifecycle:

  * `list_models()`
  * `pull_model()`
  * `delete_model()`
  * `show_model()`
* Backend health checks
* Configurable temperature, max tokens, and backend options
* Model override per request

Implementation:
See `llm_local/llm_client.py` 

Configuration object:
See `llm_local/llm_config.py` 

---

## ✅ FastAPI HTTP API (Modular Architecture)

Entrypoint:

```bash
uvicorn llm_local.api:app
```

API definition:
`llm_local/api.py` 

Routers are split into modular components:

```
llm_local/api_parts/
├── deps.py
├── schemas.py
├── routers/
│   ├── system.py
│   ├── models.py
│   ├── generation.py
│   └── sessions.py
```

This keeps the app testable and maintainable.

---

# 🌐 API Endpoints

---

## 🧠 System

### `GET /health`

Checks backend availability.

Returns:

```json
{ "status": "ok" }
```

---

## 📦 Models

### `GET /models`

List locally available models.

### `POST /models/pull`

Pull/download a model.

```json
{
  "name": "llama3.2:3b"
}
```

### `DELETE /models/{name}`

Delete a local model.

### `GET /models/{name}`

Show backend metadata for a model.

---

## ✍️ Generation

### `POST /generate`

Non-streaming completion.

```json
{
  "prompt": "Explain neural networks",
  "temperature": 0.2
}
```

---

### `POST /generate/stream`

Streaming completion (SSE).

Events emitted:

* `meta`
* `delta`
* `done`
* `error`

Content type:

```
text/event-stream
```

---

### `POST /chat`

Non-streaming message-based completion.

```json
{
  "messages": [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "Explain RL."}
  ]
}
```

---

### `POST /chat/stream`

Streaming chat completion (SSE).

---

## 💬 Sessions (Persistent In-Memory Chat)

Prefix: `/sessions`

> ⚠ In-memory only. Not durable.
> See `deps.py` 

---

### `POST /sessions`

Create session.

```json
{
  "system_prompt": "You are helpful.",
  "model": "llama3.2:3b"
}
```

---

### `GET /sessions/{session_id}`

Session metadata.

---

### `GET /sessions/{session_id}/history`

Full chat history.

---

### `POST /sessions/{session_id}/messages`

Append message and return assistant reply.

---

### `POST /sessions/{session_id}/messages/stream`

Streaming version of session chat.

Query parameter:

```
?include_assistant_message=true
```

---

### `DELETE /sessions/{session_id}`

Delete session.

---

# 🚀 Quick Start (Docker – Recommended)

Docker Compose file:
`docker-compose.yml` 

## Start everything

```bash
docker compose up --build
```

This starts:

* `ollama`
* `ollama-init` (auto pulls model)
* `llm-local-api`

API available at:

```
http://localhost:8000
```

---

# 🐍 Local Development (Without Docker)

## 1️⃣ Install Ollama

```bash
ollama pull llama3.2:3b
ollama serve
```

---

## 2️⃣ Install package

Using uv:

```bash
uv pip install -e .
```

Or:

```bash
pip install -e .
```

Python ≥ 3.10 required.

---

## 3️⃣ Run API

```bash
uvicorn llm_local.api:app --reload
```

---

## Environment Variables

```bash
export LLM_BASE_URL=http://localhost:11434
export LLM_MODEL=llama3.2:3b
```

Defaults are defined in:

`llm_local/api_parts/deps.py` 

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
text = llm.generate("Explain transformers simply.")
print(text)

# Streaming
for chunk in llm.generate_stream("Write a poem."):
    print(chunk, end="")

# Chat
messages = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "Explain reinforcement learning."}
]

reply = llm.chat(messages)
print(reply)
```

---

# 🧠 Architecture

```
Client (Python / curl / frontend)
            ↓
        FastAPI (llm_local.api)
            ↓
        LocalLLM
            ↓
        OllamaHTTPClient
            ↓
        Ollama backend
            ↓
        Local model
```

Key files:

* Client: `llm_local/llm_client.py` 
* HTTP Layer: `llm_local/ollama_http_client.py` 
* Request Builder: `llm_local/ollama_request_builder.py` 

---

# 🔧 Development

Install dev dependencies:

```bash
uv pip install -e ".[dev]"
```

---

## Lint

```bash
ruff check .
```

---

## Type Checking

```bash
mypy .
```

---

# 🎯 Design Philosophy

* Thin backend wrapper
* Clear separation of concerns
* Streaming-first design
* Explicit model lifecycle control
* No hidden agent logic
* Extensible for RAG, tools, workflows

This project intentionally keeps:

* No database
* No background workers
* No complex orchestration

Build those on top of it.