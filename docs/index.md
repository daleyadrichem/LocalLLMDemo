# Local LLM Demo

A modular Python library + FastAPI service for interacting with local LLM backends (e.g. Ollama).

This project provides:

- A reusable `LocalLLM` Python client
- Streaming + non-streaming generation
- Model lifecycle management
- In-memory persistent sessions
- A production-ready FastAPI HTTP API
- Docker + Docker Compose setup

---

## Architecture

Client (Python / curl / frontend)  
↓  
FastAPI (`llm_local.api`)  
↓  
LocalLLM  
↓  
OllamaHTTPClient  
↓  
Ollama backend  
↓  
Local model

---

## Quick Links

- [Docker Quickstart](guides/docker.md)
- [Local Development](guides/local-dev.md)
- [Streaming (SSE)](guides/streaming.md)
- [Sessions](guides/sessions.md)
- [Model Management](guides/models.md)
- [Python API Reference](reference/python.md)
- [HTTP API Reference](reference/http.md)