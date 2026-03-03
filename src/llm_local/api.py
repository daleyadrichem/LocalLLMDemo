"""
FastAPI application entrypoint.

This module is intentionally small to keep a stable Uvicorn import path:

    uvicorn llm_local.api:app

Implementation is split across ``llm_local.api_parts`` to keep files smaller,
more testable, and easier to maintain.
"""

from __future__ import annotations

from fastapi import FastAPI

from llm_local.api_parts.routers import generation, models, sessions, system

app = FastAPI(
    title="Local LLM API",
    description="HTTP API wrapper for a local LLM backend (e.g. Ollama), with streaming + sessions + model lifecycle",
    version="0.2.0",
)

app.include_router(system.router)
app.include_router(models.router)
app.include_router(generation.router)
app.include_router(sessions.router)
