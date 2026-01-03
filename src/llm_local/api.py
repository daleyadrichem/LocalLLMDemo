from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llm_local import LocalLLM, LocalLLMConfig


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://ollama:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")

llm = LocalLLM(
    config=LocalLLMConfig(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
    )
)


# ---------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str


class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    options: Optional[Dict[str, Any]] = None


class GenerateResponse(BaseModel):
    response: str


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    options: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str


class PersistentChatRequest(BaseModel):
    message: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    options: Optional[Dict[str, Any]] = None


class ModelsResponse(BaseModel):
    models: List[str]


class ChatHistoryResponse(BaseModel):
    history: List[Dict[str, str]]


# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------

app = FastAPI(
    title="Local LLM API",
    description="HTTP API wrapper for a local LLM backend (e.g. Ollama)",
    version="0.1.0",
)


# ---------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    if not llm.is_backend_available():
        raise HTTPException(status_code=503, detail="LLM backend not available")
    return HealthResponse(status="ok")


@app.get("/models", response_model=ModelsResponse, tags=["system"])
def list_models() -> ModelsResponse:
    try:
        return ModelsResponse(models=llm.list_models())
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------
# Generation endpoints
# ---------------------------------------------------------------------

@app.post("/generate", response_model=GenerateResponse, tags=["generation"])
def generate(req: GenerateRequest) -> GenerateResponse:
    try:
        text = llm.generate(
            prompt=req.prompt,
            system_prompt=req.system_prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            options=req.options,
        )
        return GenerateResponse(response=text)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse, tags=["generation"])
def chat(req: ChatRequest) -> ChatResponse:
    try:
        text = llm.chat(
            messages=req.messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            options=req.options,
        )
        return ChatResponse(response=text)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------
# Persistent chat endpoints
# ---------------------------------------------------------------------

@app.post("/chat/start", tags=["persistent-chat"])
def start_chat(system_prompt: Optional[str] = None) -> dict:
    llm.start_chat(system_prompt=system_prompt)
    return {"status": "chat started"}


@app.post("/chat/send", response_model=ChatResponse, tags=["persistent-chat"])
def send_chat(req: PersistentChatRequest) -> ChatResponse:
    try:
        reply = llm.send_chat_message(
            user_message=req.message,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            options=req.options,
        )
        return ChatResponse(response=reply)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/chat/history", response_model=ChatHistoryResponse, tags=["persistent-chat"])
def get_history() -> ChatHistoryResponse:
    return ChatHistoryResponse(history=llm.get_history())


@app.post("/chat/reset", tags=["persistent-chat"])
def reset_chat() -> dict:
    llm.reset_chat()
    return {"status": "chat reset"}
