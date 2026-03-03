from __future__ import annotations

from collections.abc import Iterator
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from llm_local.api_parts.deps import llm
from llm_local.api_parts.schemas import ChatRequest, ChatResponse, GenerateRequest, GenerateResponse
from llm_local.api_parts.sse import sse

router = APIRouter(tags=["generation"])


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate a completion for a single prompt (non-streaming).

    Parameters
    ----------
    req
        Prompt and generation options.

    Returns
    -------
    GenerateResponse
        Generated text response.

    Raises
    ------
    fastapi.HTTPException
        If the backend errors (500).
    """
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


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Chat completion given an explicit message list (non-streaming).

    Parameters
    ----------
    req
        Message list and generation options.

    Returns
    -------
    ChatResponse
        Assistant response text.

    Raises
    ------
    fastapi.HTTPException
        If the backend errors (500).
    """
    backend_messages = [m.to_backend() for m in req.messages]
    try:
        text = llm.chat(
            messages=backend_messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            options=req.options,
        )
        return ChatResponse(response=text)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/generate/stream")
def generate_stream(req: GenerateRequest) -> StreamingResponse:
    """Stream a completion as Server-Sent Events (SSE).

    Parameters
    ----------
    req
        Prompt and generation options.

    Returns
    -------
    fastapi.responses.StreamingResponse
        SSE stream emitting events: ``meta``, repeated ``delta``,
        then ``done`` (or ``error``).
    """

    def iterator() -> Iterator[str]:
        start = time.time()
        yield sse("meta", {"type": "generate", "started_at": start})

        try:
            for delta in llm.generate_stream(
                prompt=req.prompt,
                system_prompt=req.system_prompt,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                options=req.options,
            ):
                if delta:
                    yield sse("delta", {"content": delta})

            yield sse("done", {"ok": True, "elapsed_s": time.time() - start})
        except RuntimeError as exc:
            yield sse("error", {"ok": False, "error": str(exc)})

    return StreamingResponse(iterator(), media_type="text/event-stream")


@router.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Stream a chat completion as Server-Sent Events (SSE).

    Parameters
    ----------
    req
        Message list and generation options.

    Returns
    -------
    fastapi.responses.StreamingResponse
        SSE stream emitting events: ``meta``, repeated ``delta``,
        then ``done`` (or ``error``).
    """

    backend_messages = [m.to_backend() for m in req.messages]

    def iterator() -> Iterator[str]:
        start = time.time()
        yield sse("meta", {"type": "chat", "started_at": start})

        try:
            for delta in llm.chat_stream(
                messages=backend_messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                options=req.options,
            ):
                if delta:
                    yield sse("delta", {"content": delta})

            yield sse("done", {"ok": True, "elapsed_s": time.time() - start})
        except RuntimeError as exc:
            yield sse("error", {"ok": False, "error": str(exc)})

    return StreamingResponse(iterator(), media_type="text/event-stream")
