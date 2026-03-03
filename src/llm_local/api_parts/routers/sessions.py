from __future__ import annotations

from collections.abc import Iterator
import time

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from llm_local.api_parts.deps import SESSIONS, get_session, llm, new_session_id
from llm_local.api_parts.schemas import (
    ChatHistoryResponse,
    ChatSession,
    CreateSessionRequest,
    CreateSessionResponse,
    DeleteSessionResponse,
    Message,
    Role,
    SessionInfoResponse,
    SessionMessageRequest,
    SessionMessageResponse,
)
from llm_local.api_parts.sse import sse

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest) -> CreateSessionResponse:
    """Create a new in-memory chat session.

    Parameters
    ----------
    req
        Session configuration (optional model override, optional system prompt).

    Returns
    -------
    CreateSessionResponse
        Newly created session metadata.
    """
    session_id = new_session_id()
    created_at = time.time()
    model = req.model or llm.config.model

    messages: list[Message] = []
    if req.system_prompt:
        messages.append(Message(role=Role.system, content=req.system_prompt))

    SESSIONS[session_id] = ChatSession(
        session_id=session_id,
        created_at=created_at,
        model=model,
        system_prompt=req.system_prompt,
        messages=messages,
    )

    return CreateSessionResponse(session_id=session_id, model=model, created_at=created_at)


@router.get("/{session_id}", response_model=SessionInfoResponse)
def get_session_info(session_id: str) -> SessionInfoResponse:
    """Get session metadata.

    Parameters
    ----------
    session_id
        Session identifier.

    Returns
    -------
    SessionInfoResponse
        Session metadata including message count.

    Raises
    ------
    fastapi.HTTPException
        If the session does not exist (404).
    """
    sess = get_session(session_id)
    return SessionInfoResponse(
        session_id=sess.session_id,
        model=sess.model,
        system_prompt=sess.system_prompt,
        created_at=sess.created_at,
        message_count=len(sess.messages),
    )


@router.get("/{session_id}/history", response_model=ChatHistoryResponse)
def get_session_history(session_id: str) -> ChatHistoryResponse:
    """Get full message history for a session.

    Parameters
    ----------
    session_id
        Session identifier.

    Returns
    -------
    ChatHistoryResponse
        Stored messages in chronological order.

    Raises
    ------
    fastapi.HTTPException
        If the session does not exist (404).
    """
    sess = get_session(session_id)
    return ChatHistoryResponse(history=list(sess.messages))


@router.delete("/{session_id}", response_model=DeleteSessionResponse)
def delete_session(session_id: str) -> DeleteSessionResponse:
    """Delete a session.

    Parameters
    ----------
    session_id
        Session identifier.

    Returns
    -------
    DeleteSessionResponse
        ``{"status": "ok"}`` on success.

    Raises
    ------
    fastapi.HTTPException
        If the session does not exist (404).
    """
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    del SESSIONS[session_id]
    return DeleteSessionResponse(status="ok")


@router.post("/{session_id}/messages", response_model=SessionMessageResponse)
def send_session_message(session_id: str, req: SessionMessageRequest) -> SessionMessageResponse:
    """Append a user message and return an assistant reply (non-streaming).

    Parameters
    ----------
    session_id
        Session identifier.
    req
        Message text and generation options.

    Returns
    -------
    SessionMessageResponse
        Assistant reply text.

    Raises
    ------
    fastapi.HTTPException
        If the session does not exist (404) or backend errors (500).
    """
    sess = get_session(session_id)
    sess.messages.append(Message(role=Role.user, content=req.message))

    try:
        reply = llm.chat(
            messages=[m.to_backend() for m in sess.messages],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            options=req.options,
            model_override=sess.model,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sess.messages.append(Message(role=Role.assistant, content=reply))
    return SessionMessageResponse(response=reply)


@router.post("/{session_id}/messages/stream")
def send_session_message_stream(
    session_id: str,
    req: SessionMessageRequest,
    include_assistant_message: bool = Query(
        True,
        description="If true, the assistant message is appended to the session on completion.",
    ),
) -> StreamingResponse:
    """Append a user message and stream assistant output as SSE.

    Parameters
    ----------
    session_id
        Session identifier.
    req
        Message text and generation options.
    include_assistant_message
        Whether to append the final assistant message to session history.

    Returns
    -------
    fastapi.responses.StreamingResponse
        SSE stream emitting events: ``meta``, repeated ``delta``,
        then ``done`` (or ``error``).

    Notes
    -----
    The user message is stored in the session immediately, before streaming begins.
    """
    sess = get_session(session_id)
    sess.messages.append(Message(role=Role.user, content=req.message))

    def iterator() -> Iterator[str]:
        start = time.time()
        yield sse("meta", {"type": "session_chat", "session_id": session_id, "started_at": start})

        assistant_accum: list[str] = []
        try:
            for delta in llm.chat_stream(
                messages=[m.to_backend() for m in sess.messages],
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                options=req.options,
                model_override=sess.model,
            ):
                if delta:
                    assistant_accum.append(delta)
                    yield sse("delta", {"content": delta})

            final_text = "".join(assistant_accum).strip()

            if include_assistant_message:
                sess.messages.append(Message(role=Role.assistant, content=final_text))

            yield sse(
                "done",
                {"ok": True, "elapsed_s": time.time() - start, "final_length": len(final_text)},
            )
        except RuntimeError as exc:
            yield sse("error", {"ok": False, "error": str(exc)})

    return StreamingResponse(iterator(), media_type="text/event-stream")
