from __future__ import annotations

from fastapi import APIRouter, HTTPException

from llm_local.api_parts.deps import llm
from llm_local.api_parts.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint.

    Returns
    -------
    HealthResponse
        ``{"status": "ok"}`` when the LLM backend is reachable.

    Raises
    ------
    fastapi.HTTPException
        If the LLM backend is unavailable (503).
    """
    if not llm.is_backend_available():
        raise HTTPException(status_code=503, detail="LLM backend not available")
    return HealthResponse(status="ok")
