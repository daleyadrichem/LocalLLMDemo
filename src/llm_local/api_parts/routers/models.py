from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path

from llm_local.api_parts.deps import llm
from llm_local.api_parts.schemas import (
    ModelDeleteResponse,
    ModelPullRequest,
    ModelPullResponse,
    ModelShowResponse,
    ModelsResponse,
)

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    """List locally available models on the backend.

    Returns
    -------
    ModelsResponse
        List of model identifiers.

    Raises
    ------
    fastapi.HTTPException
        If the backend errors (500).
    """
    try:
        return ModelsResponse(models=llm.list_models())
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/models/pull", response_model=ModelPullResponse)
def pull_model(req: ModelPullRequest) -> ModelPullResponse:
    """Pull (download) a model by name.

    Parameters
    ----------
    req
        Request with model name.

    Returns
    -------
    ModelPullResponse
        ``{"status": "ok"}`` if the pull is started/completed successfully.

    Raises
    ------
    fastapi.HTTPException
        If the backend errors (500).
    """
    try:
        llm.pull_model(req.name)
        return ModelPullResponse(status="ok")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/models/{name}", response_model=ModelDeleteResponse)
def delete_model(
    name: str = Path(..., description="Model name, e.g. llama3.2:3b"),
) -> ModelDeleteResponse:
    """Delete a local model by name.

    Parameters
    ----------
    name
        Model identifier.

    Returns
    -------
    ModelDeleteResponse
        ``{"status": "ok"}`` on success.

    Raises
    ------
    fastapi.HTTPException
        If the backend errors (500).
    """
    try:
        llm.delete_model(name)
        return ModelDeleteResponse(status="ok")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/models/{name}", response_model=ModelShowResponse)
def show_model(
    name: str = Path(..., description="Model name, e.g. llama3.2:3b"),
) -> ModelShowResponse:
    """Show backend-provided metadata for a model.

    Parameters
    ----------
    name
        Model identifier.

    Returns
    -------
    ModelShowResponse
        Model name and backend-provided details.

    Raises
    ------
    fastapi.HTTPException
        If the backend errors (500).
    """
    try:
        details = llm.show_model(name)
        return ModelShowResponse(model=name, details=details)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
