# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Job lifecycle endpoints for data recipe."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from core.data_recipe.huggingface import (
    RecipeDatasetPublishError,
    publish_recipe_dataset,
)
from core.data_recipe.jobs import get_job_manager
from models.data_recipe import (
    JobCreateResponse,
    PublishDatasetRequest,
    PublishDatasetResponse,
    RecipePayload,
)

from datetime import timedelta

router = APIRouter()


def _inject_local_providers(recipe: dict[str, Any], request_base_url: str) -> None:
    """
    Mutate recipe dict in-place: for any provider with is_local=True,
    generate a 24h JWT and fill in the endpoint pointing at this server.
    """
    providers = recipe.get("model_providers")
    if not providers:
        return

    # Collect local providers and pop is_local from ALL dicts unconditionally
    local_indices: list[int] = []
    for i, provider in enumerate(providers):
        if not isinstance(provider, dict):
            continue
        if provider.pop("is_local", None):
            local_indices.append(i)

    if not local_indices:
        return

    # Verify a model is loaded
    from routes.inference import get_llama_cpp_backend
    from core.inference import get_inference_backend

    llama = get_llama_cpp_backend()
    model_loaded = llama.is_loaded
    if not model_loaded:
        backend = get_inference_backend()
        model_loaded = bool(backend.active_model_name)
    if not model_loaded:
        raise ValueError(
            "No model loaded in Chat. Load a model first, then run the recipe."
        )

    from auth.authentication import create_access_token

    token = create_access_token(
        subject="unsloth",
        expires_delta=timedelta(hours=24),
    )

    # Always use loopback — request.base_url may reflect a proxy or 0.0.0.0
    from urllib.parse import urlparse
    parsed = urlparse(request_base_url)
    port = parsed.port or 8888
    endpoint = f"http://127.0.0.1:{port}/v1"

    for i in local_indices:
        providers[i]["endpoint"] = endpoint
        providers[i]["api_key"] = token
        providers[i]["provider_type"] = "openai"


def _normalize_run_name(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPException(
            status_code = 400, detail = "invalid run_name: must be a string"
        )
    trimmed = value.strip()
    if not trimmed:
        return None
    return trimmed[:120]


@router.post("/jobs", response_class = JSONResponse, response_model = JobCreateResponse)
def create_job(payload: RecipePayload, request: Request):
    recipe = payload.recipe
    if not recipe.get("columns"):
        raise HTTPException(status_code = 400, detail = "Recipe must include columns.")

    run: dict[str, Any] = payload.run or {}
    run.pop("artifact_path", None)
    run.pop("dataset_name", None)
    execution_type = str(run.get("execution_type") or "full").strip().lower()
    if execution_type not in {"preview", "full"}:
        raise HTTPException(
            status_code = 400,
            detail = "invalid execution_type: must be 'preview' or 'full'",
        )
    run["execution_type"] = execution_type
    run["run_name"] = _normalize_run_name(run.get("run_name"))
    run_config_raw = run.get("run_config")
    if run_config_raw is not None:
        try:
            from data_designer.config.run_config import RunConfig

            RunConfig.model_validate(run_config_raw)
        except (ImportError, ValidationError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code = 400, detail = f"invalid run_config: {exc}"
            ) from exc

    try:
        _inject_local_providers(recipe, str(request.base_url))
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc

    mgr = get_job_manager()
    try:
        job_id = mgr.start(recipe = recipe, run = run)
    except RuntimeError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc

    return {"job_id": job_id}


@router.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    mgr = get_job_manager()
    state = mgr.get_status(job_id)
    if state is None:
        raise HTTPException(status_code = 404, detail = "job not found")
    return state


@router.get("/jobs/current")
def current_job():
    mgr = get_job_manager()
    state = mgr.get_current_status()
    if state is None:
        raise HTTPException(status_code = 404, detail = "no job")
    return state


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    mgr = get_job_manager()
    ok = mgr.cancel(job_id)
    if not ok:
        raise HTTPException(status_code = 404, detail = "job not found")
    return mgr.get_status(job_id)


@router.get("/jobs/{job_id}/analysis")
def job_analysis(job_id: str):
    mgr = get_job_manager()
    analysis = mgr.get_analysis(job_id)
    if analysis is None:
        raise HTTPException(status_code = 404, detail = "analysis not ready")
    return analysis


@router.get("/jobs/{job_id}/dataset")
def job_dataset(
    job_id: str,
    limit: int = Query(default = 20, ge = 1, le = 500),
    offset: int = Query(default = 0, ge = 0),
):
    mgr = get_job_manager()
    result = mgr.get_dataset(job_id, limit = limit, offset = offset)
    if result is None:
        raise HTTPException(status_code = 404, detail = "dataset not ready")
    if "error" in result:
        raise HTTPException(status_code = 422, detail = result["error"])
    return {
        "dataset": result["dataset"],
        "total": result["total"],
        "limit": limit,
        "offset": offset,
    }


@router.post(
    "/jobs/{job_id}/publish",
    response_class = JSONResponse,
    response_model = PublishDatasetResponse,
)
def publish_job_dataset(job_id: str, payload: PublishDatasetRequest):
    repo_id = payload.repo_id.strip()
    description = payload.description.strip()
    hf_token = payload.hf_token.strip() if isinstance(payload.hf_token, str) else None
    artifact_path = (
        payload.artifact_path.strip()
        if isinstance(payload.artifact_path, str)
        else None
    )

    if not repo_id:
        raise HTTPException(status_code = 400, detail = "repo_id is required")
    if not description:
        raise HTTPException(status_code = 400, detail = "description is required")

    mgr = get_job_manager()
    status = mgr.get_status(job_id)
    if status is not None:
        if (
            status.get("status") != "completed"
            or status.get("execution_type") != "full"
        ):
            raise HTTPException(
                status_code = 409,
                detail = "Only completed full runs can be published.",
            )
        status_artifact = status.get("artifact_path")
        if isinstance(status_artifact, str) and status_artifact.strip():
            artifact_path = status_artifact.strip()

    if not artifact_path:
        raise HTTPException(
            status_code = 400,
            detail = "This execution does not have publishable dataset artifacts.",
        )

    try:
        url = publish_recipe_dataset(
            artifact_path = artifact_path,
            repo_id = repo_id,
            description = description,
            hf_token = hf_token or None,
            private = payload.private,
        )
    except RecipeDatasetPublishError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code = 500, detail = str(exc)) from exc

    return {
        "success": True,
        "url": url,
        "message": f"Published dataset to {repo_id}.",
    }


@router.get("/jobs/{job_id}/events")
async def job_events(request: Request, job_id: str):
    mgr = get_job_manager()
    last_id = request.headers.get("last-event-id")
    after_seq: int | None = None
    if last_id:
        try:
            after_seq = int(str(last_id).strip())
        except (TypeError, ValueError):
            after_seq = None

    after_q = request.query_params.get("after")
    if after_q:
        try:
            after_seq = int(str(after_q).strip())
        except (TypeError, ValueError):
            pass

    sub = mgr.subscribe(job_id, after_seq = after_seq)
    if sub is None:
        raise HTTPException(status_code = 404, detail = "job not found")

    async def gen():
        try:
            for event in sub.replay:
                yield sub.format_sse(event)

            while True:
                if await request.is_disconnected():
                    break
                event = await sub.next_event(timeout_sec = 1.0)
                if event is None:
                    continue
                yield sub.format_sse(event)
        finally:
            mgr.unsubscribe(sub)

    return StreamingResponse(gen(), media_type = "text/event-stream")
