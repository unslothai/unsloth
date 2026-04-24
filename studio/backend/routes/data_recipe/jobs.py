# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Job lifecycle endpoints for data recipe."""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from urllib.parse import urlparse

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

router = APIRouter()


def _resolve_local_v1_endpoint(request: Request) -> str:
    """Return the loopback /v1 URL for the actual backend listen port.

    Resolution order:
      1. ``app.state.server_port`` - explicitly published by run.py after
         the uvicorn server has bound. This is the most reliable source
         because it survives reverse proxies, TLS terminators and tunnels.
      2. ``request.scope["server"]`` - the real (host, port) tuple uvicorn
         sets when the request is dispatched. Used when Studio is started
         outside ``run_server`` (e.g. ``uvicorn studio.backend.main:app``).
      3. ``request.base_url`` parsed - last resort for test fixtures that
         do not route through a live uvicorn server.
    """
    port: Any = getattr(request.app.state, "server_port", None)
    if not isinstance(port, int) or port <= 0:
        server = request.scope.get("server")
        if (
            isinstance(server, tuple)
            and len(server) >= 2
            and isinstance(server[1], int)
            and server[1] > 0
        ):
            port = server[1]
        else:
            parsed = urlparse(str(request.base_url))
            port = parsed.port if parsed.port is not None else 8888
    return f"http://127.0.0.1:{int(port)}/v1"


def _request_has_desktop_access_token(request: Request) -> bool:
    auth_header = request.headers.get("authorization")
    if not auth_header:
        return False

    parts = auth_header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False

    from auth.authentication import is_desktop_access_token

    return is_desktop_access_token(parts[1])


def _used_llm_model_aliases(recipe: dict[str, Any]) -> set[str]:
    """Return the set of model_aliases that are actually referenced by an
    LLM column. Used to narrow the "Chat model loaded" gate so that orphan
    model_config nodes on the canvas do not block unrelated recipe runs.

    The ``llm-`` prefix matches the existing convention in
    ``core/data_recipe/service.py::_recipe_has_llm_columns`` and covers all
    LLM column types emitted by the frontend (llm-text, llm-code,
    llm-structured, llm-judge).
    """
    aliases: set[str] = set()
    for column in recipe.get("columns", []):
        if not isinstance(column, dict):
            continue
        column_type = column.get("column_type")
        if not isinstance(column_type, str) or not column_type.startswith("llm-"):
            continue
        alias = column.get("model_alias")
        if isinstance(alias, str) and alias:
            aliases.add(alias)
    return aliases


def _inject_local_providers(recipe: dict[str, Any], request: Request) -> None:
    """
    Mutate recipe dict in-place: for any provider with is_local=True,
    generate a JWT and fill in the endpoint pointing at this server.
    """
    providers = recipe.get("model_providers")
    if not providers:
        return

    # Collect local providers and pop is_local from ALL dicts unconditionally.
    # Strict `is True` guard so malformed payloads (is_local: 1,
    # is_local: "true") do not accidentally trigger the loopback rewrite.
    local_indices: list[int] = []
    for i, provider in enumerate(providers):
        if not isinstance(provider, dict):
            continue
        is_local = provider.pop("is_local", None)
        if is_local is True:
            local_indices.append(i)

    if not local_indices:
        return

    endpoint = _resolve_local_v1_endpoint(request)

    # Only gate on model-loaded if a local provider is actually reachable
    # from an LLM column through a model_config. Orphan model_config nodes
    # that reference a local provider but that no LLM column uses should
    # not block runs; the recipe would never call /v1 for them.
    local_names = {
        providers[i].get("name") for i in local_indices if providers[i].get("name")
    }
    used_aliases = _used_llm_model_aliases(recipe)
    referenced_providers = {
        mc.get("provider")
        for mc in recipe.get("model_configs", [])
        if (
            isinstance(mc, dict)
            and mc.get("provider")
            and mc.get("alias") in used_aliases
        )
    }

    token = ""
    if local_names & referenced_providers:
        # Verify a model is loaded.
        # NOTE: This is a point-in-time check (TOCTOU). The model could be unloaded
        # or swapped after this check but before the recipe subprocess calls /v1.
        # The inference endpoint returns a clear 400 in that case.
        #
        # Imports are deferred to avoid circular dependencies with inference modules.
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

        from auth.authentication import (
            create_access_token,
        )  # deferred: avoids circular import

        # Uses the "unsloth" admin subject. If the user changes their password,
        # the JWT secret rotates and this token becomes invalid mid-run.
        # Acceptable for v1 - recipes typically finish well within one session.
        token = create_access_token(
            subject = "unsloth",
            expires_delta = timedelta(hours = 24),
            desktop = _request_has_desktop_access_token(request),
        )

    # Defensively strip any stale "external"-only fields the frontend may
    # have left on the dict (extra_headers/extra_body/api_key_env). The UI
    # hides these inputs in local mode but the payload builder still serializes
    # them, so a previously external provider that flipped to local can carry
    # invalid JSON or rogue auth headers into the local /v1 call.
    #
    # Force ``stream: false`` on the request body via provider extra_body.
    # data_designer's OpenAICompatibleClient merges provider.extra_body into
    # the top-level chat-completion payload (see TransportKwargs.from_request),
    # and our /v1/chat/completions proxy streams by default when ``stream`` is
    # absent. Recipes consume a single JSON response (no SSE parser), so we
    # pin stream=false here for the local loopback path.
    for i in local_indices:
        providers[i]["endpoint"] = endpoint
        providers[i]["api_key"] = token
        providers[i]["provider_type"] = "openai"
        providers[i].pop("api_key_env", None)
        providers[i].pop("extra_headers", None)
        providers[i]["extra_body"] = {"stream": False}

    # Force skip_health_check on any model_config that references a local
    # provider. The local /v1/models endpoint only lists the real loaded
    # model (e.g. "unsloth/llama-3.2-1b") and not the placeholder "local"
    # that the recipe sends as the model id, so data_designer's pre-flight
    # health check would otherwise fail before the first completion call.
    # The backend route ignores the model id field in chat completions, so
    # skipping the check is safe.
    for mc in recipe.get("model_configs", []):
        if not isinstance(mc, dict):
            continue
        if mc.get("provider") in local_names:
            mc["skip_health_check"] = True


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
        _inject_local_providers(recipe, request)
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
