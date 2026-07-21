# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Job lifecycle endpoints for data recipe."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from core.data_recipe.huggingface import (
    RecipeDatasetPublishError,
    publish_recipe_dataset,
)
from core.data_recipe.jobs import get_job_manager
from loggers import get_logger
from models.data_recipe import (
    JobCreateResponse,
    PublishDatasetRequest,
    PublishDatasetResponse,
    RecipePayload,
)
from utils.utils import safe_error_detail, safe_curated_detail, log_and_http_error

logger = get_logger(__name__)
router = APIRouter()


def _resolve_local_v1_endpoint(request: Request) -> str:
    """Return the loopback /v1 URL for the actual backend listen port.

    Resolution order:
      1. ``app.state.server_port`` (run.py, post-bind) - survives proxies/tunnels.
      2. ``request.scope["server"]`` - when Unsloth starts outside ``run_server``.
      3. parsed ``request.base_url`` - last resort for test fixtures.
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
    """Return model_aliases actually referenced by an LLM column.

    Narrows the "Chat model loaded" gate so orphan model_config nodes don't
    block unrelated runs. The ``llm-`` prefix matches
    ``service.py::_recipe_has_llm_columns`` and covers all LLM column types.
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


def _used_local_model_selections(
    recipe: dict[str, Any], local_provider_names: set[str]
) -> dict[tuple[str, str], list[str]]:
    used_aliases = _used_llm_model_aliases(recipe)
    selections: dict[tuple[str, str], list[str]] = {}
    for mc in recipe.get("model_configs", []):
        if not isinstance(mc, dict):
            continue
        alias = mc.get("alias")
        if not isinstance(alias, str) or alias not in used_aliases:
            continue
        provider = mc.get("provider")
        if not isinstance(provider, str) or provider not in local_provider_names:
            continue
        model = mc.get("model")
        target = model.strip() if isinstance(model, str) else ""
        if not target or target.lower() == "local":
            continue
        variant = mc.get("gguf_variant")
        gguf_variant = variant.strip() if isinstance(variant, str) else ""
        selections.setdefault((target, gguf_variant), []).append(alias)
    return selections


def _single_used_local_model_selection(
    recipe: dict[str, Any], local_provider_names: set[str]
) -> tuple[str, str] | None:
    selections = _used_local_model_selections(recipe, local_provider_names)
    if not selections:
        return None
    if len(selections) > 1:
        aliases = ", ".join(alias for values in selections.values() for alias in values)
        raise ValueError(
            "Recipes supports one active local model per run. "
            f"Select the same local model and GGUF variant for: {aliases}."
        )
    return next(iter(selections))


def _loaded_local_model_identity() -> tuple[bool, str, str]:
    from routes.inference import get_llama_cpp_backend
    from core.inference import get_inference_backend

    llama = get_llama_cpp_backend()
    if llama.is_loaded:
        model = str(getattr(llama, "model_identifier", "") or "").strip()
        variant = str(getattr(llama, "hf_variant", "") or "").strip()
        return True, model, variant

    backend = get_inference_backend()
    active_model = str(getattr(backend, "active_model_name", "") or "").strip()
    if active_model:
        return True, active_model, ""
    return False, "", ""


def _ensure_selected_local_model_loaded(
    recipe: dict[str, Any], local_provider_names: set[str]
) -> None:
    model_loaded, active_model, active_variant = _loaded_local_model_identity()
    if not model_loaded:
        raise ValueError(
            "No model loaded in Chat. Load a model first, then run the recipe."
        )

    selection = _single_used_local_model_selection(recipe, local_provider_names)
    if selection is None:
        return

    target, gguf_variant = selection
    variant_matches = not gguf_variant or active_variant == gguf_variant
    if active_model.lower() != target.lower() or not variant_matches:
        selected = f"{target} ({gguf_variant})" if gguf_variant else target
        active = (
            f"{active_model} ({active_variant})" if active_variant else active_model
        )
        raise ValueError(
            "Selected local model is not loaded. "
            f"Selected {selected}; active {active or 'none'}. "
            "Load the selected model again, then run the recipe."
        )


def _inject_local_structured_response_format(
    recipe: dict[str, Any], local_provider_names: set[str]
) -> None:
    """Inject an OpenAI ``response_format`` for each local llm-structured column.

    Clones the model_config and repoints the column at the clone so llm-text /
    llm-judge columns sharing the alias keep free-form sampling. Without this,
    data_designer only adds a prompt-level "return JSON" hint, which small GGUFs
    often break. Forwarding ``response_format`` lets llama-server apply
    grammar-constrained sampling, guaranteeing parseable output.
    """
    columns = recipe.get("columns")
    model_configs = recipe.get("model_configs")
    if not isinstance(columns, list) or not isinstance(model_configs, list):
        return

    # alias -> model_config (only configs referencing a local provider qualify).
    alias_to_local_mc: dict[str, dict[str, Any]] = {}
    for mc in model_configs:
        if not isinstance(mc, dict):
            continue
        if mc.get("provider") in local_provider_names and isinstance(
            mc.get("alias"), str
        ):
            alias_to_local_mc[mc["alias"]] = mc

    if not alias_to_local_mc:
        return

    # Clone per (alias, column) so each llm-structured column gets its own
    # schema without leaking response_format onto other columns sharing the
    # base alias.
    seen_clone_aliases: set[str] = {
        mc.get("alias") for mc in model_configs if isinstance(mc.get("alias"), str)
    }
    new_configs: list[dict[str, Any]] = []
    for column in columns:
        if not isinstance(column, dict):
            continue
        if column.get("column_type") != "llm-structured":
            continue
        alias = column.get("model_alias")
        if not isinstance(alias, str) or alias not in alias_to_local_mc:
            continue
        output_format = column.get("output_format")
        if not isinstance(output_format, dict) or not output_format:
            continue
        base_mc = alias_to_local_mc[alias]
        column_name = column.get("name") or "structured"
        clone_alias_base = f"{alias}__{column_name}_structured"
        clone_alias = clone_alias_base
        counter = 1
        while clone_alias in seen_clone_aliases:
            counter += 1
            clone_alias = f"{clone_alias_base}_{counter}"
        seen_clone_aliases.add(clone_alias)

        clone = copy.deepcopy(base_mc)
        clone["alias"] = clone_alias
        params = clone.get("inference_parameters")
        if not isinstance(params, dict):
            params = {}
            clone["inference_parameters"] = params
        # BaseInferenceParams is extra="forbid", so response_format can't sit at
        # the top level. Its `extra_body` passthrough is spread into the request
        # body top level by the OpenAI client, where llama-server reads
        # response_format. Per tools/server/README.md the schema sits directly
        # under response_format (not nested in a json_schema object as OpenAI
        # expects) and is converted to a GBNF grammar for sampling.
        extra_body = params.get("extra_body")
        if not isinstance(extra_body, dict):
            extra_body = {}
        extra_body["response_format"] = {
            "type": "json_schema",
            "schema": output_format,
        }
        # The OpenAI chat endpoint now returns raw JSON by default for
        # response_format requests (spec compliance for public clients). This
        # internal opt-in flag rides through the OpenAI SDK's extra_body
        # passthrough alongside response_format and re-enables the ```json
        # markdown fence that data_designer's structured-output parser expects.
        extra_body["_unsloth_guided_fence"] = True
        params["extra_body"] = extra_body
        new_configs.append(clone)
        column["model_alias"] = clone_alias

    if new_configs:
        model_configs.extend(new_configs)


def _inject_local_providers(recipe: dict[str, Any], request: Request) -> Optional[int]:
    """Mutate recipe in-place: point is_local providers at this server and mint
    a short-lived internal sk-unsloth-* key for workflow auth.

    Returns the minted key's row id (for the caller to revoke on completion), or
    ``None`` when no local provider is reachable from an LLM column.
    """
    providers = recipe.get("model_providers")
    if not providers:
        return None

    # Collect local providers and pop is_local from ALL dicts. Strict `is True`
    # guard so malformed payloads (1, "true") don't trigger the loopback rewrite.
    local_indices: list[int] = []
    for i, provider in enumerate(providers):
        if not isinstance(provider, dict):
            continue
        is_local = provider.pop("is_local", None)
        if is_local is True:
            local_indices.append(i)

    if not local_indices:
        return None

    endpoint = _resolve_local_v1_endpoint(request)

    # Only gate on model-loaded if a local provider is reachable from an LLM
    # column via a model_config. Orphan model_config nodes shouldn't block runs;
    # the recipe never calls /v1 for them.
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
    internal_key_id: Optional[int] = None
    if local_names & referenced_providers:
        # Verify the selected local model is loaded before minting a key. Still
        # a point-in-time check (TOCTOU); the /v1 endpoint returns a clear 400 if
        # the model is unloaded or swapped before the subprocess calls it.
        _ensure_selected_local_model_loaded(recipe, local_names)

        from auth import storage  # deferred: avoids circular import

        # Mint an internal sk-unsloth-* key scoped to this run via the unified
        # API-key path. Marked internal so it's hidden from the user's key list;
        # the caller revokes it when the job terminates.
        expires_at = (datetime.now(timezone.utc) + timedelta(hours = 24)).isoformat()
        token, row = storage.create_api_key(
            username = "unsloth",
            name = "data-recipe workflow",
            expires_at = expires_at,
            internal = True,
        )
        internal_key_id = int(row["id"])

    # Strip stale "external"-only fields (extra_headers/extra_body/api_key_env)
    # the frontend may have serialized; a provider flipped from external to local
    # could otherwise carry invalid JSON or rogue auth headers into the /v1 call.
    for i in local_indices:
        providers[i]["endpoint"] = endpoint
        providers[i]["api_key"] = token
        providers[i]["provider_type"] = "openai"
        providers[i].pop("api_key_env", None)
        providers[i].pop("extra_headers", None)
        providers[i].pop("extra_body", None)

    # Force skip_health_check on local model_configs. llama-server's /v1/models
    # response can differ from the selected id (cache aliases, GGUF variants),
    # and we already gated on a loaded backend, so the health check would be
    # redundant and could reject valid local selections.
    for mc in recipe.get("model_configs", []):
        if not isinstance(mc, dict):
            continue
        if mc.get("provider") in local_names:
            mc["skip_health_check"] = True
            # Disable thinking for local data-recipe inference. The
            # <think>...</think> preamble roughly doubles tokens per row and
            # pushes answers past data_designer's json-fence regex. Forward
            # chat_template_kwargs={enable_thinking: False} via extra_body so
            # llama-server renders the template without it: llm-text columns get
            # the latency cut, structured columns stop leaking think tags.
            params = mc.get("inference_parameters")
            if not isinstance(params, dict):
                params = {}
                mc["inference_parameters"] = params
            extra_body = params.get("extra_body")
            if not isinstance(extra_body, dict):
                extra_body = {}
            tpl_kwargs = extra_body.get("chat_template_kwargs")
            if not isinstance(tpl_kwargs, dict):
                tpl_kwargs = {}
            tpl_kwargs["enable_thinking"] = False
            extra_body["chat_template_kwargs"] = tpl_kwargs
            params["extra_body"] = extra_body

    # Forward each llm-structured column's output_format as a response_format so
    # llama-server uses grammar-constrained sampling instead of broken JSON.
    _inject_local_structured_response_format(recipe, local_names)

    return internal_key_id


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
            raise log_and_http_error(
                exc,
                400,
                "invalid run_config",
                event = "data_recipe.jobs.run_config_invalid",
                log = logger,
            ) from exc

    try:
        internal_api_key_id = _inject_local_providers(recipe, request)
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            safe_curated_detail(exc),
            event = "data_recipe.jobs.inject_local_providers_failed",
            log = logger,
        ) from exc

    # Single try over get_job_manager() AND mgr.start() so a minted key never
    # outlives the request on an unexpected exception; without the bare except it
    # would live until its 24h TTL.
    try:
        mgr = get_job_manager()
        job_id = mgr.start(
            recipe = recipe,
            run = run,
            internal_api_key_id = internal_api_key_id,
        )
    except RuntimeError as exc:
        if internal_api_key_id is not None:
            _revoke_internal_api_key_safe(internal_api_key_id)
        raise log_and_http_error(
            exc,
            409,
            safe_curated_detail(exc),
            event = "data_recipe.jobs.start_conflict",
            log = logger,
        ) from exc
    except ValueError as exc:
        if internal_api_key_id is not None:
            _revoke_internal_api_key_safe(internal_api_key_id)
        raise log_and_http_error(
            exc,
            400,
            safe_curated_detail(exc),
            event = "data_recipe.jobs.start_failed",
            log = logger,
        ) from exc
    except Exception:
        if internal_api_key_id is not None:
            _revoke_internal_api_key_safe(internal_api_key_id)
        raise

    return {"job_id": job_id}


def _revoke_internal_api_key_safe(key_id: int) -> None:
    """Best-effort revoke of a workflow-minted key; never mask the caller's error."""
    try:
        from auth import storage  # deferred: avoids circular import
        storage.revoke_internal_api_key(key_id)
    except Exception:
        pass


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
        raise log_and_http_error(
            exc,
            400,
            safe_curated_detail(exc),
            event = "data_recipe.jobs.publish_failed",
            log = logger,
        ) from exc
    except Exception as exc:
        raise log_and_http_error(
            exc,
            500,
            safe_error_detail(exc),
            event = "data_recipe.jobs.publish_error",
            log = logger,
        ) from exc

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
