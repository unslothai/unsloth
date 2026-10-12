# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Answer engine discovery probes with JSON, or with a 404, but never with the app.

Probes for engine endpoints used to land on main.py's SPA catch-all, so ``GET /props`` returned 200 and a page
of HTML. That is worse than a 404: a probe reads the status before the body. Served here: ``/props``,
``/v1/props``, ``/version``, a passthrough to the resident llama-server for ``GET /metrics``, ``GET /slots``,
``POST /tokenize`` and ``POST /detokenize`` (#11705), owner only, and Open WebUI's llama.cpp model management
(``GET /models``, ``POST /models/load``, ``POST /models/unload``, #10610). Everything else in llama-server's
table gets an explicit 404, on its real method as well as GET.

Deliberately NOT served: Ollama's ``/api/tags`` and ``/api/show``. Answering them makes a client select Ollama
and then fail on ``/api/chat``, which Studio does not implement; the reporting user's client instead fell back
to the OpenAI surface and worked. Advertising a protocol we do not have is the HTML 200 again, one layer up.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from loggers import get_logger

logger = get_logger(__name__)

router = APIRouter()


# llama-server's table (tools/server/server.cpp) minus /props, so the set is complete
# rather than growing per complaint. Bare only, so it cannot shadow /api/ or /v1/.
_ENGINE_PROBE_PATHS = frozenset(
    {
        "apply-template",
        "audio/transcriptions",
        "chat/completions",
        "chat/completions/input_tokens",
        "completion",
        "completions",
        "cors-proxy",
        "detokenize",
        "embedding",
        "embeddings",
        "health",
        "infill",
        "lora-adapters",
        "metrics",
        "models",
        "models/load",
        "models/sse",
        "models/unload",
        "rerank",
        "reranking",
        "responses",
        "responses/input_tokens",
        "slots",
        "tokenize",
        "tools",
    }
)

# /slots/:id_slot is the one dynamic entry in that table, and Studio calls it itself.
_ENGINE_PROBE_PREFIXES = ("slots/",)

# Forwarded to the resident llama-server (#11705). Slot save / restore stays denied: Studio owns slot persistence.
_PASSTHROUGH_ROUTES = frozenset(
    {
        ("GET", "metrics"),
        ("GET", "slots"),
        ("POST", "tokenize"),
        ("POST", "detokenize"),
    }
)
_PASSTHROUGH_TIMEOUT_S = 30.0

# Open WebUI's llama.cpp model management (#10610). Download, delete and /models/sse stay denied.
_MANAGEMENT_ROUTES = frozenset(
    {
        ("GET", "models"),
        ("POST", "models/load"),
        ("POST", "models/unload"),
    }
)
_SERVED_ROUTES = _PASSTHROUGH_ROUTES | _MANAGEMENT_ROUTES

# The /v1 entries Studio does not implement. A test asserts each is really absent from
# the assembled app, so implementing one fails CI instead of being shadowed by a 404.
_UNSERVED_V1_PROBE_PATHS = frozenset(
    {
        "v1/chat/completions/control",
        "v1/chat/completions/input_tokens",
        "v1/health",
        "v1/rerank",
        "v1/reranking",
        "v1/responses/input_tokens",
        "v1/stream",
        "v1/streams/lookup",
    }
)


def is_engine_probe_path(full_path: str) -> bool:
    """True for an engine endpoint that must 404 rather than render the app shell."""
    normalized = full_path.strip("/").lower()
    return normalized in _ENGINE_PROBE_PATHS or normalized.startswith(_ENGINE_PROBE_PREFIXES)


def _inference():
    """``routes.inference``, deferred because it pulls the whole inference stack. One
    indirection, so a test replaces this instead of mutating sys.modules."""
    from routes import inference
    return inference


@functools.lru_cache(maxsize = 1)
def _studio_version() -> str:
    """The version string, resolved once: get_studio_version() shells out to git twice
    on a source checkout, and uncached that lands on the event loop per probe."""
    try:
        # Inside the guard: an ImportError here would 500 a probe about something else.
        from utils.studio_version import get_studio_version
        return get_studio_version()
    except Exception:  # noqa: BLE001 -- discovery must not 500 on a version lookup
        return "dev"


def _loaded_public_model_id() -> Optional[str]:
    """The id ``/v1/models`` publishes for whatever is resident, or None."""
    inf = _inference()
    llama_backend = inf.get_llama_cpp_backend()
    if getattr(llama_backend, "is_loaded", False):
        return inf._llama_public_model_id(llama_backend)
    # peek, not get: the orchestrator's cold build waits on hardware detection.
    orchestrator = inf._peek_inference_backend()
    if orchestrator is not None and getattr(orchestrator, "active_model_name", None):
        return inf._orchestrator_public_model_id(orchestrator)
    return None


def _server_props() -> dict:
    """llama-server's /props, with model_path mapped to the public id like every other
    Studio response: upstream reports the absolute .gguf path and this is LAN reachable."""
    llama_backend = _inference().get_llama_cpp_backend()
    public_id = _loaded_public_model_id()
    llama_loaded = bool(getattr(llama_backend, "is_loaded", False))
    props: dict[str, Any] = {}
    if llama_loaded:
        # Documented not to raise, but a probe is the wrong place to find out it does.
        try:
            upstream = llama_backend._query_server_props()
        except Exception:  # noqa: BLE001
            upstream = None
        if isinstance(upstream, dict):
            props = dict(upstream)

    props.pop("model_path", None)
    if public_id:
        props["model_path"] = public_id

    # These describe the CHILD's route table and web UI, not ours: Studio launches with --metrics, so
    # endpoint_metrics arrives true while public /metrics 404s here (tools/server/server-context.cpp puts all three
    # flags in the payload).
    for _child_only in ("ui", "ui_settings", "cors_proxy_enabled"):
        props.pop(_child_only, None)
    props["endpoint_props"] = False
    passthrough = llama_loaded and not _caller_is_managed_account()
    props["endpoint_slots"] = passthrough and props.get("endpoint_slots") is True
    props["endpoint_metrics"] = passthrough and props.get("endpoint_metrics") is True

    # Only when llama-server owns the resident model: with MLX loaded, reading the
    # unloaded llama backend describes a serving model as having no context or slots.
    if llama_loaded:
        props.setdefault("default_generation_settings", {})
        settings = props["default_generation_settings"]
        if isinstance(settings, dict) and "n_ctx" not in settings:
            n_ctx = getattr(llama_backend, "context_length", None)
            if n_ctx:
                settings["n_ctx"] = int(n_ctx)
        if "total_slots" not in props:
            try:
                slots = int(getattr(llama_backend, "effective_parallel_slots", 0) or 0)
            except Exception:  # noqa: BLE001
                slots = 0
            if slots > 0:
                props["total_slots"] = slots
        props.setdefault("chat_template", getattr(llama_backend, "chat_template", "") or "")
    props["build_info"] = f"unsloth-studio/{_studio_version()}"
    return props


# Slash forms too: FastAPI's redirect never fires. The catch-all fully matches "/props/" and returns
# index.html; routes/inference.py registers "/v1/models/" likewise.
@router.get("/props", include_in_schema = False)
@router.get("/props/", include_in_schema = False)
@router.get("/v1/props", include_in_schema = False)
@router.get("/v1/props/", include_in_schema = False)
async def llama_props(current_subject: str = Depends(get_current_subject)):
    """llama-server-compatible ``GET /props``. The body reveals the resident model, so another account gets the same answer ``/api/inference/status`` gives it."""
    if await asyncio.to_thread(_resident_hidden_from_caller):
        from hub.services.models.account_access import hidden_resident_response
        return hidden_resident_response()
    return await asyncio.to_thread(_server_props)


def _resident_hidden_from_caller() -> bool:
    from hub.services.models import account_access

    if not account_access.managed_account():
        return False
    if account_access.resident_hidden("chat"):
        return True
    try:
        slot = _inference()._loaded_slot_ident()
    except Exception:  # noqa: BLE001 - a backend that cannot say has nothing to hide
        return False
    return bool(slot) and account_access.resident_hidden("chat", slot)


def _caller_is_managed_account() -> bool:
    from hub.services.models import account_access
    return account_access.managed_account()


def _upstream_client() -> httpx.AsyncClient:
    # verify=False: plain HTTP to loopback, skips loading the CA bundle per call (~35 ms).
    return httpx.AsyncClient(trust_env = False, verify = False)


def _engine_error(status_code: int, message: str) -> JSONResponse:
    error_type = "unavailable_error" if status_code == 503 else "not_found_error"
    return JSONResponse(
        status_code = status_code,
        content = {"error": {"code": status_code, "message": message, "type": error_type}},
    )


async def _forward_to_llama_server(request: Request, path: str) -> Response:
    # Owner only: /slots and /metrics are server-wide, not scoped per account.
    if await asyncio.to_thread(_caller_is_managed_account):
        return _engine_error(404, "API endpoint not found")
    llama_backend = _inference().get_llama_cpp_backend()
    # Unload flips is_loaded before clearing _port; "127.0.0.1:None" raises InvalidURL, not HTTPError.
    port = getattr(llama_backend, "_port", None)
    if not getattr(llama_backend, "is_loaded", False) or not port:
        return _engine_error(503, "No llama.cpp model is loaded")
    url = f"http://127.0.0.1:{port}/{path}"
    headers = dict(llama_backend._auth_headers or {})
    body = await request.body() if request.method == "POST" else None
    if body is not None:
        headers["Content-Type"] = request.headers.get("content-type", "application/json")
    try:
        async with _upstream_client() as client:
            upstream = await client.request(
                request.method,
                url,
                params = request.query_params,
                content = body,
                headers = headers,
                timeout = _PASSTHROUGH_TIMEOUT_S,
            )
    except httpx.HTTPError as exc:
        logger.info("llama-server passthrough %s /%s failed: %s", request.method, path, exc)
        return _engine_error(503, "llama.cpp server is not reachable")
    return Response(
        content = upstream.content,
        status_code = upstream.status_code,
        media_type = upstream.headers.get("content-type"),
    )


def _make_passthrough(path: str):
    async def passthrough(request: Request, current_subject: str = Depends(get_current_subject)):
        return await _forward_to_llama_server(request, path)

    passthrough.__name__ = f"llama_passthrough_{path}"
    return passthrough


for _method, _path in sorted(_PASSTHROUGH_ROUTES):
    for _form in (f"/{_path}", f"/{_path}/"):
        router.add_api_route(
            _form,
            _make_passthrough(_path),
            methods = [_method],
            include_in_schema = False,
        )


@router.get("/version", include_in_schema = False)
@router.get("/version/", include_in_schema = False)
async def studio_version(current_subject: str = Depends(get_current_subject)):
    """Bare /version only: Ollama spells it /api/version, and answering there is part
    of claiming to be Ollama."""
    # Threaded like /props: the first call resolves the version.
    return {"version": await asyncio.to_thread(_studio_version)}


class _ModelManageBody(BaseModel):
    model: str = Field(..., min_length = 1)


def _catalog_match(catalog: list[dict], model_id: str) -> Optional[tuple[str, dict, bool]]:
    """``(id, row, pinned)``: the catalog's own id, or ``model_id`` as ``<listed id>:<quant>`` (pinned)."""
    wanted = model_id.strip()
    rows = {e["id"].lower(): e for e in catalog if isinstance(e.get("id"), str)}
    if wanted.lower() in rows:
        row = rows[wanted.lower()]
        return row["id"], row, False
    base, sep, _ = wanted.rpartition(":")
    row = rows.get(base.strip().lower()) if sep else None
    return (wanted, row, True) if row is not None else None


async def _chat_catalog() -> list[dict]:
    # A task row (image, video, speech) loads through its own engine, not /load and /unload.
    rows = await _inference()._openai_catalog_objects(quant_pins = True)
    return [e for e in rows if e.get("task") is None]


async def _catalog_row(model_id: str) -> tuple[str, dict, bool]:
    # The caller's own catalog, so a managed account only reaches models listed to it.
    matched = _catalog_match(await _chat_catalog(), model_id)
    if matched is None:
        raise HTTPException(status_code = 404, detail = "Model not found")
    return matched


async def _resolve_local(public_id: str) -> Optional[tuple]:
    from core.inference.local_model_resolver import (
        resolve_local_gguf,
        resolve_trusted_cached_local_gguf,
    )
    return resolve_trusted_cached_local_gguf(public_id) or await asyncio.to_thread(
        resolve_local_gguf, public_id
    )


def _same_path(a: Optional[str], b: Optional[str]) -> bool:
    import os
    return bool(a and b) and os.path.normpath(a) == os.path.normpath(b)


async def _unload_target(model_id: str, row: dict) -> Optional[str]:
    """What /unload must be named to stop this row, or None when nothing of it runs.

    /unload matches the resident or loading identifier, not an alias (LM Studio, scan folder,
    Ollama tag) nor a path-loaded row's listed id, and on a managed install it releases the
    caller's shared resident before comparing names, so an unrelated row must never reach it.
    """
    from core.inference.openai_auto_download import looks_like_quant, split_model_ref

    inf = _inference()
    resolved = await _resolve_local(model_id)
    path = resolved[0] if resolved else None
    base, suffix = split_model_ref(model_id)
    # Only a quant the caller named: a bare id resolves to the preferred quant, not the resident one.
    wanted_variant = suffix if suffix and looks_like_quant(suffix) else None

    llama = inf.get_llama_cpp_backend()
    orchestrator = inf._peek_inference_backend()
    if getattr(llama, "is_loaded", False) or getattr(llama, "is_active", False):
        candidates = [(llama, llama.model_identifier, getattr(llama, "hf_variant", None))]
    else:
        # Loading and active apart: a switch keeps the old model active while the new one loads.
        loading = getattr(orchestrator, "get_loading_model", lambda: None)()
        active = getattr(orchestrator, "active_model_name", None)
        candidates = [(orchestrator, loading, None), (orchestrator, active, None)]
    for backend, running, running_variant in candidates:
        if not running:
            continue
        advertised = getattr(backend, "_openai_advertised_id", None)
        # The full id first: an Ollama tag such as "llama3:latest" is the advertised id, not a quant pin.
        named = _same_path(running, path) or any(
            advertised and name.strip().lower() == advertised.strip().lower()
            for name in (model_id, base)
        )
        if not named:
            continue
        if wanted_variant and running_variant and wanted_variant.lower() != running_variant.lower():
            return None
        return running
    filling = inf.model_slots.visible_loading()
    if filling and _same_path(filling[1], path):
        return path
    # A kept slot routes by the listed id inside /unload.
    return model_id if row.get("loaded") else None


async def _resolve_load_target(model_id: str):
    """``(LoadRequest, listed id)``, from the on-disk copy and saved launch settings auto-switch would use."""
    from core.inference.local_model_resolver import local_target_is_gguf
    from core.inference.openai_auto_download import split_model_ref
    from models.inference import LoadRequest
    from utils.openai_auto_switch_settings import (
        model_override_load_kwargs,
        resolve_override_for_load,
        stored_gpu_index_kind,
    )

    public_id, _, pinned = await _catalog_row(model_id)
    resolved = await _resolve_local(public_id)
    if resolved is None:
        if pinned:
            # A quant this machine does not have: no remote download from a catalog-gated route.
            raise HTTPException(status_code = 404, detail = "Model not found")
        # The row's own id, colon and all (an NPU "lemonade:..." id is not a quant pin).
        return LoadRequest(model_path = public_id), public_id
    model_path, variant, loader_id = resolved[:3]
    alias = loader_id or split_model_ref(public_id)[0]
    _, override = resolve_override_for_load(model_path, alias, variant)
    # Not from the variant: a single-file GGUF or an Ollama tag resolves with no quant to pin.
    is_gguf = await asyncio.to_thread(local_target_is_gguf, model_path, alias)
    kwargs = model_override_load_kwargs(override, is_gguf = is_gguf)
    # A stale pin (GPU removed or renumbered) is dropped, as auto-switch does, rather than 400 the load.
    if kwargs.get("gpu_ids") and not await _inference()._override_gpu_ids_still_resolve(
        kwargs["gpu_ids"], stored_gpu_index_kind(override)
    ):
        kwargs.pop("gpu_ids", None)
        kwargs.pop("tensor_split", None)
    request = LoadRequest(model_path = model_path, gguf_variant = variant, **kwargs)
    request._override_alias_id = alias
    request._advertised_id = alias
    return request, public_id


async def _lifecycle_answer(work, model_id: str, *, label: str):
    """Studio's /load and /unload wrapping (GPU-busy 409, tunnel padding), answered as llama-server does."""
    from hub.services.models.account_access import gpu_busy_route

    async def _answer():
        await work
        # The listed id, never the resident's path; a padded reply serializes this after the stream starts.
        return {"success": True, "model": model_id}

    @gpu_busy_route
    async def _padded():
        return await _inference()._tunnel_safe_json(_answer(), label = label)

    return await _padded()


# Outside /v1: Open WebUI strips it for management calls (MODEL_MANAGEMENT_ENDPOINTS).
@router.get("/models", include_in_schema = False)
@router.get("/models/", include_in_schema = False)
async def manage_list_models(current_subject: str = Depends(get_current_subject)):
    return {"object": "list", "data": await _chat_catalog()}


@router.post("/models/load", include_in_schema = False)
@router.post("/models/load/", include_in_schema = False)
async def manage_load_model(
    body: _ModelManageBody,
    fastapi_request: Request,
    current_subject: str = Depends(get_current_subject),
):
    request, public_id = await _resolve_load_target(body.model)
    work = _inference().load_model_gated(
        request, fastapi_request, current_subject, user_initiated = True
    )
    return await _lifecycle_answer(work, public_id, label = "Model load")


@router.post("/models/unload", include_in_schema = False)
@router.post("/models/unload/", include_in_schema = False)
async def manage_unload_model(
    body: _ModelManageBody, current_subject: str = Depends(get_current_subject)
):
    from models.inference import UnloadRequest

    model_id, row, pinned = await _catalog_row(body.model)
    if pinned and await _resolve_local(model_id) is None:
        # A quant this machine does not have names nothing running; never the bare model.
        raise HTTPException(status_code = 404, detail = "Model not found")
    target = await _unload_target(model_id, row)
    if target is None:
        return {"success": True, "model": model_id}
    request = UnloadRequest(model_path = target)
    work = _inference()._unload_model_impl(request, current_subject)
    return await _lifecycle_answer(work, model_id, label = "Model unload")


async def _probe_not_found():
    raise HTTPException(status_code = 404, detail = "API endpoint not found")


# Without these a POST hit the GET-only catch-all and returned 405, reading as "exists, wrong method". HEAD is
# here because Starlette does not admit it on a GET route (measured, fastapi 0.141.1); GET stays with the catch-all
# so its asset lookup wins, and OPTIONS is untouched for CORS preflight.
_PROBE_DENIED_METHODS = ["HEAD", "POST", "PUT", "PATCH", "DELETE"]


# Both forms of every path: no redirect rescues "POST /completion/"
def _both_forms(path: str) -> tuple:
    return (path, path + "/")


def _denied_methods(path: str) -> list:
    return [m for m in _PROBE_DENIED_METHODS if (m, path) not in _SERVED_ROUTES]


for _probe_path in sorted(_ENGINE_PROBE_PATHS):
    for _form in _both_forms(f"/{_probe_path}"):
        router.add_api_route(
            _form,
            _probe_not_found,
            methods = _denied_methods(_probe_path),
            include_in_schema = False,
        )

# main.py already 404s an unknown GET under /v1/, so only the other methods need these.
for _v1_path in sorted(_UNSERVED_V1_PROBE_PATHS):
    for _form in _both_forms(f"/{_v1_path}"):
        router.add_api_route(
            _form,
            _probe_not_found,
            methods = _PROBE_DENIED_METHODS,
            include_in_schema = False,
        )

for _slots_form in _both_forms("/slots/{id_slot}"):
    router.add_api_route(
        _slots_form,
        _probe_not_found,
        methods = _PROBE_DENIED_METHODS,
        include_in_schema = False,
    )


def add_get_denials(app) -> None:
    """404 the engine paths on GET as well, for an app with no frontend mounted. The GET denial normally comes
    from main.py's SPA catch-all, registered only when setup_frontend() finds a build. In API-only mode there is
    none, so these paths matched on method alone and answered 405, which a client reads as "endpoint exists".
    """
    for path in sorted(_ENGINE_PROBE_PATHS) + sorted(_UNSERVED_V1_PROBE_PATHS):
        if ("GET", path) in _SERVED_ROUTES:
            continue
        for form in _both_forms(f"/{path}"):
            app.add_api_route(form, _probe_not_found, methods = ["GET"], include_in_schema = False)
    for form in _both_forms("/slots/{id_slot}"):
        app.add_api_route(form, _probe_not_found, methods = ["GET"], include_in_schema = False)
