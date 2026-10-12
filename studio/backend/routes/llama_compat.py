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


def _catalog_id_match(catalog: list[dict], model_id: str) -> Optional[str]:
    """The catalog's id for ``model_id``, or ``model_id`` itself when it is ``<listed id>:<quant>``."""
    wanted = model_id.strip()
    ids = {e["id"].lower(): e["id"] for e in catalog if isinstance(e.get("id"), str)}
    if wanted.lower() in ids:
        return ids[wanted.lower()]
    base, sep, _ = wanted.rpartition(":")
    return wanted if sep and base.strip().lower() in ids else None


async def _chat_catalog() -> list[dict]:
    # A task row (image, video, speech) loads through its own engine, not /load and /unload.
    return [e for e in await _inference()._openai_catalog_objects() if e.get("task") is None]


async def _catalog_model_id(model_id: str) -> str:
    # The caller's own catalog, so a managed account only reaches models listed to it.
    matched = _catalog_id_match(await _chat_catalog(), model_id)
    if matched is None:
        raise HTTPException(status_code = 404, detail = "Model not found")
    return matched


def _resident_identifier(model_id: str) -> str:
    """The primary resident's concrete identifier when ``model_id`` is the alias it advertises.

    /unload matches the identifier and the id derived from it, not an alias such as an LM Studio
    or scan-folder name. Kept slots route by alias inside /unload already.
    """
    inf = _inference()
    from core.inference.openai_auto_download import split_model_ref

    base = split_model_ref(model_id)[0]
    backend = inf.get_llama_cpp_backend()
    if getattr(backend, "is_loaded", False):
        resident = getattr(backend, "model_identifier", None)
    else:
        backend = inf._peek_inference_backend()
        resident = getattr(backend, "active_model_name", None)
    advertised = getattr(backend, "_openai_advertised_id", None)
    if resident and advertised and base.strip().lower() == advertised.strip().lower():
        return resident
    return model_id


async def _resolve_load_target(model_id: str):
    """A ``LoadRequest`` for /load, from the on-disk copy auto-switch would use when there is one."""
    from core.inference.local_model_resolver import (
        resolve_local_gguf,
        resolve_trusted_cached_local_gguf,
    )
    from core.inference.openai_auto_download import split_model_ref
    from models.inference import LoadRequest

    public_id = await _catalog_model_id(model_id)
    resolved = resolve_trusted_cached_local_gguf(public_id) or await asyncio.to_thread(
        resolve_local_gguf, public_id
    )
    if resolved is None:
        model_path, variant = split_model_ref(public_id)
        return LoadRequest(model_path = model_path, gguf_variant = variant)
    model_path, variant, loader_id = resolved[:3]
    request = LoadRequest(model_path = model_path, gguf_variant = variant)
    # As auto-switch does: launch overrides keyed by the listed id, which stays the advertised id.
    alias = loader_id or split_model_ref(public_id)[0]
    request._override_alias_id = alias
    request._advertised_id = alias
    return request


# Outside /v1: Open WebUI strips it for management calls (MODEL_MANAGEMENT_ENDPOINTS).
@router.get("/models", include_in_schema = False)
@router.get("/models/", include_in_schema = False)
async def manage_list_models(current_subject: str = Depends(get_current_subject)):
    return {"object": "list", "data": await _chat_catalog()}


# Delegate to Studio's own /load and /unload routes: same gates, account checks and tunnel padding.
@router.post("/models/load", include_in_schema = False)
@router.post("/models/load/", include_in_schema = False)
async def manage_load_model(
    body: _ModelManageBody,
    fastapi_request: Request,
    current_subject: str = Depends(get_current_subject),
):
    request = await _resolve_load_target(body.model)
    return await _inference().load_model(request, fastapi_request, current_subject)


@router.post("/models/unload", include_in_schema = False)
@router.post("/models/unload/", include_in_schema = False)
async def manage_unload_model(
    body: _ModelManageBody, current_subject: str = Depends(get_current_subject)
):
    from models.inference import UnloadRequest

    model_id = await _catalog_model_id(body.model)
    resident = _resident_identifier(model_id)
    response = await _inference().unload_model(UnloadRequest(model_path = resident), current_subject)
    # Answer with the id the client named, never the resident's on-disk path.
    if resident != model_id and getattr(response, "model", None) == resident:
        response = response.model_copy(update = {"model": model_id})
    return response


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
