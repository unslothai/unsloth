# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Answer engine discovery probes with JSON, or with a 404, but never with the app.

Probes for engine endpoints used to land on main.py's SPA catch-all, so ``GET /props``
returned 200 and a page of HTML. That is worse than a 404: a probe reads the status
before the body. Served here: ``/props``, ``/v1/props``, ``/version``. Everything else
in llama-server's table gets an explicit 404, on its real method as well as GET.

Deliberately NOT served: Ollama's ``/api/tags`` and ``/api/show``. Answering them makes
a client select Ollama and then fail on ``/api/chat``, which Studio does not implement;
the reporting user's client instead fell back to the OpenAI surface and worked.
Advertising a protocol we do not have is the HTML 200 again, one layer up.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException

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

    # These describe the CHILD's route table and web UI, not ours: Studio launches with
    # --metrics, so endpoint_metrics arrives true while public /metrics 404s here
    # (tools/server/server-context.cpp puts all three flags in the payload).
    for _child_only in ("ui", "ui_settings", "cors_proxy_enabled"):
        props.pop(_child_only, None)
    props["endpoint_slots"] = False
    props["endpoint_props"] = False
    props["endpoint_metrics"] = False

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


# Slash forms too: FastAPI's redirect never fires.
# The catch-all fully matches "/props/" and returns index.html; routes/inference.py registers "/v1/models/" likewise.
@router.get("/props", include_in_schema = False)
@router.get("/props/", include_in_schema = False)
@router.get("/v1/props", include_in_schema = False)
@router.get("/v1/props/", include_in_schema = False)
async def llama_props(current_subject: str = Depends(get_current_subject)):
    """llama-server-compatible ``GET /props``."""
    return await asyncio.to_thread(_server_props)


@router.get("/version", include_in_schema = False)
@router.get("/version/", include_in_schema = False)
async def studio_version(current_subject: str = Depends(get_current_subject)):
    """Bare /version only: Ollama spells it /api/version, and answering there is part
    of claiming to be Ollama."""
    # Threaded like /props: the first call resolves the version.
    return {"version": await asyncio.to_thread(_studio_version)}


async def _probe_not_found():
    raise HTTPException(status_code = 404, detail = "API endpoint not found")


# Without these a POST hit the GET-only catch-all and returned 405, reading as "exists, wrong method". HEAD is here
# because Starlette does not admit it on a GET route (measured, fastapi 0.141.1); GET stays with the catch-all so its
# asset lookup wins, and OPTIONS is untouched for CORS preflight.
_PROBE_DENIED_METHODS = ["HEAD", "POST", "PUT", "PATCH", "DELETE"]


# Both forms of every path: no redirect rescues "POST /completion/"
def _both_forms(path: str) -> tuple:
    return (path, path + "/")


for _probe_path in sorted(_ENGINE_PROBE_PATHS):
    for _form in _both_forms(f"/{_probe_path}"):
        router.add_api_route(
            _form,
            _probe_not_found,
            methods = _PROBE_DENIED_METHODS,
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
    """404 the engine paths on GET as well, for an app with no frontend mounted.

    The GET denial normally comes from main.py's SPA catch-all, registered only when
    setup_frontend() finds a build. In API-only mode there is none, so these paths
    matched on method alone and answered 405, which a client reads as "endpoint exists".
    """
    for path in sorted(_ENGINE_PROBE_PATHS) + sorted(_UNSERVED_V1_PROBE_PATHS):
        for form in _both_forms(f"/{path}"):
            app.add_api_route(form, _probe_not_found, methods = ["GET"], include_in_schema = False)
    for form in _both_forms("/slots/{id_slot}"):
        app.add_api_route(form, _probe_not_found, methods = ["GET"], include_in_schema = False)
