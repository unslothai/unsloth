# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Answer engine discovery probes with JSON, or with a 404, but never with the app.

Studio already answers the OpenAI surface (``GET /v1/models``,
``POST /v1/chat/completions``), and clients that reach a local port probe the
engine endpoints alongside it to work out what they are talking to. Those probes
used to land on the SPA catch-all in main.py, which serves index.html for anything
that is not under ``/api/`` or ``/v1/``: a client asking ``GET /props`` got 200 and
a page of HTML. 200-with-HTML is worse than a 404, because a probe reads the status
first and only then fails on the body.

GET  /props, /v1/props  -> llama-server's props, with model_path replaced by the
                           public model id (never the on-disk .gguf path)
GET  /version           -> {"version": ...}

Everything else llama-server serves and Studio does not -- /slots, /completion,
/tokenize, ... -- gets an explicit 404 (``_ENGINE_PROBE_PATHS``), on the real HTTP
method as well as GET.

Deliberately NOT served: Ollama's ``/api/tags`` and ``/api/show``. Answering those
identifies this server as Ollama, and a client that completes Ollama discovery then
posts to ``/api/chat`` or ``/api/generate``, which Studio does not serve, so it would
fail at generation instead of falling back to the OpenAI surface that does work. The
reporting user's client did exactly that fallback and succeeded. Advertising a
protocol we do not implement is the same defect as the HTML 200, one layer up.
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


# Unprefixed endpoints llama-server and Ollama serve that Studio does not. The SPA
# catch-all in main.py answers anything outside /api/ and /v1/ with index.html, so
# without this a probe for /slots or /completion gets 200 plus a page of HTML --
# which reads as "supported" to every client that checks the status before the body.
# Transcribed from llama-server's own route table (tools/server/server.cpp,
# ctx_http.get/post), minus /props, which this module serves. Enumerated from the
# source rather than added one at a time as clients trip over them, so the set is
# complete instead of merely growing. Bare paths only: Studio's own API lives under
# /api/ and /v1/, so nothing here can shadow it.
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

# /slots/:id_slot is a real llama-server route and Studio calls it itself
# (restore_slots_for_resume), so the id has to be matched dynamically rather than
# enumerated. Only this one prefix is dynamic in the table above.
_ENGINE_PROBE_PREFIXES = ("slots/",)


def is_engine_probe_path(full_path: str) -> bool:
    """True for an engine endpoint that must 404 rather than render the app shell."""
    normalized = full_path.strip("/").lower()
    return normalized in _ENGINE_PROBE_PATHS or normalized.startswith(_ENGINE_PROBE_PREFIXES)


def _inference():
    """``routes.inference``, imported at call time.

    routes.inference pulls the whole inference stack, and this module is imported
    from main.py's route table, so the import has to be deferred. Everything here
    goes through this one function rather than importing inline in each handler:
    a test can then hand these handlers a double by replacing it, instead of
    mutating sys.modules for the rest of the suite.
    """
    from routes import inference
    return inference


@functools.lru_cache(maxsize = 1)
def _studio_version() -> str:
    """The version string, resolved once.

    Cached because get_studio_version() shells out to git twice on a source checkout,
    and these handlers are the only ones that call it per request. The answer cannot
    change inside a process, and uncached it would put two subprocess spawns on the
    event loop for every /version probe: cheap on Linux, not on Windows, where
    spawning costs an order of magnitude more.
    """
    try:
        # Inside the guard, not above it: an ImportError here would 500 a probe that
        # has nothing to do with versions.
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
    # _peek_inference_backend does not force the orchestrator's cold build, which
    # waits on hardware detection; a probe must not pay for that.
    orchestrator = inf._peek_inference_backend()
    if orchestrator is not None and getattr(orchestrator, "active_model_name", None):
        return inf._orchestrator_public_model_id(orchestrator)
    return None


# ── /props ────────────────────────────────────────────────────────────────────


def _server_props() -> dict:
    """llama-server's /props with the on-disk path stripped.

    Upstream reports model_path as the absolute .gguf path. Every other Studio
    response maps that to the public id (see core.inference.model_ids), and this
    one is reachable over LAN, so it does the same rather than leaking the layout
    of the user's disk.
    """
    llama_backend = _inference().get_llama_cpp_backend()
    public_id = _loaded_public_model_id()
    llama_loaded = bool(getattr(llama_backend, "is_loaded", False))
    props: dict[str, Any] = {}
    if llama_loaded:
        # Bounded (5s) and documented to return None rather than raise, so an engine
        # that is mid-restart degrades to the local view instead of failing the probe.
        # Belt and braces anyway: a probe is the wrong place to discover that the
        # contract slipped, and the local view is always answerable.
        try:
            upstream = llama_backend._query_server_props()
        except Exception:  # noqa: BLE001
            upstream = None
        if isinstance(upstream, dict):
            props = dict(upstream)

    props.pop("model_path", None)
    if public_id:
        props["model_path"] = public_id

    # The child's props describe the CHILD's route table and its built-in web UI, none
    # of which is reachable through Studio. Copying them verbatim points a client using
    # props for capability discovery at endpoints that 404 here: Studio launches
    # llama-server with --metrics, so endpoint_metrics arrives true while public
    # /metrics is one of the paths this module deliberately denies (verified against
    # tools/server/server-context.cpp, which puts all three flags in the payload).
    for _child_only in ("ui", "ui_settings", "cors_proxy_enabled"):
        props.pop(_child_only, None)
    props["endpoint_slots"] = False
    props["endpoint_props"] = False
    props["endpoint_metrics"] = False

    # Only when llama-server owns the resident model. With a Transformers or MLX model
    # loaded the llama backend is unloaded, and reading its fields anyway would pair the
    # orchestrator's model_path with n_ctx 0, an empty template and total_slots 0 -- a
    # self-contradictory description of a model that is in fact serving. Omitting them
    # says "no llama-server props to report", which is true.
    if llama_loaded:
        props.setdefault("default_generation_settings", {})
        settings = props["default_generation_settings"]
        if isinstance(settings, dict) and "n_ctx" not in settings:
            n_ctx = getattr(llama_backend, "context_length", None)
            if n_ctx:
                settings["n_ctx"] = int(n_ctx)
        props.setdefault("total_slots", 0)
        props.setdefault("chat_template", getattr(llama_backend, "chat_template", "") or "")
    props["build_info"] = f"unsloth-studio/{_studio_version()}"
    return props


@router.get("/props", include_in_schema = False)
@router.get("/v1/props", include_in_schema = False)
async def llama_props(current_subject: str = Depends(get_current_subject)):
    """llama-server-compatible ``GET /props``."""
    return await asyncio.to_thread(_server_props)


# ── /version ──────────────────────────────────────────────────────────────────


@router.get("/version", include_in_schema = False)
async def studio_version(current_subject: str = Depends(get_current_subject)):
    """Version probe. Bare /version only -- Ollama spells it /api/version, and
    answering there is part of claiming to be Ollama."""
    return {"version": _studio_version()}


# ── probe paths Studio does not serve ─────────────────────────────────────────


async def _probe_not_found():
    raise HTTPException(status_code = 404, detail = "API endpoint not found")


# GET is deliberately absent: it stays with the SPA catch-all, which looks for a real
# asset first and only then applies is_engine_probe_path(). Registering GET here would
# shadow that lookup. Without these, a POST to /completion or /tokenize matched the
# GET-only catch-all and returned 405, which a discovery client reads as "the endpoint
# exists, wrong method" -- the same false positive the 404 is meant to close, and these
# endpoints are POST in llama-server. HEAD is listed explicitly: Starlette does NOT
# admit HEAD on a GET route (measured on the pinned fastapi 0.141.1 / starlette 1.6.0,
# HEAD /completion -> 405), so a HEAD probe would infer the endpoint exists. OPTIONS is
# left alone so CORS preflight is unaffected.
_PROBE_DENIED_METHODS = ["HEAD", "POST", "PUT", "PATCH", "DELETE"]

for _probe_path in sorted(_ENGINE_PROBE_PATHS):
    router.add_api_route(
        f"/{_probe_path}",
        _probe_not_found,
        methods = _PROBE_DENIED_METHODS,
        include_in_schema = False,
    )

# The one dynamic entry in llama-server's table, so it needs a path parameter rather
# than a literal: without it POST /slots/0 falls through to the GET-only catch-all and
# answers 405, which is the false positive again. Both slash forms, because FastAPI's
# slash redirect does not apply to a path that matches no route at all.
for _slots_form in ("/slots/{id_slot}", "/slots/{id_slot}/"):
    router.add_api_route(
        _slots_form,
        _probe_not_found,
        methods = _PROBE_DENIED_METHODS,
        include_in_schema = False,
    )
