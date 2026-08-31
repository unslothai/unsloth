# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The llama-server / Ollama discovery surface, so third-party clients can probe us.

Studio already answers the OpenAI surface (``GET /v1/models``,
``POST /v1/chat/completions``), and clients that reach a local port commonly probe
llama-server's and Ollama's discovery endpoints alongside it to work out what they
are talking to. Those probes used to land on the SPA catch-all in main.py, which
serves index.html for anything that is not under ``/api/`` or ``/v1/``: a client
asking ``GET /props`` got 200 and a page of HTML. 200-with-HTML is worse than a
404, because a probe reads the status first and only then fails on the body.

GET  /props, /v1/props  -> llama-server's props, with model_path replaced by the
                           public model id (never the on-disk .gguf path)
GET  /version, /api/version -> {"version": ...}, the Ollama shape
GET  /api/tags          -> the catalog in Ollama's list shape
POST /api/show          -> one model in Ollama's show shape

Read-only and authenticated exactly like ``GET /v1/models``: /props carries the
chat template and the slot count, and the listings carry the same ids /v1/models
publishes, so none of it may be looser than the endpoint it mirrors.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from auth.authentication import get_current_subject
from loggers import get_logger

logger = get_logger(__name__)

router = APIRouter()


# Unprefixed endpoints llama-server and Ollama serve that Studio does not. The SPA
# catch-all in main.py answers anything outside /api/ and /v1/ with index.html, so
# without this a probe for /slots or /completion gets 200 plus a page of HTML --
# which reads as "supported" to every client that checks the status before the body.
# The paths this module DOES serve are absent on purpose: they are real routes now,
# matched before the catch-all, and listing them here would be dead weight.
_ENGINE_PROBE_PATHS = frozenset(
    {
        "apply-template",
        "completion",
        "completions",
        "detokenize",
        "embedding",
        "embeddings",
        "health",
        "infill",
        "lora-adapters",
        "metrics",
        "rerank",
        "reranking",
        "slots",
        "tokenize",
    }
)


def is_engine_probe_path(full_path: str) -> bool:
    """True for an engine endpoint that must 404 rather than render the app shell."""
    return full_path.strip("/").lower() in _ENGINE_PROBE_PATHS


# Ollama clients read modified_at as an RFC3339 stamp and reject a bare epoch.
_EPOCH_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _rfc3339(epoch: Optional[int]) -> str:
    try:
        return time.strftime(_EPOCH_FMT, time.gmtime(int(epoch or 0)))
    except (OverflowError, OSError, TypeError, ValueError):
        return time.strftime(_EPOCH_FMT, time.gmtime(0))


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


def _studio_version() -> str:
    from utils.studio_version import get_studio_version
    try:
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
    props: dict[str, Any] = {}
    if getattr(llama_backend, "is_loaded", False):
        # Bounded (5s) and returns None rather than raising, so an engine that is
        # mid-restart degrades to the local view instead of failing the probe.
        upstream = llama_backend._query_server_props()
        if isinstance(upstream, dict):
            props = dict(upstream)

    props.pop("model_path", None)
    if public_id:
        props["model_path"] = public_id
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
@router.get("/api/version", include_in_schema = False)
async def ollama_version(current_subject: str = Depends(get_current_subject)):
    """Ollama-compatible version probe."""
    return {"version": _studio_version()}


# ── /api/tags ─────────────────────────────────────────────────────────────────


def _ollama_details(model: dict) -> dict:
    return {
        "parent_model": "",
        "format": "gguf",
        "family": "",
        "families": None,
        "parameter_size": "",
        # Ollama clients display this verbatim; the catalog's quant is the honest
        # answer and an empty string is the honest answer when there is none.
        "quantization_level": str(model.get("quant") or ""),
    }


def _ollama_tag(model: dict) -> dict:
    model_id = str(model.get("id") or "")
    return {
        "name": model_id,
        "model": model_id,
        "modified_at": _rfc3339(model.get("created")),
        # Ollama publishes a byte size and a blob digest. Studio's catalog carries
        # neither for every entry, and inventing them would make a client believe a
        # digest it could compare against. 0 and "" are the documented empty values.
        "size": 0,
        "digest": "",
        "details": _ollama_details(model),
    }


@router.get("/api/tags", include_in_schema = False)
async def ollama_tags(current_subject: str = Depends(get_current_subject)):
    """Ollama-compatible model listing, from the same catalog as ``GET /v1/models``."""
    models = await _inference()._openai_catalog_objects()
    return {"models": [_ollama_tag(m) for m in models]}


# ── /api/show ─────────────────────────────────────────────────────────────────


class OllamaShowRequest(BaseModel):
    # Ollama accepts either spelling and clients send both; neither is required,
    # so an empty body falls back to whatever is loaded rather than 422-ing a probe.
    model: Optional[str] = None
    name: Optional[str] = None


@router.post("/api/show", include_in_schema = False)
async def ollama_show(
    body: Optional[OllamaShowRequest] = None, current_subject: str = Depends(get_current_subject)
):
    """Ollama-compatible model detail (``POST /api/show``)."""
    from fastapi import HTTPException

    requested = None
    if body is not None:
        requested = body.model or body.name
    requested = (requested or _loaded_public_model_id() or "").strip()
    if not requested:
        raise HTTPException(status_code = 404, detail = "model not found")

    models = await _inference()._openai_catalog_objects()
    # Case-insensitive, matching /v1/models/{id}: the resolver lowercases its index.
    match = next(
        (m for m in models if str(m.get("id") or "").lower() == requested.lower()),
        None,
    )
    if match is None:
        raise HTTPException(status_code = 404, detail = "model not found")

    capabilities = ["completion"]
    if match.get("supports_tools"):
        capabilities.append("tools")
    if match.get("supports_vision"):
        capabilities.append("vision")
    return {
        "license": "",
        "modelfile": "",
        "parameters": "",
        "template": "",
        "details": _ollama_details(match),
        "model_info": {},
        "capabilities": capabilities,
    }
