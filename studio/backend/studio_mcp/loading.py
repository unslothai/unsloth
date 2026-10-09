# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load and unload models per kind. A load call blocks until the model is ready (a big download can take an hour), so progress comes from polling the progress routes in a sibling task and reporting it to the agent."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError

from studio_mcp.caller import Caller
from studio_mcp.outputs import LoadResult, UnloadResult
from studio_mcp.tools import number, route_json, text

POLL_INTERVAL_S = 2.0

Progress = Optional[tuple[float, str]]


async def with_progress(
    ctx: Optional[Context], work: Awaitable[Any], poll: Callable[[], Awaitable[Progress]]
) -> Any:
    """Await ``work`` while reporting what ``poll`` sees. A failed poll is skipped, never fatal."""
    task = asyncio.ensure_future(work)
    try:
        while True:
            done, _pending = await asyncio.wait({task}, timeout = POLL_INTERVAL_S)
            if done:
                return task.result()
            try:
                progress = await poll()
            except Exception:
                progress = None
            if progress is not None and ctx is not None:
                await ctx.report_progress(progress[0], 1.0, progress[1])
    finally:
        if not task.done():
            task.cancel()


def _llm_poll(caller: Caller, model: str) -> Callable[[], Awaitable[Progress]]:
    async def poll() -> Progress:
        loading = await route_json("GET", "/api/inference/load-progress", caller = caller)
        fraction = number(loading.get("fraction")) if isinstance(loading, dict) else None
        if fraction and text(loading.get("phase")):
            return min(fraction, 1.0), f"Loading into memory ({loading['phase']})"
        if model.startswith(("/", "\\")) or ":" in model[:3]:
            return None
        download = await route_json(
            "GET",
            "/api/models/download-progress",
            caller = caller,
            params = {"repo_id": model},
            hub_header = True,
        )
        fraction = number(download.get("progress")) if isinstance(download, dict) else None
        return (min(fraction, 1.0), "Downloading") if fraction else None

    return poll


async def load_llm(
    caller: Caller,
    ctx: Optional[Context],
    *,
    model: str,
    variant: Optional[str],
    max_seq_length: Optional[int],
    load_in_4bit: bool,
    hf_token: Optional[str],
) -> LoadResult:
    body: dict[str, Any] = {
        "model_path": model,
        "load_in_4bit": load_in_4bit,
        "max_seq_length": max_seq_length or 0,
    }
    if variant:
        body["gguf_variant"] = variant
    # The load routes read the Hub token from the body, not the header.
    token = hf_token or caller.hf_token
    if token:
        body["hf_token"] = token
    payload = await with_progress(
        ctx,
        route_json("POST", "/api/inference/load", caller = caller, json_body = body),
        _llm_poll(caller, model),
    )
    if not isinstance(payload, dict):
        raise ToolError("Studio did not confirm the load")
    evicted = payload.get("evicted")
    return LoadResult(
        kind = "llm",
        model = text(payload.get("model")) or model,
        display_name = text(payload.get("display_name")),
        evicted = [e for e in evicted if isinstance(e, str)] if isinstance(evicted, list) else [],
    )


def _strings(values: Any) -> list[str]:
    return [v for v in values if isinstance(v, str)] if isinstance(values, list) else []


async def _llm_resident(caller: Caller) -> tuple[list[str], list[str], set[str]]:
    status = await route_json("GET", "/api/inference/status", caller = caller)
    status = status if isinstance(status, dict) else {}
    serving, checkpoints = (
        _strings(status.get("serving")),
        _strings(status.get("serving_checkpoints")),
    )
    return serving, checkpoints, {*serving, *checkpoints, *_strings(status.get("loaded"))}


async def unload_llm(caller: Caller, model: Optional[str]) -> UnloadResult:
    serving, checkpoints, before = await _llm_resident(caller)
    if model is None:
        if not checkpoints:
            return UnloadResult(kind = "llm", unloaded = False)
        target, model = checkpoints[0], (serving[0] if serving else checkpoints[0])
    else:
        # The route unloads by checkpoint id, listed in the same order as serving.
        target = dict(zip(serving, checkpoints)).get(model, model)
    await route_json(
        "POST", "/api/inference/unload", caller = caller, json_body = {"model_path": target}
    )
    # An id that matches nothing still answers success, so compare what is resident before and after.
    _serving, _checkpoints, after = await _llm_resident(caller)
    was_loaded = bool({model, target} & before)
    return UnloadResult(
        kind = "llm", model = model, unloaded = was_loaded and not ({model, target} & after)
    )


MEDIA_ROUTES = {
    "image": {
        "plan": "/api/inference/images/download-plan",
        "load": "/api/inference/images/load",
        "progress": "/api/inference/images/load-progress",
        "status": "/api/inference/images/status",
        "unload": "/api/inference/images/unload",
    },
    "video": {
        "plan": "/api/inference/video/download-plan",
        "load": "/api/inference/video/load",
        "progress": "/api/inference/video/load-progress",
        "status": "/api/inference/video/status",
        "unload": "/api/inference/video/unload",
    },
}
# Polls with no load in flight and the model still not resident before the load counts as lost.
_IDLE_POLLS = 5


def _media_fraction(progress: dict) -> Optional[float]:
    # Images report bytes_downloaded/bytes_total, video downloaded_bytes/expected_bytes.
    done = number(progress.get("bytes_downloaded", progress.get("downloaded_bytes")))
    total = number(progress.get("bytes_total", progress.get("expected_bytes")))
    if done is not None and total:
        return min(done / total, 1.0)
    return number(progress.get("fraction"))


def _resident_matches(status: Any, model: str) -> bool:
    if not isinstance(status, dict) or status.get("loaded") is not True:
        return False
    names = {text(status.get("repo_id")), text(status.get("display_repo_id"))}
    return model.lower() in {name.lower() for name in names if name}


async def _gguf_filename(caller: Caller, model: str, variant: Optional[str]) -> Optional[str]:
    """The checkpoint file to load from a GGUF repo: the named variant, else the repo's default."""
    listing = await route_json(
        "GET", "/api/hub/gguf-variants", caller = caller, params = {"repo_id": model}, hub_header = True
    )
    variants = (
        [v for v in listing.get("variants") or [] if isinstance(v, dict)]
        if isinstance(listing, dict)
        else []
    )
    wanted = (
        (variant or text(listing.get("default_variant")) or "").lower()
        if isinstance(listing, dict)
        else ""
    )
    for entry in variants:
        if str(entry.get("quant", "")).lower() == wanted and text(entry.get("filename")):
            return entry["filename"]
    if variant:
        raise ToolError(f"{model} has no GGUF variant {variant}")
    return text(variants[0].get("filename")) if variants else None


async def load_media(
    caller: Caller,
    ctx: Optional[Context],
    *,
    kind: str,
    model: str,
    variant: Optional[str],
    hf_token: Optional[str],
) -> LoadResult:
    routes = MEDIA_ROUTES[kind]
    body: dict[str, Any] = {"model_path": model}
    token = hf_token or caller.hf_token
    if token:
        body["hf_token"] = token
    if variant or "gguf" in model.lower():
        filename = await _gguf_filename(caller, model, variant)
        if filename:
            body["gguf_filename"] = filename
            body["model_kind"] = "gguf"
    # The plan validates the pick the way the load does, before any download.
    plan = await route_json("POST", routes["plan"], caller = caller, json_body = body)
    if isinstance(plan, dict):
        if text(plan.get("incompatible_reason")):
            raise ToolError(plan["incompatible_reason"])
        if "gguf_filename" not in body:
            for entry in plan.get("entries") or []:
                if (
                    isinstance(entry, dict)
                    and entry.get("checkpoint")
                    and text(entry.get("gguf_filename"))
                ):
                    body["gguf_filename"] = entry["gguf_filename"]
                    break
    # Starts the load in the background; its answer describes whatever was resident before.
    await route_json("POST", routes["load"], caller = caller, json_body = body)
    idle = 0
    while True:
        progress = await route_json("GET", routes["progress"], caller = caller)
        progress = progress if isinstance(progress, dict) else {}
        phase = progress.get("phase")
        if phase == "error":
            raise ToolError(text(progress.get("error")) or f"The {kind} model failed to load")
        if phase == "ready":
            break
        status = None
        if phase is None:
            status = await route_json("GET", routes["status"], caller = caller)
            if _resident_matches(status, model):
                break
            idle += 1
            if idle >= _IDLE_POLLS:
                raise ToolError(f"Studio stopped reporting the {kind} load without loading {model}")
        else:
            idle = 0
            fraction = _media_fraction(progress)
            if ctx is not None and fraction is not None:
                await ctx.report_progress(
                    fraction, 1.0, "Downloading" if phase == "downloading" else str(phase)
                )
        await asyncio.sleep(POLL_INTERVAL_S)
    status = await route_json("GET", routes["status"], caller = caller)
    if not _resident_matches(status, model):
        raise ToolError(f"Studio did not finish loading {model} as the {kind} model")
    return LoadResult(
        kind = kind,
        model = text(status.get("display_repo_id")) or text(status.get("repo_id")) or model,
    )


async def unload_media(caller: Caller, kind: str) -> UnloadResult:
    routes = MEDIA_ROUTES[kind]
    before = await route_json("GET", routes["status"], caller = caller)
    before = before if isinstance(before, dict) else {}
    if before.get("loaded") is not True:
        return UnloadResult(kind = kind, unloaded = False)
    after = await route_json("POST", routes["unload"], caller = caller)
    model = text(before.get("display_repo_id")) or text(before.get("repo_id"))
    unloaded = isinstance(after, dict) and after.get("loaded") is not True
    return UnloadResult(kind = kind, model = model, unloaded = unloaded)
