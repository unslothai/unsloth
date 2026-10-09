# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load and unload models per kind. A load call blocks until the model is ready (a big download can take an hour), so progress comes from polling the progress routes in a sibling task and reporting it to the agent."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any, Awaitable, Callable, Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError

from studio_mcp.caller import Caller
from studio_mcp.errors import tool_error
from studio_mcp.outputs import LoadResult, UnloadResult
from studio_mcp.tools import as_dict, number, opt_text, route_json, strings

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
        if fraction and opt_text(loading.get("phase")):
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
    kind: str = "llm",
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
    before = await _serving_or_empty(caller)
    payload = await with_progress(
        ctx,
        route_json("POST", "/api/inference/load", caller = caller, json_body = body),
        _llm_poll(caller, model),
    )
    if not isinstance(payload, dict):
        raise ToolError("Unsloth Studio did not confirm the load")
    loaded = opt_text(payload.get("model")) or model
    evicted = strings(payload.get("evicted"))
    # Some loads (an engine switch for an audio model, say) unload the chat model without the
    # route listing it, so name whatever was serving before and is gone now. Only then: the
    # route names what it did drop by another id form, which would list it twice.
    if before and not evicted:
        after = set(await _serving_or_empty(caller))
        evicted += [
            m for m in before if m not in after and m not in evicted and m not in (model, loaded)
        ]
    return LoadResult(
        kind = kind, model = loaded, display_name = opt_text(payload.get("display_name")), evicted = evicted
    )


async def _serving_or_empty(caller: Caller) -> list[str]:
    try:
        serving, _checkpoints, _resident = await _llm_resident(caller)
    except Exception:
        # Only a report: a status read that fails must not turn a finished load into an error.
        return []
    return serving


async def _llm_resident(caller: Caller) -> tuple[list[str], list[str], set[str]]:
    status = as_dict(await route_json("GET", "/api/inference/status", caller = caller))
    serving, checkpoints = (
        strings(status.get("serving")),
        strings(status.get("serving_checkpoints")),
    )
    return serving, checkpoints, {*serving, *checkpoints, *strings(status.get("loaded"))}


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
    kind: {
        "plan": f"/api/inference/{prefix}/download-plan",
        "load": f"/api/inference/{prefix}/load",
        "progress": f"/api/inference/{prefix}/load-progress",
        "status": f"/api/inference/{prefix}/status",
        "unload": f"/api/inference/{prefix}/unload",
    }
    for kind, prefix in (("image", "images"), ("video", "video"))
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
    names = {opt_text(status.get("repo_id")), opt_text(status.get("display_repo_id"))}
    return model.lower() in {name.lower() for name in names if name}


async def _gguf_filename(caller: Caller, model: str, variant: Optional[str]) -> Optional[str]:
    """The checkpoint file to load from a GGUF repo: the named variant, else the repo's default."""
    listing = await route_json(
        "GET", "/api/hub/gguf-variants", caller = caller, params = {"repo_id": model}, hub_header = True
    )
    listing = as_dict(listing)
    variants = [v for v in listing.get("variants") or [] if isinstance(v, dict)]
    wanted = (variant or opt_text(listing.get("default_variant")) or "").lower()
    for entry in variants:
        if str(entry.get("quant", "")).lower() == wanted and opt_text(entry.get("filename")):
            return entry["filename"]
    if variant:
        raise ToolError(f"{model} has no GGUF variant {variant}")
    return opt_text(variants[0].get("filename")) if variants else None


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
    if hf_token:
        # The GGUF lookup reads the Hub token from the header, so a gated repo needs it there too.
        caller = dataclasses.replace(caller, hf_token = hf_token)
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
        if opt_text(plan.get("incompatible_reason")):
            raise tool_error(plan["incompatible_reason"])
        if "gguf_filename" not in body:
            for entry in plan.get("entries") or []:
                if (
                    isinstance(entry, dict)
                    and entry.get("checkpoint")
                    and opt_text(entry.get("gguf_filename"))
                ):
                    body["gguf_filename"] = entry["gguf_filename"]
                    break
    # Starts the load in the background; its answer describes whatever was resident before.
    await route_json("POST", routes["load"], caller = caller, json_body = body)
    idle = 0
    while True:
        progress = as_dict(await route_json("GET", routes["progress"], caller = caller))
        phase = progress.get("phase")
        if phase == "error":
            raise tool_error(opt_text(progress.get("error")) or f"The {kind} model failed to load")
        if phase == "ready":
            break
        status = None
        if phase is None:
            status = await route_json("GET", routes["status"], caller = caller)
            if _resident_matches(status, model):
                break
            idle += 1
            if idle >= _IDLE_POLLS:
                raise ToolError(
                    f"Unsloth Studio stopped reporting the {kind} load without loading {model}. "
                    "Another model load or a training run may be using the GPU; check studio_status and try again."
                )
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
        raise ToolError(f"Unsloth Studio did not finish loading {model} as the {kind} model")
    return LoadResult(
        kind = kind,
        model = opt_text(status.get("display_repo_id")) or opt_text(status.get("repo_id")) or model,
    )


async def unload_media(caller: Caller, kind: str) -> UnloadResult:
    routes = MEDIA_ROUTES[kind]
    before = as_dict(await route_json("GET", routes["status"], caller = caller))
    if before.get("loaded") is not True:
        return UnloadResult(kind = kind, unloaded = False)
    after = await route_json("POST", routes["unload"], caller = caller)
    model = opt_text(before.get("display_repo_id")) or opt_text(before.get("repo_id"))
    unloaded = isinstance(after, dict) and after.get("loaded") is not True
    return UnloadResult(kind = kind, model = model, unloaded = unloaded)


STT_STATUS = "/api/inference/audio/stt/status"
# Transformers first: it is Studio's default and serves custom Whisper repos no list names.
STT_ENGINES = ("transformers", "mtmd", "audiocpp", "gguf")


async def stt_status(caller: Caller, model: Optional[str] = None) -> dict:
    params = {"model": model} if model else None
    return as_dict(await route_json("GET", STT_STATUS, caller = caller, params = params))


def stt_engine(status: dict, model: str) -> str:
    for engine in STT_ENGINES:
        state = status.get(engine)
        if isinstance(state, dict) and model in {
            *strings(state.get("models")),
            *strings(state.get("downloaded_models")),
        }:
            return engine
    return "transformers"


async def download_stt(
    caller: Caller, ctx: Optional[Context], *, model: str, engine: str, hf_token: Optional[str]
) -> None:
    """Download an STT model and wait for it; the download route reads the Hub token from its header."""
    if hf_token:
        caller = dataclasses.replace(caller, hf_token = hf_token)
    started = await route_json(
        "POST",
        "/api/inference/audio/stt/download",
        caller = caller,
        json_body = {"model": model, "engine": engine},
        hub_header = True,
    )
    download_id = started.get("download_id") if isinstance(started, dict) else None
    while True:
        await asyncio.sleep(POLL_INTERVAL_S)
        state = (await stt_status(caller, model)).get(engine)
        download = as_dict(as_dict(state).get("download"))
        if download.get("downloading"):
            done, total = number(download.get("bytes_done")), number(download.get("bytes_total"))
            if ctx is not None and done is not None and total:
                await ctx.report_progress(min(done / total, 1.0), 1.0, "Downloading")
            continue
        if opt_text(download.get("error")):
            raise tool_error(f"Downloading {model} failed: {download['error']}")
        if download.get("cancelled"):
            raise ToolError(f"The download of {model} was cancelled")
        if download_id is None or download_id in strings(download.get("completed_download_ids")):
            return
        if model in strings(as_dict(state).get("downloaded_models")):
            return
        raise ToolError(f"Unsloth Studio stopped downloading {model} before it finished")


async def load_stt(
    caller: Caller,
    ctx: Optional[Context],
    *,
    model: str,
    variant: Optional[str],
    hf_token: Optional[str],
) -> LoadResult:
    status = await stt_status(caller, model)
    engine = stt_engine(status, model)
    state = as_dict(status.get(engine))
    # Load refuses a model that is not on disk, so download first and load only once that finished.
    if model not in strings(state.get("downloaded_models")):
        await download_stt(caller, ctx, model = model, engine = engine, hf_token = hf_token)
    body: dict[str, Any] = {"model": model, "engine": engine}
    if variant:
        body["gguf_variant"] = variant
    loaded = await route_json(
        "POST", "/api/inference/audio/stt/load", caller = caller, json_body = body
    )
    resident = opt_text(loaded.get("loaded_model")) if isinstance(loaded, dict) else None
    if resident is None:
        raise ToolError(
            f"Unsloth Studio did not keep {model} loaded; another load may have replaced it"
        )
    return LoadResult(kind = "stt", model = resident)


async def unload_stt(caller: Caller, model: Optional[str]) -> UnloadResult:
    status = await stt_status(caller)
    resident = {
        engine: status[engine]["loaded_model"]
        for engine in STT_ENGINES
        if isinstance(status.get(engine), dict) and opt_text(status[engine].get("loaded_model"))
    }
    targets = [(e, m) for e, m in resident.items() if model is None or m == model]
    if not targets:
        return UnloadResult(kind = "stt", model = model, unloaded = False)
    engine, name = targets[0]
    await route_json(
        "POST",
        "/api/inference/audio/stt/unload",
        caller = caller,
        params = {"engine": engine, "model": name, "wait": "true"},
    )
    after = (await stt_status(caller)).get(engine)
    still = isinstance(after, dict) and after.get("loaded_model") == name
    return UnloadResult(kind = "stt", model = name, unloaded = not still)
