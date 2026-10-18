# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Download models ahead of time through Unsloth Studio's download manager, follow and stop those downloads, and wait for one before a load."""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections import OrderedDict
from typing import Any, Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError

from studio_mcp import loading
from studio_mcp import placement as place
from studio_mcp.caller import Caller
from studio_mcp.errors import raise_for_route, tool_error
from studio_mcp.forward import forward
from studio_mcp.outputs import DownloadRef, JobStatus
from studio_mcp.tools import as_dict, number, opt_text, route_json, strings, try_json

START = "/api/hub/download"
STATUS = "/api/hub/download-status"
PROGRESS = "/api/hub/download-progress"
CANCEL = "/api/hub/download/cancel"
SCOPED = "@diffusion"
AUDIO_SCOPED = "@audio"
# Native text-to-speech plans add the companion codec repos the load reads; the Audio page stages them the same way.
AUDIO_PLAN = "/api/inference/audio/download-plan"
MAX_PARTS = 16
WAIT_INTERVAL_S = 2.0
ACTIVE = ("running", "cancelling")
# Worst first: one failed part fails the whole download.
PRECEDENCE = ("failed", "cancelling", "running", "cancelled", "idle", "completed")
STATES = {"complete": "completed", "error": "failed"}
WAITING = "Waiting for Unsloth Studio's download"
# How long a load job waits on a download before it fails and names the download to follow.
MAX_LOAD_WAIT_S = 3600.0
READ_TIMEOUT_S = 30.0

Part = tuple[str, Optional[str]]

# (account, kind, model) -> parts of the last download_model, so a load waits on the files it staged.
_remembered: "OrderedDict[tuple[str, str, str], list[Part]]" = OrderedDict()
_REMEMBER_MAX = 256


def _key(caller: Caller, kind: str, model: str) -> tuple[str, str, str]:
    # tts and audio load the same native speech repos, so a prefetch under either serves both.
    return (caller.account_id, "tts" if kind == "audio" else kind, model.lower())


def _remember(caller: Caller, kind: str, model: str, parts: list[Part]) -> None:
    key = _key(caller, kind, model)
    # Prefetches with other options stage other files; a load waits on all of them, newest last.
    merged = [part for part in _remembered.get(key, []) if part not in parts] + parts
    _remembered[key] = merged[-MAX_PARTS:]
    _remembered.move_to_end(key)
    while len(_remembered) > _REMEMBER_MAX:
        _remembered.popitem(last = False)


def format_id(parts: list[Part], generations: Optional[list[Optional[int]]] = None) -> str:
    """``repo[::variant][#generation]`` per part; the generation scopes a cancel to the run download_model started."""
    pieces = []
    for index, (repo, variant) in enumerate(parts):
        piece = repo if variant is None else f"{repo}::{variant}"
        generation = generations[index] if generations else None
        pieces.append(piece if generation is None else f"{piece}#{generation}")
    return ",".join(pieces)


def _parse(download_id: str) -> list[tuple[Part, Optional[int]]]:
    pieces = [p.strip() for p in str(download_id or "").split(",")]
    if len(pieces) > MAX_PARTS:
        raise ToolError(f"A download id names at most {MAX_PARTS} downloads")
    parsed: list[tuple[Part, Optional[int]]] = []
    for piece in pieces:
        head, _hash, tail = piece.rpartition("#")
        generation = int(tail) if head and tail.strip().isdigit() else None
        repo, _sep, variant = (head if generation is not None else piece).partition("::")
        if not repo.strip():
            raise ToolError("Pass the id download_model returned")
        parsed.append(((repo.strip(), variant.strip() or None), generation))
    return parsed


def parse_id(download_id: str) -> list[Part]:
    return [part for part, _generation in _parse(download_id)]


def _state(answer: Any) -> str:
    state = opt_text(as_dict(answer).get("state")) or "idle"
    return STATES.get(state, state)


def _worst(states: list[str]) -> str:
    ranked = [s for s in PRECEDENCE if s in states]
    return ranked[0] if ranked else (states[0] if states else "idle")


def _looks_local(model: str) -> bool:
    # A host path, or the ref: handle a checkpoint or export load sends.
    return model.startswith(("/", "\\", ".", "ref:")) or ":" in model[:3]


def _split(model: str, variant: Optional[str]) -> tuple[str, Optional[str]]:
    """``repo:variant`` shorthand, as load_model reads it, into the repo and the variant."""
    if variant is None and ":" in model and not _looks_local(model):
        model, _sep, variant = model.partition(":")
    return model, variant or None


async def _default_variant(caller: Caller, model: str) -> str:
    listing = as_dict(
        await route_json(
            "GET",
            "/api/hub/gguf-variants",
            caller = caller,
            params = {"repo_id": model},
            hub_header = True,
        )
    )
    quants = [
        q for q in (opt_text(as_dict(v).get("quant")) for v in listing.get("variants") or []) if q
    ]
    chosen = opt_text(listing.get("default_variant")) or (quants[0] if quants else None)
    if chosen is None:
        raise ToolError(
            f"Unsloth Studio found no GGUF variant of {model} to download. "
            f"Pass variant (one of: {', '.join(quants) or 'none listed'})."
        )
    return chosen


def _native_audio(kind: str, model: str, variant: Optional[str]) -> bool:
    # GGUF and audio.cpp speech models download as a GGUF variant instead.
    return kind in ("tts", "audio") and variant is None and "gguf" not in model.lower()


async def _plan_parts(
    caller: Caller,
    kind: str,
    model: str,
    variant: Optional[str],
    token: Optional[str],
    knobs: dict[str, str],
) -> list[tuple[str, list[str]]]:
    # The same plan body the load sends, so the files are the ones the load reads.
    body: dict[str, Any] = {"model_path": model}
    if token:
        body["hf_token"] = token
    if kind not in loading.MEDIA_ROUTES:
        route = AUDIO_PLAN
    else:
        route = loading.MEDIA_ROUTES[kind]["plan"]
        if variant or "gguf" in model.lower():
            entry = await loading._gguf_filename(caller, model, variant)
            if entry:
                body["gguf_filename"] = entry["filename"]
                body["model_kind"] = "gguf"
    plan = as_dict(await place.post_plan(caller, route, body, knobs))
    if opt_text(plan.get("incompatible_reason")):
        raise tool_error(plan["incompatible_reason"])
    if plan.get("plan_failed"):
        raise ToolError(
            f"Unsloth Studio could not list the files {model} needs, so it can't download them ahead of time. "
            "Call load_model, which downloads while it loads."
        )
    entries = [as_dict(e) for e in plan.get("entries") or []]
    return [
        (e["repo_id"], strings(e.get("files")))
        for e in entries
        if opt_text(e.get("repo_id")) and strings(e.get("files"))
    ]


async def start(
    caller: Caller,
    ctx: Optional[Context],
    *,
    model: str,
    kind: str,
    variant: Optional[str],
    hf_token: Optional[str],
    knobs: Optional[dict[str, str]] = None,
) -> DownloadRef:
    if hf_token:
        caller = dataclasses.replace(caller, hf_token = hf_token)
    if kind == "stt":
        status = await loading.stt_status(caller, model)
        engine = loading.stt_engine(status, model)
        wanted = f"{model}:{variant}" if variant and ":" not in model else model
        if wanted not in strings(as_dict(status.get(engine)).get("downloaded_models")):
            await loading.download_stt(
                caller, ctx, model = model, engine = engine, hf_token = hf_token, variant = variant
            )
        return DownloadRef(model = model, kind = kind, variant = variant, status = "completed")

    model, variant = _split(model, variant)
    if kind in loading.MEDIA_ROUTES or _native_audio(kind, model, variant):
        scope = SCOPED if kind in loading.MEDIA_ROUTES else AUDIO_SCOPED
        staged = await _plan_parts(caller, kind, model, variant, caller.hf_token, knobs or {})
        bodies = [
            {"repo_id": repo, "scope_id": scope[1:], "files": files, "transport_mode": "auto"}
            for repo, files in staged
        ]
        parts: list[Part] = [(repo, scope) for repo, _files in staged]
    else:
        if variant is None and "gguf" in model.lower():
            variant = await _default_variant(caller, model)
        bodies = [{"repo_id": model, "transport_mode": "auto"}]
        if variant:
            bodies[0]["gguf_variant"] = variant
        parts = [(model, variant)]
    if not parts:
        # The plan lists nothing left to fetch.
        return DownloadRef(model = model, kind = kind, variant = variant, status = "completed")

    states, attached, generations = [], False, []
    started: list[tuple[Part, Optional[int]]] = []
    try:
        for part, body in zip(parts, bodies):
            answer = as_dict(
                await route_json("POST", START, caller = caller, json_body = body, hub_header = True)
            )
            if answer.get("accepted") is False and answer.get("attached") is not True:
                raise ToolError(
                    f"Unsloth Studio did not start downloading {body['repo_id']}: "
                    f"another download of it is {opt_text(answer.get('state')) or 'in progress'}."
                )
            generation = answer.get("generation")
            generation = generation if isinstance(generation, int) else None
            if answer.get("attached") is not True:
                started.append((part, generation))
            attached = attached or answer.get("attached") is True
            states.append(_state(answer))
            generations.append(generation)
    except BaseException:
        # Stop the parts this call started, so a failed plan leaves nothing running without an id.
        await _stop(caller, started)
        raise
    _remember(caller, kind, model, parts)
    return DownloadRef(
        id = format_id(parts, generations),
        model = model,
        kind = kind,
        variant = variant,
        status = _worst(states),
        attached = attached,
    )


async def _read(caller: Caller, part: Part) -> Optional[dict]:
    """One part's status; None when it is another account's job (404)."""
    repo, variant = part
    response = await forward(
        caller, "GET", STATUS, params = {"repo_id": repo, "gguf_variant": variant or ""}
    )
    if response.status_code == 404:
        return None
    return as_dict(raise_for_route(response))


async def status(caller: Caller, download_id: str) -> JobStatus:
    parsed = _parse(download_id)
    parts = [part for part, _generation in parsed]
    answers, stale = [], None
    for part, generation in parsed:
        answer = await _read(caller, part)
        current = answer.get("generation") if answer is not None else None
        # Another run, newer or after a restart; a registry that pruned the run reads 0.
        if generation is not None and isinstance(current, int) and current not in (0, generation):
            # A newer run of the same repo replaced the one this id names: report that, not its state.
            stale = stale or (
                f"Unsloth Studio started a newer download of {part[0]} after this one; "
                f'follow it with get_job(kind="download", id="{format_id([part], [current])}").'
            )
            answer = None
        answers.append(answer)
    # A part that is not this account's reads as idle: there is nothing of yours to follow.
    states = [_state(a) if a is not None else "idle" for a in answers]
    error = stale or next(
        (opt_text(a.get("error")) for a in answers if a and opt_text(a.get("error"))), None
    )
    aggregate = _worst(states)
    percent = None
    if len(parts) == 1 and parts[0][1] is None and aggregate == "running":
        progress = await try_json(
            caller, PROGRESS, params = {"repo_id": parts[0][0]}, hub_header = True
        )
        fraction = number(as_dict(progress).get("progress"))
        percent = round(min(fraction, 1.0) * 100, 1) if fraction is not None else None
    single = parts[0][1] if len(parts) == 1 and not str(parts[0][1]).startswith("@") else None
    ref = DownloadRef(
        id = download_id, model = parts[0][0], variant = single, status = aggregate, error = error
    )
    return JobStatus(
        kind = "download",
        id = download_id,
        status = aggregate,
        progress_percent = percent,
        error = error,
        download = ref,
    )


def _cancel_body(part: Part, generation: Optional[int]) -> dict[str, Any]:
    repo, variant = part
    body: dict[str, Any] = {"repo_id": repo}
    if variant:
        body["gguf_variant"] = variant
    if generation is not None:
        body["generation"] = generation
    return body


async def _stop(caller: Caller, started: list[tuple[Part, Optional[int]]]) -> None:
    for part, generation in started:
        try:
            await forward(caller, "POST", CANCEL, json_body = _cancel_body(part, generation))
        except Exception:
            pass


async def cancel_parts(caller: Caller, download_id: str) -> tuple[bool, Optional[str]]:
    """Stop every part; returns whether any was stopped and the overall state the answers report.
    A part that refuses (a load owns it) doesn't keep the others running: its error is raised after them."""
    states, refused = [], None
    for part, generation in _parse(download_id):
        response = await forward(caller, "POST", CANCEL, json_body = _cancel_body(part, generation))
        if response.status_code == 404:
            continue
        try:
            answer = raise_for_route(response)
        except ToolError as exc:
            refused = refused or exc
            continue
        states.append(opt_text(as_dict(answer).get("state")) or "idle")
    if refused is not None:
        raise refused
    return "cancelling" in states, (_worst([STATES.get(s, s) for s in states]) if states else None)


def keys_for(caller: Caller, kind: str, model: str, variant: Optional[str]) -> list[Part]:
    """The manager jobs a load of ``model`` would read. A GGUF part with variant None still needs the repo's default."""
    if kind == "stt" or not model or _looks_local(model):
        return []
    if kind in loading.MEDIA_ROUTES:
        model = _split(model, variant)[0]
        remembered = _remembered.get(_key(caller, kind, model))
        return list(remembered or [(model, SCOPED)])
    if _native_audio(kind, model, variant):
        # The plan's parts, codecs included; without a prefetch, the repo itself.
        remembered = _remembered.get(_key(caller, kind, model))
        # Without a prefetch here, the repo itself or the Audio page's staged download of it.
        return list(remembered or [(model, None), (model, AUDIO_SCOPED)])
    # A load without a variant reads the repo's default, not the variant download_model last staged.
    return [_split(model, variant)]


async def wait_for_managed(
    caller: Caller,
    ctx: Optional[Context],
    *,
    model: str,
    kind: str,
    variant: Optional[str],
    budget_s: Optional[float],
) -> None:
    """Wait while Unsloth Studio's download manager is fetching ``model`` for this account. A read that fails proceeds."""
    keys = keys_for(caller, kind, model, variant)
    if keys and keys[0][1] is None and "gguf" in model.lower():
        try:
            keys = [(model, await _default_variant(caller, model))]
        except Exception:
            return
    deadline = None if budget_s is None else time.monotonic() + max(budget_s, 0.0)
    while keys:
        active = []
        for part in keys:
            try:
                # A stalled read must not hold the load past its cap; an unreadable part is no wait.
                answer = await asyncio.wait_for(_read(caller, part), READ_TIMEOUT_S)
            except Exception:
                answer = None
            if answer is not None and opt_text(answer.get("state")) in ACTIVE:
                active.append(part)
        if not active:
            return
        left = None if deadline is None else deadline - time.monotonic()
        if left is not None and left <= 0:
            raise ToolError(
                f"Unsloth Studio is still downloading {model}. "
                f'Follow it with get_job(kind="download", id="{format_id(keys)}"), then call load_model again.'
            )
        if ctx is not None:
            follow = f'get_job(kind="download", id="{format_id(keys)}")'
            await ctx.report_progress(0.0, None, f"{WAITING}; follow it with {follow}")
        await asyncio.sleep(WAIT_INTERVAL_S if left is None else min(WAIT_INTERVAL_S, left))
