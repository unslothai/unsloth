# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load and unload models per kind. A load (a big download can take an hour) runs as a job in the job store, so a client that stops waiting never stops it: load_model waits up to ``wait_s`` while reporting progress, then returns the job id for get_job(kind="load")."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError

from studio_mcp import job_store
from studio_mcp.caller import Caller
from studio_mcp.errors import tool_error
from studio_mcp.outputs import EvictedSlot, LoadResult, UnloadResult
from studio_mcp.tools import as_dict, number, opt_text, route_json, strings, try_json

POLL_INTERVAL_S = 2.0
# The longest a load_model or get_job call waits before returning (user decision); clients must allow more.
MAX_WAIT_S = 240
# What a caller still waiting on a job is told to wait before its next get_job without wait_s.
POLL_AFTER_S = 10
# Forwarded reads have no HTTP timeout; a status or progress read taking longer counts as failed.
READ_S = 15.0

Progress = Optional[tuple[float, str]]


async def with_progress(
    ctx: Optional[Context],
    work: Awaitable[Any],
    poll: Callable[[], Awaitable[Progress]],
    job: Optional[LoadJob] = None,
) -> Any:
    """Await ``work`` while reporting what ``poll`` sees, to the agent and onto ``job``. A failed poll is skipped, never fatal."""
    task = asyncio.ensure_future(work)
    try:
        while True:
            done, _pending = await asyncio.wait({task}, timeout = POLL_INTERVAL_S)
            if done:
                return task.result()
            try:
                progress = await asyncio.wait_for(poll(), timeout = READ_S)
            except Exception:
                progress = None
            if progress is not None and job is not None:
                job.note(*progress)
            if progress is not None and ctx is not None:
                await ctx.report_progress(progress[0], 1.0, progress[1])
    finally:
        if not task.done():
            task.cancel()


@dataclass
class LoadJob(job_store.Job):
    kind: str = "load"
    # The slot kind asked for (llm, tts, audio, image, video) and the model as asked.
    slot: str = "llm"
    model: str = ""
    variant: Optional[str] = None
    # Any other load settings; a repeat call with different ones starts its own load.
    options: Any = None
    fraction: Optional[float] = None
    # Each distinct (fraction, phase) the load went through, for every caller following it.
    marks: list[tuple[float, str]] = field(default_factory = list, repr = False)
    # The STT download this load started, so unload_model can stop it, and whether the STT load
    # request is out (it then runs to the end: cancelling its in-process route does not stop it).
    download: Optional[dict] = None
    posted: bool = False

    def note(self, fraction: Optional[float], phase: str) -> None:
        # A step without a fraction keeps the last one, so progress never drops back.
        self.phase = phase
        self.fraction = self.fraction if fraction is None else fraction
        if fraction is not None and (not self.marks or self.marks[-1] != (fraction, phase)):
            self.marks.append((fraction, phase))

    async def report_progress(
        self,
        progress: float,
        total: Optional[float] = None,
        message: Optional[str] = None,
    ) -> None:
        """Stands in for the agent's Context, so load code that reports progress records it on the job."""
        self.note(progress, message or self.phase)


LOADS = job_store.JobStore(
    capacity = 32, unknown = "No such load job", failure = "The load stopped unexpectedly."
)


def deadline_after(wait_s: Optional[float]) -> float:
    """When a wait of ``wait_s`` seconds (None: the longest) that starts now ends, on the monotonic clock."""
    wait = MAX_WAIT_S if wait_s is None else max(0, min(wait_s, MAX_WAIT_S))
    return time.monotonic() + wait


def _is_path(name: str) -> bool:
    if name.startswith(("/", "~", ".")) or "\\" in name or name[1:3] in (":/", ":\\"):
        return True
    # A relative path (models/Foo.gguf) reads like a Hub id; one that exists on disk is a path.
    try:
        return Path(name).expanduser().exists()
    except (OSError, ValueError):
        return False


def _same(left: Optional[str], right: Optional[str]) -> bool:
    # Hub ids match in any case; a local path keeps the file system's case.
    if not left or not right:
        return False
    return left == right if _is_path(left) or _is_path(right) else left.lower() == right.lower()


def _repo(name: str) -> str:
    # A GGUF is listed as repo:variant in one place and the bare repo in another. A local path
    # (C:\models\a.gguf) has a drive letter there, not a variant.
    head, sep, tail = name.partition(":")
    return head if sep and len(head) > 1 and "/" not in tail and "\\" not in tail else name


def _same_id(left: str, right: str) -> bool:
    return _same(left, right) or _same(_repo(left), _repo(right))


def _in_flight(caller: Caller, asked: LoadJob) -> Optional[LoadJob]:
    for job in LOADS.running(caller.account_id):
        if (
            job.slot == asked.slot
            and _same(job.model, asked.model)
            and (job.variant or "").lower() == (asked.variant or "").lower()
            and job.options == asked.options
        ):
            return job
    return None


def _stop_loads(
    caller: Caller,
    slots: Any,
    model: Optional[str] = None,
) -> list[LoadJob]:
    """Mark this caller's loads of ``slots`` still running in the job store cancelled. cancel has no load kind, so unload_model is how an agent stops one; the caller's unload route stops the load server-side."""
    jobs = [
        job
        for job in LOADS.running(caller.account_id)
        if job.slot in slots and (model is None or _same_id(job.model, model))
    ]
    for job in jobs:
        job_store.mark_cancelled(job)
    return jobs


def _early(job: LoadJob) -> LoadResult:
    return LoadResult(
        kind = job.slot,
        model = job.model,
        loaded = False,
        status = "loading",
        fraction = job.fraction,
        job_id = job.job_id,
        poll_after_s = POLL_AFTER_S,
    )


def outcome(job: LoadJob) -> LoadResult:
    """A finished load's result; a failed or cancelled one raises its error."""
    if job.status == "completed" and isinstance(job.result, LoadResult):
        return job.result
    if job.status == "cancelled":
        raise ToolError(f"The load of {job.model} was cancelled")
    raise ToolError(job.error or f"Unsloth Studio did not load {job.model}")


async def wait_load(ctx: Optional[Context], job: LoadJob, deadline: float) -> LoadResult:
    """Wait for ``job`` until ``deadline`` (monotonic), reporting each step it makes; never cancels it."""
    seen = 0 if job.task is None or not job.marks else len(job.marks) - 1
    while True:
        if ctx is not None:
            for fraction, phase in job.marks[seen:]:
                await ctx.report_progress(min(fraction, 1.0), 1.0, phase)
        seen = len(job.marks)
        left = deadline - time.monotonic()
        if job.finished or job.task is None or left <= 0:
            break
        await asyncio.wait({job.task}, timeout = min(POLL_INTERVAL_S, left))
    return outcome(job) if job.finished else _early(job)


async def start_load(
    caller: Caller,
    ctx: Optional[Context],
    *,
    slot: str,
    model: str,
    variant: Optional[str],
    deadline: float,
    run: Callable[[LoadJob, dict[str, list[str]]], Awaitable[LoadResult]],
    options: Any = None,
    hub_token: Optional[str] = None,
) -> LoadResult:
    """Run a load as a job and wait for it until ``deadline``. A repeat call for the same load while it runs follows it; one with another Hub token (a retry with a fixed token, say) starts its own."""
    token = hub_token or caller.hf_token
    asked = LoadJob(
        job_id = job_store.new_id(),
        account_id = caller.account_id,
        slot = slot,
        model = model,
        variant = variant,
        options = (options, hashlib.sha256(token.encode()).hexdigest()[:16] if token else None),
    )

    async def drive(job: LoadJob) -> LoadResult:
        # Every route read runs inside the job, so a slow one never holds the call past its deadline.
        before, chat = await _resident(caller)
        # A chat load Unsloth Studio is already running (started elsewhere) is followed, not started
        # again. The status names only the repo, so a load asking for a variant always posts.
        loading_now = strings(chat.get("loading"))
        if slot in CHAT_ROUTE_KINDS and not variant and any(_same(n, model) for n in loading_now):
            return await _watch_llm(caller, slot, model, before, job)
        # K16, after the attach: M5 waits on a download of the same model here, then M7 refuses a partial one.
        return await run(job, before)

    job = _in_flight(caller, asked) or LOADS.start(asked, drive)
    return await wait_load(ctx, job, deadline)


# The load_model kinds that load through the chat route.
CHAT_ROUTE_KINDS = ("llm", "tts", "audio")
# Route slot kinds as MCP kinds.
ROUTE_KINDS = {"chat": "llm", "audio": "audio", "image": "image", "video": "video"}


def _media_name(status: Any) -> Optional[str]:
    if not isinstance(status, dict) or status.get("loaded") is not True:
        return None
    return opt_text(status.get("display_repo_id")) or opt_text(status.get("repo_id"))


async def _read_json(caller: Caller, path: str) -> Any:
    """``try_json`` within READ_S: a read that stalls is None, like one that failed."""
    try:
        return await asyncio.wait_for(try_json(caller, path), timeout = READ_S)
    except asyncio.TimeoutError:
        return None


async def _resident(caller: Caller) -> tuple[dict[str, list[str]], dict]:
    chat, image, video = await asyncio.gather(
        _read_json(caller, "/api/inference/status"),
        _read_json(caller, MEDIA_ROUTES["image"]["status"]),
        _read_json(caller, MEDIA_ROUTES["video"]["status"]),
    )
    slots: dict[str, list[str]] = {}
    # A slot that could not be read is left out: this only feeds a report.
    if isinstance(chat, dict):
        slots["llm"] = list(
            dict.fromkeys([*strings(chat.get("serving")), *strings(chat.get("loaded"))])
        )
    for kind, status in (("image", image), ("video", video)):
        if isinstance(status, dict):
            name = _media_name(status)
            slots[kind] = [name] if name else []
    return slots, as_dict(chat)


async def resident_slots(caller: Caller) -> dict[str, list[str]]:
    """What each slot holds now, by MCP kind: llm (chat models), image and video."""
    return (await _resident(caller))[0]


def _slot_rows(rows: Any) -> list[EvictedSlot]:
    found = []
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and opt_text(row.get("model")):
            kind = opt_text(row.get("kind")) or "chat"
            found.append(EvictedSlot(kind = ROUTE_KINDS.get(kind, kind), model = row["model"]))
    return found


async def finish_load(
    caller: Caller, kind: str, model: str, before: dict[str, list[str]], answer: Any, **extras: Any
) -> LoadResult:
    """The LoadResult for a finished load, with what it unloaded: the load answer's ``evicted_slots`` (else its ``evicted`` ids placed by the ``before`` snapshot), plus a before/after diff for what the answer leaves out."""
    answer = as_dict(answer)
    named = strings(answer.get("evicted"))
    slots = _slot_rows(answer.get("evicted_slots"))
    if not slots:
        slots = [
            EvictedSlot(
                kind = next(
                    (k for k, models in before.items() if any(_same_id(name, m) for m in models)),
                    "llm",
                ),
                model = name,
            )
            for name in named
        ]
    # The route lists what it unloaded to make room, never the model this load replaced in its own
    # slot (and nothing at all before it had a list), so whatever else left is found by a diff.
    if any(before.values()):
        after = await resident_slots(caller)
        slots += [
            EvictedSlot(kind = slot, model = name)
            for slot, models in before.items()
            if slot in after
            for name in models
            if not any(_same(name, m) for m in after[slot])
            and not _same_id(name, model)
            and not any(_same_id(name, s.model) for s in slots)
        ]
    evicted = [slot.model for slot in slots] if slots else named
    return LoadResult(kind = kind, model = model, evicted = evicted, evicted_slots = slots, **extras)


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
    deadline: Optional[float] = None,
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

    async def run(job: LoadJob, before: dict[str, list[str]]) -> LoadResult:
        payload = await with_progress(
            None,
            route_json("POST", "/api/inference/load", caller = caller, json_body = body),
            _llm_poll(caller, model),
            job,
        )
        if not isinstance(payload, dict):
            raise ToolError("Unsloth Studio did not confirm the load")
        return await finish_load(
            caller,
            kind,
            opt_text(payload.get("model")) or model,
            before,
            payload,
            display_name = opt_text(payload.get("display_name")),
        )

    return await start_load(
        caller,
        ctx,
        slot = kind,
        model = model,
        variant = variant,
        deadline = deadline or deadline_after(None),
        run = run,
        options = (max_seq_length or 0, load_in_4bit),
        hub_token = token,
    )


async def _watch_llm(
    caller: Caller, kind: str, model: str, before: dict[str, list[str]], job: LoadJob
) -> LoadResult:
    """Follow a chat load Unsloth Studio is already running, until it serves ``model`` or stops loading it."""
    poll = _llm_poll(caller, model)
    misses = 0
    while True:
        await asyncio.sleep(POLL_INTERVAL_S)
        status = await _read_json(caller, "/api/inference/status")
        if not isinstance(status, dict):
            # A failed read is skipped; only a run of them ends the follow.
            misses += 1
            if misses >= _IDLE_POLLS:
                raise ToolError(f"Unsloth Studio stopped answering while loading {model}")
            continue
        misses = 0
        if not any(_same(name, model) for name in strings(status.get("loading"))):
            if any(_same_id(name, model) for name in strings(status.get("serving"))):
                return await finish_load(caller, kind, model, before, {})
            raise ToolError(f"Unsloth Studio stopped loading {model} before it was ready")
        try:
            progress = await asyncio.wait_for(poll(), timeout = READ_S)
        except Exception:
            progress = None
        if progress is not None:
            job.note(*progress)


async def _llm_resident(caller: Caller) -> tuple[list[str], list[str], set[str]]:
    status = as_dict(await route_json("GET", "/api/inference/status", caller = caller))
    serving, checkpoints = (
        strings(status.get("serving")),
        strings(status.get("serving_checkpoints")),
    )
    return serving, checkpoints, {*serving, *checkpoints, *strings(status.get("loaded"))}


async def unload_llm(
    caller: Caller,
    model: Optional[str],
    kind: str = "llm",
) -> UnloadResult:
    serving, checkpoints, before = await _llm_resident(caller)
    if model is None:
        if not checkpoints:
            return UnloadResult(kind = kind, unloaded = False)
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
        kind = kind, model = model, unloaded = was_loaded and not ({model, target} & after)
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
    deadline: Optional[float] = None,
) -> LoadResult:
    routes = MEDIA_ROUTES[kind]
    if hf_token:
        # The GGUF lookup reads the Hub token from the header, so a gated repo needs it there too.
        caller = dataclasses.replace(caller, hf_token = hf_token)
    body: dict[str, Any] = {"model_path": model}
    token = hf_token or caller.hf_token
    if token:
        body["hf_token"] = token

    async def prepare() -> None:
        # Hub and plan reads run inside the job, so a slow one never holds the call past ``wait_s``.
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

    async def run(job: LoadJob, before: dict[str, list[str]]) -> LoadResult:
        await prepare()
        # The answer names what the arbiter unloaded to make room; the load itself goes on in the background.
        answer = await with_progress(
            None,
            route_json("POST", routes["load"], caller = caller, json_body = body),
            _media_poll(caller, routes),
            job,
        )
        status = await _media_ready(caller, kind, model, routes, job)
        name = _media_name(status) or model
        return await finish_load(caller, kind, name, before, answer)

    return await start_load(
        caller,
        ctx,
        slot = kind,
        model = model,
        variant = variant,
        deadline = deadline or deadline_after(None),
        run = run,
    )


def _media_poll(caller: Caller, routes: dict[str, str]) -> Callable[[], Awaitable[Progress]]:
    # While the load POST waits (the arbiter freeing the GPU, a chat context being saved), its phase still shows.
    async def poll() -> Progress:
        progress = as_dict(await route_json("GET", routes["progress"], caller = caller))
        phase = opt_text(progress.get("phase"))
        if phase in (None, "ready", "error"):
            return None
        return _media_fraction(progress) or 0.0, "Downloading" if phase == "downloading" else phase

    return poll


async def _media_ready(
    caller: Caller, kind: str, model: str, routes: dict[str, str], job: LoadJob
) -> dict:
    idle = 0
    while True:
        progress = as_dict(await route_json("GET", routes["progress"], caller = caller))
        phase = progress.get("phase")
        if phase == "error":
            raise tool_error(opt_text(progress.get("error")) or f"The {kind} model failed to load")
        if phase == "ready":
            break
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
            job.note(
                _media_fraction(progress), "Downloading" if phase == "downloading" else str(phase)
            )
        await asyncio.sleep(POLL_INTERVAL_S)
    status = await route_json("GET", routes["status"], caller = caller)
    if not _resident_matches(status, model):
        raise ToolError(f"Unsloth Studio did not finish loading {model} as the {kind} model")
    return status


async def unload_media(caller: Caller, kind: str) -> UnloadResult:
    routes = MEDIA_ROUTES[kind]
    # A load_model still loading runs on in the job store; the unload route cancels its load too.
    stopped = _stop_loads(caller, (kind,))
    before = as_dict(await route_json("GET", routes["status"], caller = caller))
    if before.get("loaded") is not True and not stopped:
        return UnloadResult(kind = kind, unloaded = False)
    after = await route_json("POST", routes["unload"], caller = caller)
    model = (
        opt_text(before.get("display_repo_id"))
        or opt_text(before.get("repo_id"))
        or (stopped[-1].model if stopped else None)
    )
    unloaded = isinstance(after, dict) and after.get("loaded") is not True
    return UnloadResult(kind = kind, model = model, unloaded = unloaded)


STT_STATUS = "/api/inference/audio/stt/status"
# Transformers first: it is Studio's default and serves custom Whisper repos no list names.
STT_ENGINES = ("transformers", "mtmd", "audiocpp", "gguf")


async def stt_status(caller: Caller, model: Optional[str] = None) -> dict:
    params = {"model": model} if model else None
    return as_dict(await route_json("GET", STT_STATUS, caller = caller, params = params))


def stt_engine(status: dict, model: str) -> str:
    # Whisper ids are shared across engines: the one holding the model wins.
    for engine in STT_ENGINES:
        state = status.get(engine)
        if isinstance(state, dict) and state.get("loaded_model") == model:
            return engine
    for engine in STT_ENGINES:
        state = status.get(engine)
        if isinstance(state, dict) and model in {
            *strings(state.get("models")),
            *strings(state.get("downloaded_models")),
        }:
            return engine
    return "transformers"


async def download_stt(
    caller: Caller,
    ctx: Optional[Context],
    *,
    model: str,
    engine: str,
    hf_token: Optional[str],
    variant: Optional[str] = None,
) -> None:
    """Download an STT model and wait for it; the download route reads the Hub token from its header."""
    if hf_token:
        caller = dataclasses.replace(caller, hf_token = hf_token)
    if isinstance(ctx, LoadJob):
        ctx.download = {"model": model, "engine": engine}
    started = await route_json(
        "POST",
        "/api/inference/audio/stt/download",
        caller = caller,
        json_body = {
            "model": model,
            "engine": engine,
            **({"gguf_variant": variant} if variant else {}),
        },
        hub_header = True,
    )
    download_id = started.get("download_id") if isinstance(started, dict) else None
    if isinstance(ctx, LoadJob) and download_id is not None:
        ctx.download = {"model": model, "engine": engine, "download_id": download_id}
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
    # The route folds a variant into "model:variant"; another cached variant does not count.
    wanted = f"{model}:{variant}" if variant and ":" not in model else model
    if wanted not in strings(state.get("downloaded_models")):
        await download_stt(
            caller, ctx, model = model, engine = engine, hf_token = hf_token, variant = variant
        )
    body: dict[str, Any] = {"model": model, "engine": engine}
    if variant:
        body["gguf_variant"] = variant
    if isinstance(ctx, LoadJob):
        ctx.posted = True
    loaded = await route_json(
        "POST", "/api/inference/audio/stt/load", caller = caller, json_body = body
    )
    resident = opt_text(loaded.get("loaded_model")) if isinstance(loaded, dict) else None
    if resident is None:
        raise ToolError(
            f"Unsloth Studio did not keep {model} loaded; another load may have replaced it"
        )
    return LoadResult(kind = "stt", model = resident)


# Unloads waiting on an STT load that outlasted the unload call, held so they are not collected.
_LATE_UNLOADS: set[asyncio.Task] = set()


async def _unload_once_loaded(caller: Caller, job: LoadJob) -> None:
    if job.task is not None:
        await asyncio.wait({job.task})
    if job.status == "completed" and isinstance(job.result, LoadResult):
        try:
            await unload_stt(caller, job.result.model)
        except Exception:
            pass


async def _stop_stt_loads(caller: Caller, model: Optional[str]) -> Optional[str]:
    """Stop this caller's STT loads still downloading, and their downloads, and let one already loading finish so it can be unloaded (later, if it outlasts this call); the last model stopped, if any."""
    stopped = None
    for job in LOADS.running(caller.account_id):
        if job.slot == "stt" and job.posted and (model is None or _same_id(job.model, model)):
            await job_store.settle(job, MAX_WAIT_S)
            if not job.finished:
                late = asyncio.get_running_loop().create_task(_unload_once_loaded(caller, job))
                _LATE_UNLOADS.add(late)
                late.add_done_callback(_LATE_UNLOADS.discard)
    jobs = [
        job
        for job in LOADS.running(caller.account_id)
        if job.slot == "stt" and not job.posted and (model is None or _same_id(job.model, model))
    ]
    for job in jobs:
        job_store.mark_cancelled(job)
        stopped = job.model
        # Partial files stay cached, so a later load resumes the download.
        if job.download is not None:
            await route_json(
                "POST",
                "/api/inference/audio/stt/download/cancel",
                caller = caller,
                json_body = job.download,
            )
    return stopped


async def unload_stt(caller: Caller, model: Optional[str]) -> UnloadResult:
    # A load_model still downloading or loading runs on in the job store, so unloading stops it too.
    stopped = await _stop_stt_loads(caller, model)
    status = await stt_status(caller)
    resident = {
        engine: status[engine]["loaded_model"]
        for engine in STT_ENGINES
        if isinstance(status.get(engine), dict) and opt_text(status[engine].get("loaded_model"))
    }
    targets = [(e, m) for e, m in resident.items() if model is None or m == model]
    if not targets:
        return UnloadResult(kind = "stt", model = stopped or model, unloaded = stopped is not None)
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
