# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load and unload models per kind. A load (a big download can take an hour) runs as a job in the job store, so a client that stops waiting never stops it: load_model waits up to ``wait_s`` while reporting progress, then returns the job id for get_job(kind="load")."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError

from studio_mcp import job_store
from studio_mcp import placement as place
from studio_mcp.caller import Caller
from studio_mcp.errors import tool_error
from studio_mcp.outputs import EvictedSlot, LoadResult, SlotModel, UnloadAllResult, UnloadResult
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
    # A media load POST, shielded: cancelling the job leaves it running, since the route registers its load from a thread.
    post: Optional[asyncio.Future] = field(default = None, repr = False)
    # This process's media load request number for the slot, taken before the POST (0: none yet).
    generation: int = 0
    # What of the media load request names the load (never its Hub token), so a late unload can tell
    # whether this load is the resident one.
    body: Optional[dict] = field(default = None, repr = False)

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
    # (C:\models\a.gguf) has a drive letter there, not a variant, and a ref: handle is one opaque id.
    if name.startswith("ref:"):
        return name
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
    attach: bool = True,
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
        if (
            slot in CHAT_ROUTE_KINDS
            and attach
            and not variant
            and any(_same(n, model) for n in loading_now)
        ):
            return await _watch_llm(caller, slot, model, before, job)
        # K16, after the attach: M5 waits on a download of the same model here, then M7 refuses a partial one.
        from studio_mcp import model_downloads

        # Inside the job, so the wait outlives wait_s: the load starts once the download ends.
        await model_downloads.wait_for_managed(
            caller,
            job,
            model = model,
            kind = slot,
            variant = variant,
            budget_s = model_downloads.MAX_LOAD_WAIT_S,
        )
        await place.refuse_partial_variant(caller, model, slot, variant)
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
        if model.startswith(("/", "\\", "ref:")) or ":" in model[:3]:
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
    device: Optional[str] = None,
    alongside: Optional[bool] = None,
    kv_cache_type: Optional[str] = None,
    gpu_layers: Optional[int] = None,
    is_lora: bool = False,
) -> LoadResult:
    body: dict[str, Any] = {
        "model_path": model,
        "load_in_4bit": load_in_4bit,
        "max_seq_length": max_seq_length or 0,
    }
    if variant:
        body["gguf_variant"] = variant
    if is_lora:
        body["is_lora"] = True
    # The load routes read the Hub token from the body, not the header.
    token = hf_token or caller.hf_token
    if token:
        body["hf_token"] = token
    if hf_token:
        # The variant lookups before the load read the Hub token from the header.
        caller = dataclasses.replace(caller, hf_token = hf_token)

    async def run(job: LoadJob, before: dict[str, list[str]]) -> LoadResult:
        # The GGUF listing and status reads run inside the job, so a slow one never holds the call
        # past ``wait_s``; device='cpu' for a chat model that is not a GGUF is refused here.
        gguf = await place.require_cpu_gguf(caller, kind, model, variant, device, gpu_layers)
        body.update(
            place.llm_fields(
                kind,
                gguf = gguf,
                device = device,
                alongside = await place.auto_alongside(caller, device, alongside, gpu_layers),
                kv_cache_type = kv_cache_type,
                gpu_layers = gpu_layers,
            )
        )
        payload = await with_progress(
            None,
            route_json("POST", "/api/inference/load", caller = caller, json_body = body),
            _llm_poll(caller, model),
            job,
        )
        if not isinstance(payload, dict):
            raise ToolError("Unsloth Studio did not confirm the load")
        loaded = opt_text(payload.get("model")) or model
        chat = before.get("llm") or []
        # Serving only: a model merely held in memory behind the new primary no longer serves.
        # Bounded: the model is loaded, and a stalled read must not keep the job open.
        status = await _read_json(caller, "/api/inference/status") if chat else None
        # A failed read leaves it unknown whether the previous model still serves.
        after = strings(status.get("serving")) if isinstance(status, dict) else None
        return await finish_load(
            caller,
            kind,
            loaded,
            before,
            payload,
            display_name = opt_text(payload.get("display_name")),
            placement = place.llm_placement(payload, device),
            alongside_applied = place.alongside_applied(chat, after, loaded),
        )

    return await start_load(
        caller,
        ctx,
        slot = kind,
        model = model,
        variant = variant,
        deadline = deadline or deadline_after(None),
        run = run,
        # A load asked with other placement fields is a different load, not one to follow.
        options = (
            max_seq_length or 0,
            load_in_4bit,
            device,
            alongside,
            kv_cache_type,
            gpu_layers,
        ),
        hub_token = token,
        # A load asked with its own settings is never the one already running; following it would
        # skip them (and the CPU check).
        attach = not max_seq_length
        and load_in_4bit
        and all(v is None for v in (device, alongside, kv_cache_type, gpu_layers)),
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


def _resident_matches(
    status: Any,
    model: str,
    body: Optional[dict] = None,
    *,
    settled: bool = True,
) -> bool:
    if not isinstance(status, dict) or status.get("loaded") is not True:
        return False
    names = {opt_text(status.get("repo_id")), opt_text(status.get("display_repo_id"))}
    if model.lower() not in {name.lower() for name in names if name}:
        return False
    # Another file of the same repo is another model: a missing field reads as not loaded yet.
    wanted = opt_text((body or {}).get("gguf_filename"))
    if not wanted:
        return True
    kind = opt_text((body or {}).get("model_kind"))
    if kind and opt_text(status.get("model_kind")) not in (None, kind):
        return False
    resident = opt_text(status.get("gguf_filename"))
    if resident:
        return os.path.normcase(resident) == os.path.normcase(wanted)
    from hub.utils.gguf import extract_quant_token, quant_token_with_bpw

    # Video status names only a GGUF's quant; a plain checkpoint has none to compare.
    quant = extract_quant_token(wanted) if wanted.lower().endswith(".gguf") else None
    if not quant:
        return True
    if (opt_text(status.get("gguf_variant")) or "").lower() != quant.lower():
        return False
    # Status drops a -3.97bpw style suffix, so until the load reports ready a sibling build matches too.
    if not settled and quant_token_with_bpw(wanted) != quant:
        return False
    # MiniMax-H3 ships its fl2va and ref2va partitions at the same quant; status names the one loaded.
    h3_task = opt_text(status.get("h3_task"))
    if not h3_task:
        return True
    try:
        from core.inference.video_minimax_h3 import h3_transformer_task
    except Exception:
        return False
    return h3_task.lower() == h3_transformer_task(wanted).lower()


async def _gguf_filename(caller: Caller, model: str, variant: Optional[str]) -> Optional[dict]:
    """The variant to load from a GGUF repo (the named one, else the repo's default), with its filename, size_bytes, downloaded and partial."""
    listing = await route_json(
        "GET", "/api/hub/gguf-variants", caller = caller, params = {"repo_id": model}, hub_header = True
    )
    entry = place.pick_variant(listing, variant)
    if entry is not None and opt_text(entry.get("filename")):
        return entry
    if variant:
        raise ToolError(f"{model} has no GGUF variant {variant}")
    return None


async def load_media(
    caller: Caller,
    ctx: Optional[Context],
    *,
    kind: str,
    model: str,
    variant: Optional[str],
    hf_token: Optional[str],
    deadline: Optional[float] = None,
    knobs: Optional[dict[str, str]] = None,
) -> LoadResult:
    routes = MEDIA_ROUTES[kind]
    knobs = knobs or {}
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
            entry = await _gguf_filename(caller, model, variant)
            if entry:
                body["gguf_filename"] = entry["filename"]
                body["model_kind"] = "gguf"
        # The plan validates the pick the way the load does, before any download.
        plan = await place.post_plan(caller, routes["plan"], body, knobs)
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
        outcome = _take_media_generation(kind, job)
        try:
            await prepare()
            job.body = _media_identity(body)
            # The answer names what the arbiter unloaded to make room; the load itself goes on in the background.
            job.post = asyncio.ensure_future(
                route_json("POST", routes["load"], caller = caller, json_body = {**body, **knobs})
            )
        except BaseException:
            # Failed or stopped before its POST: this request registers no load.
            _settle_media_generation(kind, job, outcome, False)
            raise
        # Its outcome stays pending until the load is ready: a stopped older load defers to it only then.
        running, answered = [True], [False]

        def post_answered(post: asyncio.Future) -> None:
            if not answered[0]:
                answered[0] = True
                _media_post_answered(kind, job, outcome, post, settle = not running[0])

        job.post.add_done_callback(lambda post: post.cancelled() or post.exception())
        job.post.add_done_callback(post_answered)
        watch = False
        try:
            answer = await with_progress(
                None, asyncio.shield(job.post), _media_poll(caller, routes), job
            )
            status = await _media_ready(caller, kind, model, routes, job, body)
            _MEDIA_READY.setdefault(kind, set()).add(job.generation)
        except Exception as exc:
            if job.post.done():
                post_answered(job.post)
                # Only a confirmed failure withdraws it: after a transient route error the backend
                # may still be loading it, and an older load's late unload must keep deferring.
                if _posted_ok(job.post) and getattr(exc, "_unsloth_load_failed", False):
                    _MEDIA_REGISTERED.get(kind, {}).pop(job.generation, None)
                else:
                    # Its outcome is still unknown: keep reading the backend, so a load that later
                    # fails stops deferring an older load's late unload.
                    watch = job.generation in _MEDIA_REGISTERED.get(kind, ())
            raise
        finally:
            running[0] = False
            if job.post.done():
                post_answered(job.post)
                if watch:
                    _hold(_watch_media_outcome(caller, kind, model, routes, job, job.body, outcome))
                else:
                    _settle_media_generation(
                        kind, job, outcome, job.generation in _MEDIA_REGISTERED.get(kind, ())
                    )
        name = _media_name(status) or model
        return await finish_load(
            caller, kind, name, before, answer, placement = place.media_placement(status)
        )

    return await start_load(
        caller,
        ctx,
        slot = kind,
        model = model,
        variant = variant,
        deadline = deadline or deadline_after(None),
        run = run,
        options = tuple(sorted(knobs.items())),
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


def _load_failed(error: Exception) -> Exception:
    # The backend confirmed this load is not resident (an error phase, or ready but another model).
    error._unsloth_load_failed = True
    return error


async def _media_ready(
    caller: Caller,
    kind: str,
    model: str,
    routes: dict[str, str],
    job: Optional[LoadJob],
    body: Optional[dict] = None,
) -> dict:
    idle = 0
    while True:
        progress = as_dict(await route_json("GET", routes["progress"], caller = caller))
        phase = progress.get("phase")
        if phase == "error":
            raise _load_failed(
                tool_error(opt_text(progress.get("error")) or f"The {kind} model failed to load")
            )
        if phase == "ready":
            break
        if phase is None:
            status = await route_json("GET", routes["status"], caller = caller)
            if _resident_matches(status, model, body, settled = False):
                break
            idle += 1
            if idle >= _IDLE_POLLS:
                raise _load_failed(
                    ToolError(
                        f"Unsloth Studio stopped reporting the {kind} load without loading {model}. "
                        "Another model load or a training run may be using the GPU; check studio_status and try again."
                    )
                )
        else:
            idle = 0
            if job is not None:
                job.note(
                    _media_fraction(progress),
                    "Downloading" if phase == "downloading" else str(phase),
                )
        await asyncio.sleep(POLL_INTERVAL_S)
    status = await route_json("GET", routes["status"], caller = caller)
    if not _resident_matches(status, model, body):
        raise _load_failed(
            ToolError(f"Unsloth Studio did not finish loading {model} as the {kind} model")
        )
    return status


async def unload_media(caller: Caller, kind: str) -> UnloadResult:
    routes = MEDIA_ROUTES[kind]
    # A load_model still loading runs on in the job store; the unload route cancels its load too.
    stopped = _stop_loads(caller, (kind,))
    # A stopped load whose POST is still out registers its load once that returns: unload after it, so the unload cancels it.
    posts = {job.post for job in stopped if job.post is not None and not job.post.done()}
    try:
        if posts:
            await asyncio.wait(posts, timeout = MAX_WAIT_S)
        before = as_dict(await route_json("GET", routes["status"], caller = caller))
        if before.get("loaded") is not True and not stopped:
            return UnloadResult(kind = kind, unloaded = False)
        # Read just before the unload goes out: a POST finishing while it runs may register after it.
        pending = [job for job in stopped if job.post in posts and not job.post.done()]
        _mark_media_unloaded(kind)
        after = await route_json("POST", routes["unload"], caller = caller)
    except BaseException:
        # The caller went away, or a status read or the unload itself failed: unload each stopped
        # load once its POST is back, so one that registers later is not left resident.
        _unload_after_posts(caller, kind, stopped)
        raise
    # A POST that outlasted the wait registers its load after the unload above: unload that one later.
    # One that failed while the unload ran registered nothing.
    pending = [job for job in pending if not job.post.done() or _posted_ok(job.post)]
    late = _unload_after_posts(caller, kind, pending)
    model = (
        opt_text(before.get("display_repo_id"))
        or opt_text(before.get("repo_id"))
        or (stopped[-1].model if stopped else None)
    )
    # Not unloaded yet while a load is still on its way in, as for an STT load that outlasts the call.
    unloaded = not late and isinstance(after, dict) and after.get("loaded") is not True
    return UnloadResult(kind = kind, model = model, unloaded = unloaded)


def _unload_after_posts(caller: Caller, kind: str, jobs: list[LoadJob]) -> bool:
    """Unload each of ``jobs`` once its load POST returns, after this call (held in _LATE_UNLOADS); whether any was scheduled."""
    scheduled = False
    for job in jobs:
        if job.post is None:
            continue
        _hold(_unload_media_once_posted(caller, kind, job))
        scheduled = True
    return scheduled


def _hold(work: Awaitable[None]) -> None:
    # Run ``work`` in the background, held in _LATE_UNLOADS so it is not collected before it ends.
    task = asyncio.ensure_future(work)
    _LATE_UNLOADS.add(task)
    task.add_done_callback(_LATE_UNLOADS.discard)


async def _watch_media_outcome(
    caller: Caller,
    kind: str,
    model: str,
    routes: dict[str, str],
    job: LoadJob,
    body: dict,
    outcome: asyncio.Future,
) -> None:
    """Settle the outcome of ``job``'s registered load once the backend shows it ready or failed.

    Its readiness wait hit a route error, so the load may still be coming or may have failed. Until
    this settles, an older stopped load's late unload waits; a confirmed failure withdraws the
    registration so that unload goes ahead. Route errors and the wait itself are bounded: past
    them the load stays registered, as a newer load the caller asked for.
    """
    deadline = time.monotonic() + _OUTCOME_WATCH_S
    try:
        for failures in range(1, _LATE_UNLOAD_ATTEMPTS + 1):
            left = deadline - time.monotonic()
            if left <= 0:
                return
            try:
                await asyncio.wait_for(_media_ready(caller, kind, model, routes, None, body), left)
                _MEDIA_READY.setdefault(kind, set()).add(job.generation)
                return
            except Exception as exc:
                if getattr(exc, "_unsloth_load_failed", False):
                    _MEDIA_REGISTERED.get(kind, {}).pop(job.generation, None)
                    return
                await asyncio.sleep(POLL_INTERVAL_S * failures)
    finally:
        _settle_media_generation(
            kind, job, outcome, job.generation in _MEDIA_REGISTERED.get(kind, ())
        )


# Media load requests this process made, per slot: the last number taken, those whose POST answered
# (a load registered for it) and whose load has not failed since, the highest registered when an
# unload of the slot went out (that unload ended it), and those whose outcome is not known yet.
_MEDIA_GENERATIONS: dict[str, int] = {}
_MEDIA_REGISTERED: dict[str, dict[int, tuple[str, Optional[dict]]]] = {}
# Registered requests whose load the backend showed ready: none of them is a load still in flight.
_MEDIA_READY: dict[str, set[int]] = {}
_MEDIA_UNLOADED: dict[str, int] = {}
_MEDIA_OUTCOMES: dict[str, dict[int, asyncio.Future]] = {}
# Tries a late unload (or a watch on a newer load's outcome) makes before giving up on route errors.
_LATE_UNLOAD_ATTEMPTS = 5
# How long a newer load's outcome is watched after its readiness wait hit a route error.
_OUTCOME_WATCH_S = 3600.0


def _media_identity(body: dict) -> dict:
    # Only what _resident_matches reads: the registry outlives the request, so never its Hub token.
    return {key: body[key] for key in ("gguf_filename", "model_kind") if key in body}


def _posted_ok(post: Optional[asyncio.Future]) -> bool:
    return post is not None and post.done() and not post.cancelled() and post.exception() is None


def _mark_media_unloaded(kind: str) -> None:
    # Called with no await before the unload POST: every load registered so far is ended by it.
    registered = _MEDIA_REGISTERED.get(kind, {})
    _MEDIA_UNLOADED[kind] = max(_MEDIA_UNLOADED.get(kind, 0), max(registered, default = 0))
    registered.clear()
    _MEDIA_READY.pop(kind, None)


def _newer_media_registered(kind: str, generation: int) -> list[int]:
    """Requests newer than ``generation`` that registered a load no unload of this process ended since."""
    unloaded = _MEDIA_UNLOADED.get(kind, 0)
    return [g for g in _MEDIA_REGISTERED.get(kind, {}) if g > generation and g > unloaded]


def _take_media_generation(kind: str, job: LoadJob) -> asyncio.Future:
    job.generation = _MEDIA_GENERATIONS.get(kind, 0) + 1
    _MEDIA_GENERATIONS[kind] = job.generation
    outcome = asyncio.get_running_loop().create_future()
    _MEDIA_OUTCOMES.setdefault(kind, {})[job.generation] = outcome
    return outcome


def _media_post_answered(
    kind: str, job: LoadJob, outcome: asyncio.Future, post: asyncio.Future, *, settle: bool
) -> None:
    if not _posted_ok(post):
        _settle_media_generation(kind, job, outcome, False)
        return
    if job.generation > _MEDIA_UNLOADED.get(kind, 0):
        _MEDIA_REGISTERED.setdefault(kind, {})[job.generation] = (job.model, job.body)
    if settle:
        # The job already ended (stopped) before its POST answered: no readiness wait settles it.
        _settle_media_generation(
            kind, job, outcome, job.generation in _MEDIA_REGISTERED.get(kind, ())
        )


def _settle_media_generation(
    kind: str, job: LoadJob, outcome: asyncio.Future, registered: bool
) -> None:
    _MEDIA_OUTCOMES.get(kind, {}).pop(job.generation, None)
    if not outcome.done():
        outcome.set_result(registered)


def _newer_media_outcomes(kind: str, generation: int) -> list[asyncio.Future]:
    loop = asyncio.get_running_loop()
    return [
        outcome
        for taken, outcome in _MEDIA_OUTCOMES.get(kind, {}).items()
        if taken > generation and not outcome.done() and outcome.get_loop() is loop
    ]


async def _unload_media_once_posted(caller: Caller, kind: str, job: LoadJob) -> None:
    """Unload the load ``job``'s POST registered, unless it cannot be told to be that load.

    Status and load-progress carry no load identity, so the load is attributed by order: this
    process requested no newer load of the slot that registered, and the backend shows this
    model resident or a load still in flight (a second load is refused while one runs). A newer
    registered load the backend no longer shows resident was evicted some other way and stops
    counting, as does an in-flight load once every newer one was shown ready. Limit: when the
    newer load is the same model and variant, the two cannot be told apart, and the resident one
    is left alone as the model the caller asked for last.
    """
    routes = MEDIA_ROUTES[kind]
    if job.post is None:
        return
    await asyncio.wait({job.post})
    if not _posted_ok(job.post):
        return  # Its load never registered: nothing of it to unload.
    failures = 0
    while True:
        try:
            # A newer request still preparing or posting may fail: wait for its outcome first.
            newer = _newer_media_outcomes(kind, job.generation)
            if newer:
                await asyncio.wait(newer)
                continue
            # Bounded: a stalled read counts as a route error and is retried below.
            progress = as_dict(
                await asyncio.wait_for(route_json("GET", routes["progress"], caller = caller), READ_S)
            )
            status = as_dict(
                await asyncio.wait_for(route_json("GET", routes["status"], caller = caller), READ_S)
            )
            # Rechecked with no await before the unload, so it never cancels a newer request's load.
            if _newer_media_outcomes(kind, job.generation):
                continue
            phase = opt_text(progress.get("phase"))
            in_flight = phase not in (None, "ready", "error")
            registered = _MEDIA_REGISTERED.get(kind, {})
            ready = _MEDIA_READY.get(kind, set())
            replacements = _newer_media_registered(kind, job.generation)
            if replacements:
                # Only a newer load never shown ready may be the one in flight; else it is this job's.
                if (in_flight and any(g not in ready for g in replacements)) or any(
                    _resident_matches(status, *registered[g]) for g in replacements
                ):
                    return  # A newer load is (or may be) what is resident: what the caller wants now.
                # Evicted outside this process's unload (another load, Studio's own unload route):
                # those registrations are stale and no longer defer this unload.
                for g in replacements:
                    registered.pop(g, None)
                    ready.discard(g)
            # Another variant of the same repo is another model, not the load this job stopped.
            if not in_flight and not _resident_matches(status, job.model, job.body):
                return  # Gone or replaced: not a load this job can be shown to own.
            _mark_media_unloaded(kind)
            await route_json("POST", routes["unload"], caller = caller)
            return
        except Exception:
            # A transient route error or a stalled read: read the slot again rather than leave the load resident.
            failures += 1
            if failures >= _LATE_UNLOAD_ATTEMPTS:
                return
            await asyncio.sleep(POLL_INTERVAL_S * failures)


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


async def catalog_stt_engine(caller: Caller, status: dict, model: str) -> str:
    engine = stt_engine(status, model)
    known = any(
        isinstance(state, dict)
        and model in {*strings(state.get("models")), *strings(state.get("downloaded_models"))}
        for state in status.values()
    )
    if known:
        return engine
    # list_models(source="catalog") offers audio.cpp ASR rows no status list names until downloaded.
    rows = as_dict(await try_json(caller, "/api/inference/audio/catalog")).get("models")
    for row in rows if isinstance(rows, list) else []:
        if (
            isinstance(row, dict)
            and row.get("task") == "asr"
            and not row.get("unsupported_reason")
            and str(row.get("id") or "").lower() == model.split(":", 1)[0].lower()
        ):
            return "audiocpp"
    return engine


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
    engine = await catalog_stt_engine(caller, status, model)
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


# Unloads waiting on an STT load or a media load POST that outlasted the unload call, held so they are not collected.
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


async def unload_all(caller: Caller) -> UnloadAllResult:
    """Unload every chat slot, then the image, video and speech-to-text models. The embedding model is Unsloth Studio's own and stays. A step that fails does not stop the rest; the error then names what was unloaded and what was not."""
    failed: list[str] = []

    async def unload(name: str, target: str) -> None:
        try:
            await route_json(
                "POST", "/api/inference/unload", caller = caller, json_body = {"model_path": target}
            )
        except ToolError as exc:
            failed.append(f"{name}: {exc}")

    async def resident() -> tuple[list[str], list[str], set[str]]:
        # Bounded: a stalled status read fails the chat step, and the other slots still go.
        try:
            return await asyncio.wait_for(_llm_resident(caller), READ_S)
        except asyncio.TimeoutError:
            raise ToolError("Unsloth Studio did not say which chat models are loaded") from None

    unloaded: list[SlotModel] = []
    try:
        serving, checkpoints, _before = await resident()
        pairs = list(zip(serving, checkpoints))
        # Kept models first, so unloading the primary never promotes one of them in between.
        for name, target in reversed(pairs):
            await unload(name, target)
        now_serving, now_checkpoints, after = await resident()
        # A model the unloads promoted still serves (by name and by checkpoint): unload it too.
        promoted = [pair for pair in zip(now_serving, now_checkpoints) if pair not in pairs]
        for name, target in promoted:
            await unload(name, target)
        # Models held in memory behind the active one are listed only under loaded.
        known = {*serving, *checkpoints, *now_serving, *now_checkpoints}
        held = [name for name in after if name not in known]
        for name in held:
            await unload(name, name)
        if held or promoted:
            _serving, _checkpoints, after = await resident()
        pairs += promoted + [(name, name) for name in held]
        unloaded += [
            SlotModel(kind = "llm", model = name)
            for name, target in pairs
            if not {name, target} & after
        ]
    except ToolError as exc:
        # Unsloth Studio could not say which chat models are loaded; the other slots still go.
        failed.append(f"llm: {exc}")
    for kind in MEDIA_ROUTES:
        try:
            result = await unload_media(caller, kind)
        except ToolError as exc:
            failed.append(f"{kind}: {exc}")
            continue
        if result.unloaded and result.model:
            unloaded.append(SlotModel(kind = kind, model = result.model))
    for _engine in STT_ENGINES:
        try:
            result = await unload_stt(caller, None)
        except ToolError as exc:
            failed.append(f"stt: {exc}")
            break
        if not (result.unloaded and result.model):
            break
        unloaded.append(SlotModel(kind = "stt", model = result.model))
    if failed:
        done = ", ".join(f"{slot.kind} {slot.model}" for slot in unloaded) or "nothing"
        raise ToolError(f"Unloaded {done}; could not unload {'; '.join(failed)}")
    return UnloadAllResult(unloaded = unloaded)
