# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in idle auto-unload for the diffusion image and video backends.

The image and video pipelines are the largest thing Unsloth holds in VRAM, and until now only
the chat GGUF was freed when the user walked away: one generation and a navigate-away left
several GB resident forever. This is the same mechanism rather than a second one -- the same
in-flight bookkeeping (``LlamaKeepWarmMiddleware`` already tracks the generate routes) and one
step per tick of ``llama_keepwarm.idle_unload_loop``. The TTL is its own setting, off by
default, so nothing here runs until the user asks for it: the tick returns before it resolves
a backend, which is also what keeps torch out of an Unsloth that never opened these pages.

Each backend owns its teardown barrier, so this decides only WHEN: it calls the same
``unload()`` the arbiter's evictor calls, resolved through ``get_active_diffusion_engine()``
so a native sd.cpp selection stops the sd-server instead of a diffusers pipeline that was
never loaded. Device-agnostic on purpose: CUDA, ROCm, XPU and MPS free VRAM, and a CPU load
frees the host RAM the same weights occupy there, which is just as much the user's.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import threading
import time
from typing import Any, Optional

from core.inference.gpu_arbiter import DIFFUSION, VIDEO, release_if
from loggers import get_logger

logger = get_logger(__name__)


class _Tracker:
    """Per-owner twin of llama_keepwarm's module-level in-flight bookkeeping."""

    def __init__(self, owner: str) -> None:
        self.owner = owner
        self._lock = threading.Lock()
        # Held across the idle check AND the unload, so a request cannot start in between.
        self.gate = threading.Lock()
        self._inflight = 0
        # Requests blocked on the gate but not yet counted in _inflight (see llama_keepwarm).
        self._pending = 0
        self._last_active = time.monotonic()
        # The model identity the last tick saw, so a fresh load counts as activity.
        self.seen: Any = None
        # Whether the last tick found work in flight, so the tick that sees it end can
        # start the TTL there rather than at the last busy poll.
        self.was_busy = False
        # The terminal record of the last finished background job this tracker has seen, so
        # a job no poll ever sampled as busy still dates the TTL from its completion.
        self.completed: Any = None

    def note_pending(self) -> None:
        with self._lock:
            self._pending += 1

    def note_unpending(self) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)

    def note_start(self) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)
            self._inflight += 1

    def note_end(self, *, counted: bool = True) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            if counted:
                self._last_active = time.monotonic()

    def note_activity(self) -> None:
        with self._lock:
            self._last_active = time.monotonic()

    def outstanding(self, *, count_pending: bool = True) -> int:
        with self._lock:
            return self._inflight + (self._pending if count_pending else 0)

    def is_idle(self, ttl_seconds: float) -> bool:
        with self._lock:
            return (
                self._inflight == 0
                and self._pending == 0
                and (time.monotonic() - self._last_active) >= ttl_seconds
            )


_TRACKERS = {DIFFUSION: _Tracker(DIFFUSION), VIDEO: _Tracker(VIDEO)}

# Per backend, not per engine object: a diffusers <-> sd.cpp switch replaces that object.
# Keyed by the target a load was started for, since a load route records this the moment the
# background load is accepted and that load can still fail with the previous model resident.
_LOAD_ORIGINS: dict[str, tuple[tuple[str, str, str], bool]] = {}
_LOAD_ORIGINS_GUARD = threading.Lock()


def _origin_key(
    target: Optional[str],
    variant: Optional[str],
    partition: Optional[str] = None,
) -> tuple[str, str, str]:
    """The build a provenance record answers for: the repo/path AND its GGUF variant.

    The path alone is not the build: a user-loaded Q4 and an API load of Q8 from the same repo
    share it, so a failed API load would mark the resident Q4 as API-loaded and free it.

    The variant is the token ``status()`` publishes, not a fuller filename label. Two builds the
    token cannot separate therefore share a key, which errs toward reporting a model as
    user-loaded and sparing it. Keying on more would never match what the resident model
    publishes, and would spare everything.
    """
    text = str(target or "").strip()
    # A repo id folds case; a path does not, or /models/Foo and /models/foo share an origin.
    key = os.path.normcase(text) if os.path.isabs(text) else text.lower()
    return (key, str(variant or "").strip().lower(), str(partition or "").strip().lower())


def note_load_origin(
    owner: str,
    target: Optional[str],
    variant: Optional[str] = None,
    partition: Optional[str] = None,
    *,
    user_action: bool,
) -> None:
    """Record who asked for the model a load route is bringing up.

    An API load over a user-loaded build with the same key keeps the user's mark: the two are
    indistinguishable to ``status()`` (sibling GGUFs can share a quant token), and a load that
    is accepted and then fails leaves the user's model resident, which this must not reclassify.
    """
    key = _origin_key(target, variant, partition)
    with _LOAD_ORIGINS_GUARD:
        previous = _LOAD_ORIGINS.get(owner)
        if not user_action and previous is not None and previous[0] == key and previous[1]:
            return
        _LOAD_ORIGINS[owner] = (key, user_action)


def loaded_by_user_action(
    owner: str,
    resident: Optional[str] = None,
    variant: Optional[str] = None,
    partition: Optional[str] = None,
) -> bool:
    """Whether the RESIDENT model was loaded from Unsloth rather than by an API request.

    The record only answers for the build it was written against: a load that was accepted and
    then failed leaves the previous model resident, and reading its origin off the failed
    load would let the idle unload free a model the user had pinned. Anything unrecognised
    reads as user-loaded, which is the direction that spares a model.
    """
    with _LOAD_ORIGINS_GUARD:
        entry = _LOAD_ORIGINS.get(owner)
    if entry is None:
        return True
    key, user_action = entry
    if resident is not None and key[0] and key != _origin_key(resident, variant, partition):
        return True
    return user_action


def other_request_count(
    owner: str,
    *,
    current_request_counted: bool = False,
    count_pending: bool = True,
) -> int:
    """Tracked media requests in flight on *owner*, excluding this one when it is counted.

    The auto-switch drain reads this from inside a tracked request, so its own entry must
    not make the backend look permanently busy. ``count_pending`` False drops requests that
    have registered but are still blocked on the gate, which a gate holder must not wait for.
    """
    total = _TRACKERS[owner].outstanding(count_pending = count_pending)
    return max(0, total - 1) if current_request_counted else total


# The concrete mounted paths, not any path that ends in one of them: FastAPI answers
# /v1/anything/images/generations with a 404 without ever running an endpoint, and stamping
# that as activity would let unauthenticated 404s keep a pipeline resident past every TTL.
# Only the generate and load routes: */generate-progress and */generate/cancel are polled
# while the user watches, so they must not count as activity.
#
# The load routes are here because a load registers with the backend only part way through
# its POST, so sampling loading_repo_ids() cannot see one that has been accepted but not yet
# started. Holding the gate for the whole request closes that window rather than narrowing it.
#
# test_media_keepwarm asserts every one of these is a real route on the routers main.py
# mounts, so a rename cannot silently drop the protection an in-flight generation rides on.
_TRACKED_PATHS = {
    # studio_router, mounted at /api/inference only.
    "/api/inference/images/generate": DIFFUSION,
    "/api/inference/images/load": DIFFUSION,
    # video_router, likewise.
    "/api/inference/video/generate": VIDEO,
    "/api/inference/video/load": VIDEO,
    # The OpenAI-compatible route is on inference_router, mounted at both prefixes.
    "/api/inference/images/generations": DIFFUSION,
    "/v1/images/generations": DIFFUSION,
    "/api/inference/videos": VIDEO,
    "/v1/videos": VIDEO,
}


def owner_for_path(path: str) -> Optional[str]:
    """Which media backend a tracked inference path generates or loads on, if any."""
    return _TRACKED_PATHS.get(path)


@contextlib.asynccontextmanager
async def admission_gate(owner: str):
    """Hold new tracked media requests off *owner* for the duration of the block.

    The media auto-switch keeps this closed from its final drain check through registering
    the load: the load path cancels active work as it tears the pipeline down, so a
    generation admitted in that gap would be cut short by a swap that just waited for the
    queue to clear. Requests arriving meanwhile park in ``begin_request`` until it reopens.
    """
    async with _gate(_TRACKERS[owner]):
        yield


@contextlib.asynccontextmanager
async def _gate(tracker: _Tracker):
    # Polled non-blocking acquire, exactly like llama_keepwarm's: it keeps the wait off this
    # loop AND cancellation-safe, since a cancel lands during the sleep with the gate free.
    while not tracker.gate.acquire(blocking = False):
        await asyncio.sleep(0.02)
    try:
        yield
    finally:
        tracker.gate.release()


async def begin_request(owner: str) -> None:
    """Count a generation request in, holding the gate off the idle unload."""
    tracker = _TRACKERS[owner]
    tracker.note_pending()
    started = False
    try:
        async with _gate(tracker):
            tracker.note_start()
            started = True
    finally:
        if not started:
            tracker.note_unpending()


def end_request(owner: str, *, counted: bool = True) -> None:
    """Count a generation request out. ``counted`` False drops it without stamping
    activity, for a request rejected before it ever reached the backend."""
    _TRACKERS[owner].note_end(counted = counted)


def _diffusion_engine() -> Any:
    # Through the router: a native sd.cpp selection must unload the sd-server, not the
    # diffusers pipeline. Nothing can be resident before either module is imported, and
    # importing them here just to find that out would drag torch into every idle tick.
    if not {"core.inference.diffusion", "core.inference.sd_cpp_backend"} & set(sys.modules):
        return None
    from core.inference.diffusion_engine_router import get_active_diffusion_engine
    return get_active_diffusion_engine()


def _video_engine() -> Any:
    if "core.inference.video" not in sys.modules:
        return None
    from core.inference.video import get_video_backend
    return get_video_backend()


_ENGINES = {DIFFUSION: _diffusion_engine, VIDEO: _video_engine}


def engine_if_imported(owner: str) -> Any:
    return _ENGINES[owner]()


def _completed_token(progress: dict[str, Any]) -> Optional[tuple[Any, ...]]:
    """Identity of the last FINISHED job, for a backend that publishes a terminal record.

    The video backend runs its generation as a job that outlives the POST and holds the
    terminal record until the next job starts, so a job that begins and ends between two
    15s polls is never sampled as busy: without this the TTL would still date from the POST
    that started it, and the model could be freed a poll interval short of the window the
    user configured. The image backend generates inside its request (which the middleware
    covers end to end) and publishes no terminal record, so this is None there.
    """
    phase = progress.get("phase")
    if progress.get("active") or phase not in ("completed", "failed"):
        return None
    video = progress.get("video")
    return (
        phase,
        (video or {}).get("id") if isinstance(video, dict) else None,
        progress.get("error"),
    )


def _probe(backend: Any) -> tuple[bool, Optional[tuple[Any, ...]]]:
    """One off-loop read of the state both backends publish: whether a load or generation
    is in flight, plus the terminal record of the last finished job.

    Both publish progress BEFORE the slow pre-generate setup and keep it active until the
    work is done, so this covers a video job that outlives its POST as well as a denoise.
    One read of generate_progress(), so the two answers cannot come from different jobs."""
    loading = bool(backend.loading_repo_ids())
    progress = backend.generate_progress() or {}
    return loading or bool(progress.get("active")), _completed_token(progress)


# What makes one resident build different from another. The repo id is not enough:
# MiniMax-H3 stages a different denoiser per h3_task, so a cached fl2va -> ref2va reload
# is a new build under the same id, and the quants are picked per load too. Only fields
# fixed at load time, so nothing that moves under a resident model (a Speed=Auto compile
# flips speed_optims mid-life) can be mistaken for a reload and keep it warm forever.
_IDENTITY_FIELDS = (
    "repo_id",
    "base_repo",
    "model_kind",
    "gguf_variant",
    "h3_task",
    "transformer_quant",
    "text_encoder_quant",
)


def _identity(status: dict[str, Any]) -> Optional[tuple[Any, ...]]:
    if not status.get("loaded"):
        return None
    return tuple(status.get(field) for field in _IDENTITY_FIELDS)


async def _tick(tracker: _Tracker, ttl: float) -> None:
    backend = _ENGINES[tracker.owner]()
    if backend is None:
        return
    async with _gate(tracker):
        status = await asyncio.to_thread(backend.status)
        identity = _identity(status)
        busy, completed = await asyncio.to_thread(_probe, backend)
        # A job whose whole life fell between two polls is only ever visible as a terminal
        # record this tracker has not seen before. The first tick to see any record spends
        # one TTL on work that may be older, which is the direction that keeps the model.
        finished = completed is not None and completed != tracker.completed
        tracker.completed = completed
        if busy:
            # A load or generation in flight is activity: the TTL restarts from the end of
            # the work, and the backend is never torn down under it.
            tracker.note_activity()
            tracker.seen = identity
            tracker.was_busy = True
            return
        if tracker.was_busy or finished:
            # The work ended between two ticks. A video generation outlives its POST, so the
            # only activity it stamps after that is the busy polls, and dating the TTL from
            # the last of those spends up to one poll interval of the keep-warm window the
            # user configured before the model was even free. Start it here instead.
            tracker.was_busy = False
            tracker.seen = identity
            tracker.note_activity()
            return
        if identity != tracker.seen:
            # A (re)loaded model counts as activity so it survives at least one TTL before
            # its first generation: loads never pass through the request middleware.
            tracker.seen = identity
            if identity is not None:
                tracker.note_activity()
            return
        if identity is None or not tracker.is_idle(ttl):
            return
        # Re-read the effective setting immediately before the teardown. One step covers both
        # backends and an unload frees several GB, so a residency veto applied while it runs
        # (Model Memory, or the TTL itself moved) would otherwise be ignored by every teardown
        # left in the step, freeing a model the settings page now calls pinned.
        ttl = await asyncio.to_thread(_effective_ttl)
        if ttl <= 0 or not tracker.is_idle(ttl):
            return
        if await asyncio.to_thread(
            _user_pinned,
            tracker.owner,
            status.get("repo_id"),
            status.get("gguf_variant"),
            status.get("h3_task"),
        ):
            return
        # A request may register _pending during an off-loop setting read.
        # Recheck idleness before unloading.
        if not tracker.is_idle(ttl):
            return
        await asyncio.to_thread(backend.unload)
        # Drop ownership only if nothing came back meanwhile, and check it under the arbiter
        # lock so a same-owner load that re-registered keeps it. Mirrors /images/unload.
        await asyncio.to_thread(
            release_if,
            tracker.owner,
            lambda: not backend.loading_repo_ids() and not backend.status().get("loaded"),
        )
        tracker.seen = None
        logger.info("Idle auto-unload: freed the %s model after %ss idle", tracker.owner, ttl)


def _effective_ttl() -> float:
    """The media TTL with the residency veto applied: 0 means nothing is unloaded."""
    from utils.openai_auto_switch_settings import get_media_auto_unload_idle_seconds
    return float(get_media_auto_unload_idle_seconds())


def _user_pinned(
    owner: str, resident: Optional[str], variant: Optional[str], partition: Optional[str]
) -> bool:
    """Whether "only unload models loaded by the API" spares this backend's model.

    Read immediately before the teardown, like the TTL: the setting can be turned on
    while a step is running, and a model it now pins must not be freed by the rest of it.
    """
    from utils.openai_auto_switch_settings import get_auto_unload_api_only
    return get_auto_unload_api_only() and loaded_by_user_action(owner, resident, variant, partition)


async def idle_unload_step() -> None:
    """The media half of one idle_unload_loop tick. Inert when the TTL is off."""
    # Keep this SQLite-backed setting read off the event loop.
    ttl = await asyncio.to_thread(_effective_ttl)
    if ttl <= 0:
        return
    for tracker in _TRACKERS.values():
        try:
            await _tick(tracker, ttl)
        except Exception as exc:
            # One backend failing to tear down must not stop the other, nor the chat unload.
            logger.debug("idle media unload (%s) failed: %s", tracker.owner, exc)
