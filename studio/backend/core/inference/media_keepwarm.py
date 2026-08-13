# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in idle auto-unload for the diffusion image and video backends.

The image and video pipelines are the largest thing Studio holds in VRAM, and until now only
the chat GGUF was freed when the user walked away: one generation and a navigate-away left
several GB resident forever. This is the same mechanism rather than a second one -- the same
in-flight bookkeeping (``LlamaKeepWarmMiddleware`` already tracks the generate routes) and one
step per tick of ``llama_keepwarm.idle_unload_loop``. The TTL is its own setting, off by
default, so nothing here runs until the user asks for it: the tick returns before it resolves
a backend, which is also what keeps torch out of a Studio that never opened these pages.

Each backend owns its teardown barrier, so this decides only WHEN: it calls the same
``unload()`` the arbiter's evictor calls, resolved through ``get_active_diffusion_engine()``
so a native sd.cpp selection stops the sd-server instead of a diffusers pipeline that was
never loaded. Device-agnostic on purpose: CUDA, ROCm, XPU and MPS free VRAM, and a CPU load
frees the host RAM the same weights occupy there, which is just as much the user's.
"""

from __future__ import annotations

import asyncio
import contextlib
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

    def is_idle(self, ttl_seconds: float) -> bool:
        with self._lock:
            return (
                self._inflight == 0
                and self._pending == 0
                and (time.monotonic() - self._last_active) >= ttl_seconds
            )


_TRACKERS = {DIFFUSION: _Tracker(DIFFUSION), VIDEO: _Tracker(VIDEO)}

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
}


def owner_for_path(path: str) -> Optional[str]:
    """Which media backend a tracked inference path generates or loads on, if any."""
    return _TRACKED_PATHS.get(path)


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


def _busy(backend: Any) -> bool:
    """An in-flight load or generation, read off the state both backends publish.

    Both publish progress BEFORE the slow pre-generate setup and keep it active until the
    work is done, so this covers a video job that outlives its POST as well as a denoise."""
    return bool(backend.loading_repo_ids()) or bool(backend.generate_progress().get("active"))


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
        if await asyncio.to_thread(_busy, backend):
            # A load or generation in flight is activity: the TTL restarts from the end of
            # the work, and the backend is never torn down under it.
            tracker.note_activity()
            tracker.seen = identity
            tracker.was_busy = True
            return
        if tracker.was_busy:
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


async def idle_unload_step() -> None:
    """The media half of one idle_unload_loop tick. Inert when the TTL is off."""
    from utils.openai_auto_switch_settings import get_media_auto_unload_idle_seconds

    ttl = get_media_auto_unload_idle_seconds()
    if ttl <= 0:
        return
    for tracker in _TRACKERS.values():
        try:
            await _tick(tracker, ttl)
        except Exception as exc:
            # One backend failing to tear down must not stop the other, nor the chat unload.
            logger.debug("idle media unload (%s) failed: %s", tracker.owner, exc)
