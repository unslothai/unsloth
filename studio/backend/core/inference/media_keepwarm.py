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
        self.seen: Any = None
        self.was_busy = False
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

# Per backend (an engine switch replaces the object), keyed by the load target since it may fail
_LOAD_ORIGINS: dict[str, tuple[tuple[str, str, str], bool]] = {}
_LOAD_ORIGINS_GUARD = threading.Lock()


def _origin_key(
    target: Optional[str],
    variant: Optional[str],
    partition: Optional[str] = None,
) -> tuple[str, str, str]:
    """Keys on path plus variant token, since a user Q4 and an API Q8 from one repo share a path."""
    text = str(target or "").strip()
    # a repo id folds case; a path does not, or /models/Foo and /models/foo share an origin
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
    """An API load keeps the user's mark on a shared key, since a failed load leaves that model resident."""
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
    """Unrecognised records read as user-loaded, sparing the model; a failed load keeps the old origin."""
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
    """Excludes the caller's own request, so a drain inside it does not find the backend busy forever."""
    total = _TRACKERS[owner].outstanding(count_pending = count_pending)
    return max(0, total - 1) if current_request_counted else total


# Exact mounted paths only (unauthenticated 404s must not count); progress/cancel polls excluded.
# Load routes count since a load registers with the backend only part way through its POST.
# test_media_keepwarm asserts each is a real mounted route.
_TRACKED_PATHS = {
    "/api/inference/images/generate": DIFFUSION,
    "/api/inference/images/load": DIFFUSION,
    "/api/inference/video/generate": VIDEO,
    "/api/inference/video/load": VIDEO,
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
    """Parks new media requests until the load registers, so a swap cannot cancel work admitted midway."""
    async with _gate(_TRACKERS[owner]):
        yield


@contextlib.asynccontextmanager
async def _gate(tracker: _Tracker):
    # Polled non-blocking acquire (as in llama_keepwarm): keeps the loop free and cancellation-safe
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
    # Via the router (sd.cpp must unload sd-server); skip if neither module is imported, avoids torch
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
    """Identity of the last finished job, so a job that starts and ends between two polls is not missed."""
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
    """One generate_progress() read, so the busy flag and terminal record come from the same job."""
    loading = bool(backend.loading_repo_ids())
    progress = backend.generate_progress() or {}
    return loading or bool(progress.get("active")), _completed_token(progress)


# Load-time fields only: same repo id can be a new build (H3 task, quants); not speed_optims, which moves mid-life
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
        finished = completed is not None and completed != tracker.completed
        tracker.completed = completed
        if busy:
            tracker.note_activity()
            tracker.seen = identity
            tracker.was_busy = True
            return
        if tracker.was_busy or finished:
            # Start the TTL when work ends, not at the last busy poll
            tracker.was_busy = False
            tracker.seen = identity
            tracker.note_activity()
            return
        if identity != tracker.seen:
            # A (re)loaded model counts as activity so it survives at least one TTL before its first generation: loads
            # never pass through the request middleware.
            tracker.seen = identity
            if identity is not None:
                tracker.note_activity()
            return
        if identity is None or not tracker.is_idle(ttl):
            return
        # Re-read just before teardown so a residency veto applied mid-step is honoured
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
        # A request may register _pending during an off-loop setting read. Recheck idleness before unloading.
        if not tracker.is_idle(ttl):
            return
        await asyncio.to_thread(backend.unload)
        # Drop ownership under the arbiter lock so a re-registered same-owner load keeps it
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
    """Read right before teardown, like the TTL, since the setting can be switched on mid-step."""
    from utils.openai_auto_switch_settings import get_auto_unload_api_only
    return get_auto_unload_api_only() and loaded_by_user_action(owner, resident, variant, partition)


async def idle_unload_step() -> None:
    """The media half of one idle_unload_loop tick. Inert when the TTL is off."""
    ttl = await asyncio.to_thread(_effective_ttl)
    if ttl <= 0:
        return
    for tracker in _TRACKERS.values():
        try:
            await _tick(tracker, ttl)
        except Exception as exc:
            # One backend failing to tear down must not stop the other, nor the chat unload.
            logger.debug("idle media unload (%s) failed: %s", tracker.owner, exc)
