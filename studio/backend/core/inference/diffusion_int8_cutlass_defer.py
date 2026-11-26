# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The CUTLASS int8 race, deferred until after the first image (``diffusion_int8_gemm._plan_or_defer``).

A load with no stored race verdict for some weight shape used to race them before the first image (22-26 s on an
L4, plus 7-8 s of row-bucket races inside the first render). Those shapes now run the probed Triton int8 GEMM, which
passes the same bit-exact probe against the same eager epilogue, and the race runs on a background thread once a
render has ended and Studio has been idle for ``UNSLOTH_DIFFUSION_INT8_CUTLASS_DEFER_IDLE_S`` seconds. It races one
weight shape at a time; a render arriving meanwhile waits only for the shape in flight, and the rest wait for the
next idle spell. Verdicts persist exactly as a load-time race's, so later processes race nothing. Each shape that
wins is published before a waiting render is let in, and that render drops its CUDA graphs so it re-records them on
the winning kernels.

A back-to-back client never leaves Studio idle (Triton stays 2-4% slower on RTX 6000 Ada / 3090, 3-6% on L4), so after
``UNSLOTH_DIFFUSION_INT8_CUTLASS_DEFER_MAX_RENDERS`` (default 1, 0 = off) such renders the race starts anyway and the
next render waits for it. Waits end at cancel or ``_WAIT_MAX_S``; a deadline give-up disables this for the load.

Only image loads arm it (``allow``); each arm is an owner, and its queued shapes and finalizer belong to it
(``cancel(owner = ...)`` on a failed load, ``cancel()`` on unload). ``UNSLOTH_DIFFUSION_INT8_CUTLASS_DEFER=0`` races at
load, as before."""

from __future__ import annotations

import contextlib
import contextvars
import itertools
import os
import threading
import time
from typing import Any, Callable, Optional

DEFER_ENV = "UNSLOTH_DIFFUSION_INT8_CUTLASS_DEFER"
IDLE_ENV = "UNSLOTH_DIFFUSION_INT8_CUTLASS_DEFER_IDLE_S"
MAX_RENDERS_ENV = "UNSLOTH_DIFFUSION_INT8_CUTLASS_DEFER_MAX_RENDERS"
# Covers the route saving the image and a back-to-back client's next request.
_IDLE_S = 2.0
_IDLE_MAX_S = 600.0
_MAX_RENDERS = 1
_WAIT_MAX_S = 600.0
_WAIT_POLL_S = 0.1
# A unit refused because a CUDA graph was recording elsewhere retries after this long.
_RETRY_S = 0.5
_CANCEL_WAIT_S = 60.0

# Set by the image backend's load and render to the arming owner: a video load (no render hooks here) keeps racing at
# load. None = not armed.
_ALLOWED: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "unsloth_int8_cutlass_defer_allowed", default = None
)
_OWNERS = itertools.count(1)


def enabled() -> bool:
    raw = (os.environ.get(DEFER_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def allowed() -> bool:
    return enabled() and _ALLOWED.get() is not None


def current_owner() -> Optional[int]:
    """The owner (one per ``arm``) the shapes deferred in this context are queued under; None = not armed."""
    return _ALLOWED.get()


def arm() -> Any:
    return _ALLOWED.set(next(_OWNERS))


def disarm(token: Any) -> None:
    _ALLOWED.reset(token)


@contextlib.contextmanager
def allow():
    token = arm()
    try:
        yield
    finally:
        disarm(token)


def max_renders() -> int:
    """Back-to-back renders before the pending race runs regardless of idle; 0 = no bound."""
    raw = (os.environ.get(MAX_RENDERS_ENV) or "").strip()
    try:
        value = int(raw) if raw else _MAX_RENDERS
    except ValueError:
        value = _MAX_RENDERS
    return max(0, value)


def idle_seconds() -> float:
    raw = (os.environ.get(IDLE_ENV) or "").strip()
    try:
        value = float(raw) if raw else _IDLE_S
    except ValueError:
        value = _IDLE_S
    if value != value:  # NaN
        value = _IDLE_S
    return max(0.0, min(_IDLE_MAX_S, value))


class _Job:
    __slots__ = (
        "index",
        "owner",
        "units",
        "logger",
        "before",
        "after",
        "n_shapes",
        "won",
        "cancelled",
    )

    def __init__(
        self, index: Any, owner: Any, logger: Any, before: Callable, after: Callable
    ) -> None:
        self.index = index
        self.owner = owner
        # callables, one weight shape each: False = retried after _RETRY_S, an int = shapes whose winner it published
        self.units: list = []
        self.logger = logger
        self.before = before
        self.after = after
        self.n_shapes = 0
        self.won = 0
        # set under _COND by cancel: nothing of this job publishes or runs after that
        self.cancelled = False


_COND = threading.Condition(threading.Lock())
_JOBS: dict = {}  # (card index, owner) -> _Job
_STATE: dict = {
    "active": 0,  # renders under way
    "last_end": None,  # perf_counter of the last render's end
    "rendered": False,  # a render ended since the jobs were scheduled
    "running": False,  # a unit or a finalizer is running now
    "job": None,  # the job that unit / finalizer belongs to
    "refresh": False,  # winners landed since the last render began: it re-records its graphs
    "thread": None,
    "busy": None,  # extra idle check (a background compile running)
    "ended": 0,  # back-to-back outermost renders with the race pending
    "drain": False,  # run the race now, idle or not
    "retry_at": 0.0,
    "gave_up": False,  # hit _WAIT_MAX_S: no more waits for this load
}
_STATUS: dict = {
    "state": "none",
    "scheduled_at": None,
    "started_at": None,
    "done_at": None,
    "shapes": 0,
    "raced": 0,
    "won": 0,
    "waits": 0,
    "drains": 0,
    "gave_up": 0,
    "error": None,
}


def status() -> dict:
    with _COND:
        out = dict(_STATUS)
        out["pending_units"] = sum(len(j.units) for j in _JOBS.values())
        out["running"] = bool(_STATE["running"])
    return out


def pending(index: Any = None) -> bool:
    with _COND:
        if index is None:
            return bool(_JOBS)
        return any(key[0] == index for key in _JOBS)


def schedule(
    index: Any,
    units: list,
    logger: Any,
    before: Callable,
    after: Callable,
    owner: Any = None,
) -> None:
    """Queue ``units`` (one weight shape each) for card ``index`` under ``owner`` (default: this context's
    ``current_owner``); ``before()`` runs once ahead of them, ``after()`` once they are done (not after a cancel).
    A unit returns False (retry), or how many shapes it published a winner for. Nothing runs until a render ends."""
    if owner is None:
        owner = current_owner() or 0
    with _COND:
        job = _JOBS.get((index, owner))
        if job is None:
            job = _JOBS[(index, owner)] = _Job(index, owner, logger, before, after)
            job.units.append(before)
        job.units.extend(units)
        job.n_shapes += len(units)
        if _STATUS["state"] not in ("pending", "running"):
            _STATUS.update(scheduled_at = time.time(), started_at = None, shapes = 0, raced = 0, won = 0)
            _STATE["ended"] = 0
            _STATE["gave_up"] = False
        _STATUS.update(state = "pending", done_at = None, error = None)
        _STATUS["shapes"] += len(units)
        _COND.notify_all()


def _starved() -> bool:
    """Run the race now: ``max_renders`` back-to-back renders passed. Never during a background compile."""
    bound = max_renders()
    return bool(
        bound
        and not _STATE["gave_up"]
        and _JOBS
        and _STATE["rendered"]
        and _STATE["ended"] >= bound
        and not _busy()
    )


def _arm_drain() -> Optional[tuple]:
    """Caller holds ``_COND``; returns the log line's args."""
    if _STATE["drain"]:
        return None
    _STATE["drain"] = True
    _STATUS["drains"] += 1
    _ensure_worker()
    _COND.notify_all()
    return ([job.logger for job in _JOBS.values() if job.logger is not None], _STATE["ended"])


def _log_drain(drained: Optional[tuple]) -> None:
    if drained and drained[0]:
        drained[0][0].info(
            "diffusion.int8_gemm: %d render(s) ran back to back on the Triton int8 GEMM with the CUTLASS race "
            "pending; the race runs now and the next render waits for what is left (%s=0 waits for an idle "
            "spell instead)",
            drained[1],
            MAX_RENDERS_ENV,
        )


def render_begin(cancel: Any = None) -> bool:
    """A render starts: waits for a unit in flight, or the whole race if ``_starved``, until ``cancel`` or
    ``_WAIT_MAX_S``. Returns True once if winners landed since the last render began (re-record CUDA graphs)."""
    drained = None
    gave_up = False
    with _COND:
        _STATE["active"] += 1
        waited = False
        deadline = time.perf_counter() + _WAIT_MAX_S

        def stop() -> bool:
            if cancel is not None and cancel.is_set():
                return True
            return time.perf_counter() >= deadline

        draining = False
        if _STATE["active"] == 1 and (_STATE["drain"] or _starved()):
            # nested passes (hires, Detailer) never drain
            drained = _arm_drain()
            draining = True
            while (_JOBS or _STATE["running"]) and not stop():
                waited = True
                _COND.wait(_WAIT_POLL_S)
        while _STATE["running"] and not _STATE["gave_up"] and not stop():
            waited = True
            _COND.wait(_WAIT_POLL_S)
        if draining:
            _STATE["drain"] = False
        if (
            not _STATE["gave_up"]
            and (_STATE["running"] or (draining and _JOBS))
            and not (cancel is not None and cancel.is_set())
        ):
            _STATE["gave_up"] = True
            _STATUS["gave_up"] += 1
            gave_up = True
        if waited:
            _STATUS["waits"] += 1
        refresh = False
        if _STATE["active"] == 1:  # a nested pass never drops the outer render's graphs
            refresh, _STATE["refresh"] = _STATE["refresh"], False
        loggers = [job.logger for job in _JOBS.values() if job.logger is not None]
    _log_drain(drained)
    if gave_up and loggers:
        loggers[0].warning(
            "diffusion.int8_gemm: the deferred CUTLASS race did not finish within %.0f s; this render runs on the "
            "Triton int8 GEMM and no render waits on the race again for this load",
            _WAIT_MAX_S,
        )
    return refresh


def render_end(busy: Optional[Callable[[], bool]] = None) -> None:
    """A render ended: the race may start once idle (``idle_seconds``), or now if ``_starved``."""
    drained = None
    with _COND:
        _STATE["active"] = max(0, _STATE["active"] - 1)
        _STATE["last_end"] = time.perf_counter()
        _STATE["rendered"] = True
        if busy is not None:
            _STATE["busy"] = busy
        if _JOBS and not _STATE["active"]:
            _STATE["ended"] += 1
            if _starved():
                drained = _arm_drain()
        if _JOBS:
            _ensure_worker()
        _COND.notify_all()
    _log_drain(drained)


def _ensure_worker() -> None:
    """Caller holds ``_COND``."""
    if _STATE["thread"] is None:
        thread = threading.Thread(target = _worker, name = "unsloth-int8-cutlass-defer", daemon = True)
        thread.start()
        _STATE["thread"] = thread


def _busy() -> bool:
    fn = _STATE["busy"]
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001 - an unreadable check never blocks the race
        return False


def _idle_left() -> Optional[float]:
    """Seconds until the race may run (0 = now), None = wait for a notify (a render is under way)."""
    if _STATE["drain"]:
        return max(0.0, _STATE["retry_at"] - time.perf_counter())
    if _STATE["active"] or not _STATE["rendered"]:
        return None
    if _busy():
        return 1.0
    since = time.perf_counter() - (_STATE["last_end"] or 0.0)
    return max(0.0, idle_seconds() - since)


def _worker() -> None:
    try:
        while True:
            with _COND:
                while True:
                    if not _JOBS:
                        return
                    left = _idle_left()
                    if left == 0.0:
                        break
                    _COND.wait(left)
                key = next(iter(_JOBS))
                job = _JOBS[key]
                if not job.units:
                    _JOBS.pop(key)
                    unit = None
                else:
                    unit = job.units.pop(0)
                    if not _STATE["drain"]:
                        _STATE["ended"] = 0  # idle spell: reset back-to-back count
                    if _STATUS["started_at"] is None:
                        _STATUS["started_at"] = time.time()
                    _STATUS["state"] = "running"
                # The finalizer counts as running too: cancel() waits for it, so it never runs past its load.
                _STATE["running"] = True
                _STATE["job"] = job
            if unit is None:
                _finish(job)
                continue
            result: Any = True
            try:
                result = unit()
            except Exception as exc:  # noqa: BLE001 - a failed shape keeps the Triton path
                _STATUS["error"] = f"{type(exc).__name__}: {exc}"
            with _COND:
                _STATE["running"] = False
                _STATE["job"] = None
                if not job.cancelled:
                    if result is False:
                        job.units.insert(0, unit)
                        _STATE["last_end"] = time.perf_counter() - idle_seconds() + _RETRY_S
                        _STATE["retry_at"] = time.perf_counter() + _RETRY_S
                    else:
                        if unit is not job.before:
                            _STATUS["raced"] += 1
                        won = result if type(result) is int and result > 0 else 0
                        if won:
                            # Published with the winner, before a render waiting on this unit is let in.
                            job.won += won
                            _STATUS["won"] += won
                            _STATE["refresh"] = True
                _COND.notify_all()
    finally:
        with _COND:
            _STATE["thread"] = None
            _STATE["running"] = False
            _STATE["job"] = None
            if _JOBS and _STATE["rendered"]:
                # a job queued while this thread was leaving
                thread = threading.Thread(
                    target = _worker, name = "unsloth-int8-cutlass-defer", daemon = True
                )
                thread.start()
                _STATE["thread"] = thread
            _COND.notify_all()


def _finish(job: _Job) -> None:
    try:
        if not job.cancelled:
            job.after()
    except Exception as exc:  # noqa: BLE001
        _STATUS["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        with _COND:
            _STATE["running"] = False
            _STATE["job"] = None
            cancelled = job.cancelled
            if not cancelled and not _JOBS:
                _STATE["drain"] = False
                _STATUS["state"] = "done"
                _STATUS["done_at"] = time.time()
            _COND.notify_all()
    if job.logger is not None and not cancelled:
        took = (
            (_STATUS["done_at"] or time.time()) - _STATUS["started_at"]
            if _STATUS["started_at"]
            else 0.0
        )
        job.logger.info(
            "diffusion.int8_gemm: deferred CUTLASS race done on cuda:%s in %.1f s: %d of %d weight shape(s) won%s",
            job.index,
            took,
            job.won,
            job.n_shapes,
            "; renders after each win re-record their CUDA graphs on it" if job.won else "",
        )


def cancel(timeout: float = _CANCEL_WAIT_S, owner: Any = None) -> None:
    """Drop the queued units of ``owner`` (a failed load), or every one (None: unload / a new load); waits for a unit
    or finalizer of theirs in flight so no race outlives the weights, and none of theirs publishes afterwards."""
    with _COND:
        mine = lambda job: owner is None or job.owner == owner  # noqa: E731
        had = False
        for key in [key for key, job in _JOBS.items() if mine(job)]:
            _JOBS.pop(key).cancelled = True
            had = True
        flying = _STATE["job"]
        if flying is not None and mine(flying):
            flying.cancelled = True
            had = True
            deadline = time.perf_counter() + timeout
            while _STATE["job"] is flying and time.perf_counter() < deadline:
                _COND.wait(0.5)
        if owner is None:
            _STATE["refresh"] = False
            _STATE["rendered"] = False
            _STATE["busy"] = None
            _STATE["ended"] = 0
            _STATE["drain"] = False
            _STATE["gave_up"] = False
        if had and not _JOBS and _STATUS["state"] in ("pending", "running"):
            _STATUS["state"] = "cancelled"
        _COND.notify_all()


def _reset_for_tests() -> None:
    cancel(timeout = 5.0)
    with _COND:
        _STATE.update(
            active = 0,
            last_end = None,
            rendered = False,
            running = False,
            job = None,
            refresh = False,
            ended = 0,
            drain = False,
            retry_at = 0.0,
            gave_up = False,
        )
        _STATUS.update(
            state = "none",
            scheduled_at = None,
            started_at = None,
            done_at = None,
            shapes = 0,
            raced = 0,
            won = 0,
            waits = 0,
            drains = 0,
            gave_up = 0,
            error = None,
        )
