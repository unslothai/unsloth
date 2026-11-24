# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The CUTLASS int8 race, deferred until after the first image (``diffusion_int8_gemm._plan_or_defer``).

A load with no stored race verdict for some weight shape used to race them before the first image (22-26 s on an
L4, plus 7-8 s of row-bucket races inside the first render). Those shapes now run the probed Triton int8 GEMM, which
passes the same bit-exact probe against the same eager epilogue, and the race runs on a background thread once a
render has ended and Studio has been idle for ``UNSLOTH_DIFFUSION_INT8_CUTLASS_DEFER_IDLE_S`` seconds. It races one
weight shape at a time; a render arriving meanwhile waits only for the shape in flight, and the rest wait for the
next idle spell. Verdicts persist exactly as a load-time race's, so later processes race nothing. Once winners
land, the next render drops its CUDA graphs so it re-records them on the winning kernels.

Only image loads arm it (``allow``); ``UNSLOTH_DIFFUSION_INT8_CUTLASS_DEFER=0`` races at load, as before."""

from __future__ import annotations

import contextlib
import contextvars
import os
import threading
import time
from typing import Any, Callable, Optional

DEFER_ENV = "UNSLOTH_DIFFUSION_INT8_CUTLASS_DEFER"
IDLE_ENV = "UNSLOTH_DIFFUSION_INT8_CUTLASS_DEFER_IDLE_S"
# Covers the route saving the image and a back-to-back client's next request.
_IDLE_S = 2.0
_IDLE_MAX_S = 600.0
# A unit refused because a CUDA graph was recording elsewhere retries after this long.
_RETRY_S = 0.5
_CANCEL_WAIT_S = 60.0

# Set by the image backend's load and render: a video load (no render hooks here) keeps racing at load.
_ALLOWED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unsloth_int8_cutlass_defer_allowed", default = False
)


def enabled() -> bool:
    raw = (os.environ.get(DEFER_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def allowed() -> bool:
    return enabled() and _ALLOWED.get()


def arm() -> Any:
    return _ALLOWED.set(True)


def disarm(token: Any) -> None:
    _ALLOWED.reset(token)


@contextlib.contextmanager
def allow():
    token = arm()
    try:
        yield
    finally:
        disarm(token)


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
    __slots__ = ("index", "units", "logger", "before", "after", "n_shapes")

    def __init__(self, index: Any, logger: Any, before: Callable, after: Callable) -> None:
        self.index = index
        # callables, one weight shape each; a unit returning False is retried after _RETRY_S
        self.units: list = []
        self.logger = logger
        self.before = before
        self.after = after
        self.n_shapes = 0


_COND = threading.Condition(threading.Lock())
_JOBS: dict = {}  # card index -> _Job
_STATE: dict = {
    "active": 0,  # renders under way
    "last_end": None,  # perf_counter of the last render's end
    "rendered": False,  # a render ended since the jobs were scheduled
    "running": False,  # a unit is racing now
    "refresh": False,  # winners landed since the last render began: it re-records its graphs
    "thread": None,
    "busy": None,  # extra idle check (a background compile running)
    "generation": 0,  # bumped by cancel: a unit of an older generation publishes nothing new
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
        return index in _JOBS


def schedule(index: Any, units: list, logger: Any, before: Callable, after: Callable) -> None:
    """Queue ``units`` (one weight shape each) for card ``index``; ``before()`` runs once ahead of them, ``after()``
    once they are done and returns the number of shapes that won. Nothing runs until a render has ended."""
    with _COND:
        job = _JOBS.get(index)
        if job is None:
            job = _JOBS[index] = _Job(index, logger, before, after)
            job.units.append(before)
        job.units.extend(units)
        job.n_shapes += len(units)
        if _STATUS["state"] not in ("pending", "running"):
            _STATUS.update(scheduled_at = time.time(), started_at = None, shapes = 0, raced = 0, won = 0)
        _STATUS.update(state = "pending", done_at = None, error = None)
        _STATUS["shapes"] += len(units)
        _COND.notify_all()


def render_begin() -> bool:
    """A render starts: waits for a race unit in flight (one weight shape), then returns True once if winners
    landed since the last render began (drop the CUDA graphs so they re-record on the winning kernels)."""
    with _COND:
        _STATE["active"] += 1
        waited = False
        while _STATE["running"]:
            waited = True
            _COND.wait(1.0)
        if waited:
            _STATUS["waits"] += 1
        if _STATE["active"] != 1:
            return False  # a nested pass (hires, Detailer) never drops the outer render's graphs
        refresh, _STATE["refresh"] = _STATE["refresh"], False
        return refresh


def render_end(busy: Optional[Callable[[], bool]] = None) -> None:
    """A render ended: the race may start once Studio stays idle (``idle_seconds``)."""
    with _COND:
        _STATE["active"] = max(0, _STATE["active"] - 1)
        _STATE["last_end"] = time.perf_counter()
        _STATE["rendered"] = True
        if busy is not None:
            _STATE["busy"] = busy
        if _JOBS and _STATE["thread"] is None:
            thread = threading.Thread(
                target = _worker, name = "unsloth-int8-cutlass-defer", daemon = True
            )
            thread.start()
            _STATE["thread"] = thread
        _COND.notify_all()


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
                index = next(iter(_JOBS))
                job = _JOBS[index]
                if not job.units:
                    _JOBS.pop(index)
                    unit = None
                else:
                    unit = job.units.pop(0)
                    _STATE["running"] = True
                    generation = _STATE["generation"]
                    if _STATUS["started_at"] is None:
                        _STATUS["started_at"] = time.time()
                    _STATUS["state"] = "running"
            if unit is None:
                _finish(job)
                continue
            ok = True
            try:
                ok = unit() is not False
            except Exception as exc:  # noqa: BLE001 - a failed shape keeps the Triton path
                _STATUS["error"] = f"{type(exc).__name__}: {exc}"
            with _COND:
                _STATE["running"] = False
                if generation == _STATE["generation"]:
                    if ok:
                        if unit is not job.before:
                            _STATUS["raced"] += 1
                    else:
                        job.units.insert(0, unit)
                        _STATE["last_end"] = time.perf_counter() - idle_seconds() + _RETRY_S
                _COND.notify_all()
    finally:
        with _COND:
            _STATE["thread"] = None
            _STATE["running"] = False
            if _JOBS and _STATE["rendered"]:
                # a job queued while this thread was leaving
                thread = threading.Thread(
                    target = _worker, name = "unsloth-int8-cutlass-defer", daemon = True
                )
                thread.start()
                _STATE["thread"] = thread
            _COND.notify_all()


def _finish(job: _Job) -> None:
    won = 0
    try:
        won = int(job.after() or 0)
    except Exception as exc:  # noqa: BLE001
        _STATUS["error"] = f"{type(exc).__name__}: {exc}"
    with _COND:
        _STATUS["won"] += won
        if won:
            _STATE["refresh"] = True
        if not _JOBS:
            _STATUS["state"] = "done"
            _STATUS["done_at"] = time.time()
    if job.logger is not None:
        took = (
            (_STATUS["done_at"] or time.time()) - _STATUS["started_at"]
            if _STATUS["started_at"]
            else 0.0
        )
        job.logger.info(
            "diffusion.int8_gemm: deferred CUTLASS race done on cuda:%s in %.1f s: %d of %d weight shape(s) won%s",
            job.index,
            took,
            won,
            job.n_shapes,
            "; the next render re-records its CUDA graphs on them" if won else "",
        )


def cancel(timeout: float = _CANCEL_WAIT_S) -> None:
    """Drop every queued unit (unload / a new load); waits for a unit in flight so no race outlives the weights."""
    with _COND:
        _STATE["generation"] += 1
        had = bool(_JOBS)
        _JOBS.clear()
        deadline = time.perf_counter() + timeout
        while _STATE["running"] and time.perf_counter() < deadline:
            _COND.wait(0.5)
        _STATE["refresh"] = False
        _STATE["rendered"] = False
        _STATE["busy"] = None
        if had and _STATUS["state"] in ("pending", "running"):
            _STATUS["state"] = "cancelled"
        _COND.notify_all()


def _reset_for_tests() -> None:
    cancel(timeout = 5.0)
    with _COND:
        _STATE.update(active = 0, last_end = None, rendered = False, running = False, refresh = False)
        _STATUS.update(
            state = "none",
            scheduled_at = None,
            started_at = None,
            done_at = None,
            shapes = 0,
            raced = 0,
            won = 0,
            waits = 0,
            error = None,
        )
