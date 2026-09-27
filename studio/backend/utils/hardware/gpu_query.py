# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cached, coalesced nvidia-smi reads: drop-in for ``subprocess.run`` with a bounded wait.

Categories: ``static`` (inventory fields, TTL 600 s), ``display`` (live fields inside
:func:`display_reads`, TTL 3 s, stale-while-revalidate), ``critical`` (live fields anywhere
else: never cached or joined, since another process's allocation raises no Studio event).
``subprocess.run(timeout=...)`` waits unboundedly for a killed child on a blocked driver, so
the child runs on a daemon thread. Failures pass through uncached. ``UNSLOTH_GPU_QUERY_CACHE=0``
disables all of this.
"""

from __future__ import annotations

import contextlib
import copy
import contextvars
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

from loggers import get_logger
from utils import gpu_memory_events as _events

logger = get_logger(__name__)

# Bound at import: tests stubbing threading.Thread must not stop foreground reads.
_Thread = threading.Thread

STATIC = "static"
DISPLAY = "display"
CRITICAL = "critical"

_STATIC_FIELDS = frozenset(
    {
        "index",
        "uuid",
        "gpu_uuid",
        "name",
        "gpu_name",
        "serial",
        "pci.bus_id",
        "gpu_bus_id",
        "memory.total",
        "compute_cap",
        "driver_version",
        "vbios_version",
        "count",
    }
)
_STATIC_SUBCOMMANDS = frozenset({"-L", "--list-gpus", "topo"})

_DISPLAY_MAX_STALE_S = 60.0
_SLOW_BACKOFF_S = 30.0
_SLOW_WAIT_S = 1.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def enabled() -> bool:
    return os.environ.get("UNSLOTH_GPU_QUERY_CACHE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def ttl_for(kind: str) -> float:
    if kind == STATIC:
        return _env_float("UNSLOTH_GPU_QUERY_STATIC_TTL", 600.0)
    return _env_float("UNSLOTH_GPU_QUERY_DISPLAY_TTL", 3.0)


def _background_timeout() -> float:
    return _env_float("UNSLOTH_GPU_QUERY_BACKGROUND_TIMEOUT", 120.0)


_display_mode: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unsloth_gpu_query_display", default = False
)
_fresh_mode: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unsloth_gpu_query_fresh", default = False
)


@contextlib.contextmanager
def display_reads() -> Iterator[None]:
    """Live reads may be seconds old. Never wrap anything that decides placement or fit."""
    token = _display_mode.set(True)
    try:
        yield
    finally:
        _display_mode.reset(token)


@contextlib.contextmanager
def fresh_reads() -> Iterator[None]:
    """Every live read runs the CLI (settle loops, fenced paired readings)."""
    token = _fresh_mode.set(True)
    try:
        yield
    finally:
        _fresh_mode.reset(token)


def classify(argv: Sequence[str]) -> str:
    args = list(argv[1:])
    if args and args[0] in _STATIC_SUBCOMMANDS:
        return STATIC
    fields = _query_fields(argv)
    if fields and set(fields) <= _STATIC_FIELDS:
        return STATIC
    return DISPLAY if _display_mode.get() else CRITICAL


def _query_fields(argv: Sequence[str]) -> list[str]:
    for arg in argv[1:]:
        if arg.startswith("--query-gpu="):
            return [f.strip() for f in arg.split("=", 1)[1].split(",") if f.strip()]
    return []


def _nounits(argv: Sequence[str]) -> bool:
    return any(arg.startswith("--format=") and "nounits" in arg for arg in argv[1:])


@dataclass
class _Entry:
    result: subprocess.CompletedProcess
    at: float
    gen: int
    static_gen: int
    started: float = 0.0


@dataclass
class _Flight:
    key: tuple
    gen: int
    static_gen: int
    started: float
    epoch: int = 0
    done: threading.Event = field(default_factory = threading.Event)
    result: Optional[subprocess.CompletedProcess] = None
    exc: Optional[BaseException] = None


@dataclass
class _Stats:
    calls: int = 0
    spawned: int = 0
    hits: int = 0
    coalesced: int = 0
    stale_served: int = 0
    timeouts: int = 0
    background_refreshes: int = 0


_lock = threading.Lock()
_cache: dict[tuple, _Entry] = {}
_inflight: dict[tuple, _Flight] = {}
_static_gen = 0
# Bumped by reset(): a child that outlives a reset must not refill the emptied cache.
_reset_epoch = 0
_slow_until = 0.0
_stats = _Stats()


def invalidate_gpu_memory(reason: str = "") -> None:
    _events.invalidate_gpu_memory(reason)
    if reason:
        logger.debug("GPU memory readings invalidated: %s", reason)


def invalidate_static(reason: str = "") -> None:
    global _static_gen
    _events.invalidate_gpu_memory(reason)
    with _lock:
        _static_gen += 1
    if reason:
        logger.debug("GPU inventory readings invalidated: %s", reason)


def reset() -> None:
    global _static_gen, _slow_until, _stats, _reset_epoch
    _events.invalidate_gpu_memory("reset")
    with _lock:
        _reset_epoch += 1
        _cache.clear()
        _inflight.clear()
        _static_gen += 1
        _slow_until = 0.0
        _stats = _Stats()


def stats() -> dict[str, Any]:
    with _lock:
        out = dict(_stats.__dict__)
        out["driver_slow"] = time.monotonic() < _slow_until
        out["cached_keys"] = len(_cache)
        out["inflight"] = len(_inflight)
        return out


def driver_slow() -> bool:
    return time.monotonic() < _slow_until


def _mark_slow() -> None:
    global _slow_until
    with _lock:
        _stats.timeouts += 1
        _slow_until = time.monotonic() + _SLOW_BACKOFF_S


def _resolved(exe: str) -> str:
    """Never raises: a cache key must not add a failure the direct call lacked."""
    try:
        return shutil.which(exe) or exe
    except Exception:
        return exe


def _copy(result: Any) -> Any:
    return copy.copy(result)


def _entry_fresh(entry: _Entry, kind: str, now: float) -> bool:
    if now - entry.at > ttl_for(kind):
        return False
    if kind == STATIC:
        return entry.static_gen == _static_gen
    return entry.gen == _events.generation()


def _run_child(flight: _Flight, argv: list, kind: str, kwargs: dict) -> None:
    try:
        with _lock:
            _stats.spawned += 1
        # Looked up at call time so a patched subprocess.run (tests) is honoured.
        result = subprocess.run(argv, **kwargs)
        flight.result = result
        stdout = getattr(result, "stdout", None)
        if (
            getattr(result, "returncode", None) == 0
            and isinstance(stdout, str)
            and not stdout.strip()
        ):
            # An answered "no rows" is never cached, but it must not leave an older non-empty one served.
            with _lock:
                if flight.epoch == _reset_epoch:
                    existing = _cache.get(flight.key)
                    if existing is not None and existing.started <= flight.started:
                        del _cache[flight.key]
        elif getattr(result, "returncode", None) == 0 and isinstance(stdout, str):
            with _lock:
                if flight.epoch != _reset_epoch:
                    return
                existing = _cache.get(flight.key)
                # A slow child that began before the current entry's must not replace it.
                if existing is None or existing.started <= flight.started:
                    _cache[flight.key] = _Entry(
                        result = result,
                        at = flight.started,
                        gen = flight.gen,
                        static_gen = flight.static_gen,
                        started = flight.started,
                    )
    except BaseException as exc:  # handed to the waiters, never raised on this thread
        flight.exc = exc
        if isinstance(exc, subprocess.TimeoutExpired) and flight.epoch == _reset_epoch:
            _mark_slow()
    finally:
        with _lock:
            if _inflight.get(flight.key) is flight:
                del _inflight[flight.key]
        flight.done.set()


def _start_or_join(key: tuple, argv: list, kind: str, kwargs: dict, timeout: float) -> _Flight:
    """Never joins a child started before the last invalidation of its kind."""
    with _lock:
        flight = _inflight.get(key)
        if flight is not None and (
            flight.static_gen == _static_gen
            if kind == STATIC
            else flight.gen == _events.generation()
        ):
            _stats.coalesced += 1
            return flight
        flight = _Flight(
            key = key,
            gen = _events.generation(),
            static_gen = _static_gen,
            started = time.monotonic(),
            epoch = _reset_epoch,
        )
        _inflight[key] = flight
    child_kwargs = dict(kwargs)
    child_kwargs["timeout"] = max(float(timeout), _background_timeout())
    thread = _Thread(
        target = _run_child,
        args = (flight, argv, kind, child_kwargs),
        name = "nvidia-smi-query",
        daemon = True,
    )
    thread.start()
    return flight


def _flight_outcome(flight: _Flight, argv: list, timeout: float) -> subprocess.CompletedProcess:
    if flight.exc is not None:
        exc = flight.exc
        if isinstance(exc, subprocess.TimeoutExpired):
            raise subprocess.TimeoutExpired(argv, timeout) from None
        raise exc
    assert flight.result is not None
    return _copy(flight.result)


def _fallback(key: tuple, kind: str) -> Optional[subprocess.CompletedProcess]:
    now = time.monotonic()
    with _lock:
        entry = _cache.get(key)
    if entry is not None:
        if kind == STATIC and entry.static_gen == _static_gen:
            with _lock:
                _stats.stale_served += 1
            return _copy(entry.result)
        if kind == DISPLAY and now - entry.at <= _DISPLAY_MAX_STALE_S:
            with _lock:
                _stats.stale_served += 1
            return _copy(entry.result)
    return None


def run_nvidia_smi(
    argv: Sequence[str],
    *,
    timeout: float,
    kind: Optional[str] = None,
    cache: bool = True,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """Drop-in for ``subprocess.run``. ``cache=False``: CLI always runs, only the bounded wait applies."""
    argv = list(argv)
    if not enabled():
        return subprocess.run(argv, timeout = timeout, **kwargs)
    kind = kind or classify(argv)
    text_mode = bool(
        kwargs.get("text") or kwargs.get("encoding") or kwargs.get("universal_newlines")
    )
    # Runner object (not id: ids are reused) and resolved binary keyed so swapped fakes / PATH never share.
    key = (subprocess.run, _resolved(argv[0]), tuple(argv), text_mode)

    if not cache or kind == CRITICAL or (_fresh_mode.get() and kind != STATIC):
        flight = _Flight(
            key = key,
            gen = _events.generation(),
            static_gen = _static_gen,
            started = time.monotonic(),
            epoch = _reset_epoch,
        )
        child_kwargs = dict(kwargs)
        child_kwargs["timeout"] = timeout
        _Thread(
            target = _run_child,
            args = (flight, argv, kind, child_kwargs),
            name = "nvidia-smi-query",
            daemon = True,
        ).start()
        # Whole caller timeout (+1 s reap grace): a slow driver that answers must still be heard.
        if not flight.done.wait(timeout + 1.0):
            _mark_slow()
            raise subprocess.TimeoutExpired(argv, timeout)
        return _flight_outcome(flight, argv, timeout)

    now = time.monotonic()
    with _lock:
        _stats.calls += 1
        entry = _cache.get(key)
        if entry is not None and _entry_fresh(entry, kind, now):
            _stats.hits += 1
            return _copy(entry.result)
        serve_stale = entry is not None and (
            (kind == STATIC and entry.static_gen == _static_gen)
            or (
                kind == DISPLAY
                and now - entry.at <= _DISPLAY_MAX_STALE_S
                and entry.gen == _events.generation()
            )
        )
    if serve_stale:
        with _lock:
            _stats.stale_served += 1
            _stats.background_refreshes += 1
        _start_or_join(key, argv, kind, kwargs, timeout)
        return _copy(entry.result)

    flight = _start_or_join(key, argv, kind, kwargs, timeout)
    first_wait = min(timeout, _SLOW_WAIT_S) if driver_slow() else timeout
    if not flight.done.wait(first_wait):
        if first_wait < timeout:
            answer = _fallback(key, kind)
            if answer is not None:
                return answer
            flight.done.wait(max(timeout - first_wait, 0.0))
    if not flight.done.is_set():
        _mark_slow()
        answer = _fallback(key, kind)
        if answer is not None:
            return answer
        raise subprocess.TimeoutExpired(argv, timeout)
    if isinstance(flight.exc, subprocess.TimeoutExpired):
        answer = _fallback(key, kind)
        if answer is not None:
            return answer
    return _flight_outcome(flight, argv, timeout)
