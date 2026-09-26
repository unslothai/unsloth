# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cached, coalesced nvidia-smi reads for the Studio backend.

Every nvidia-smi call in the backend goes through :func:`run_nvidia_smi`. It returns the
same ``subprocess.CompletedProcess`` (or raises the same ``FileNotFoundError`` /
``OSError`` / ``subprocess.TimeoutExpired``) a direct ``subprocess.run`` would, so callers
keep their parsing and their existing fallbacks. What changes is how often the CLI runs
and how long a caller can be held by it.

Each query is put in one of three categories from its arguments:

``static``
    ``-L``, ``topo -m`` and ``--query-gpu`` lists made only of fields that do not change
    while the driver is loaded (index, uuid, name, memory.total, compute_cap, ...).
    Cached for ``UNSLOTH_GPU_QUERY_STATIC_TTL`` seconds (600) and dropped by
    :func:`invalidate_static` (hardware re-detection).

``display``
    A query carrying live fields (memory.used/free, utilization, power, temperature)
    made inside :func:`display_reads`, which the hardware panel routes use. Fresh for
    ``UNSLOTH_GPU_QUERY_DISPLAY_TTL`` seconds (3), then served stale while one
    background refresh runs.

``critical``
    The same live queries anywhere else: VRAM fit checks, llama.cpp layer placement,
    GPU auto-selection, training memory checks. Fresh for
    ``UNSLOTH_GPU_QUERY_CRITICAL_TTL`` seconds (1), never served stale, never joined to a
    child started more than that long ago, and discarded by :func:`invalidate_gpu_memory`,
    which Studio calls when it loads or unloads a model or starts or stops training.
    Memory other processes allocate is only seen by a new reading, so this 1 s is the
    whole window in which a fit check can miss it (one nvidia-smi run already takes
    ~0.1-0.2 s). Code that polls for a change (VRAM settle loops, fenced readings) opts
    out entirely with :func:`fresh_reads`.

Concurrent identical queries share one child process (single flight), so at most one
nvidia-smi per distinct command line is ever running and a page load no longer starts one
CLI per panel on a driver that is already congested.

Timeouts: ``subprocess.run(timeout=...)`` kills the child and then waits for it without a
deadline, so on a driver whose ioctls block a 5 second timeout turned into 25-100 seconds.
Here the child runs on a daemon thread and the caller waits only its own timeout. A static
or display child may keep running (up to ``UNSLOTH_GPU_QUERY_BACKGROUND_TIMEOUT``, 120 s)
and its answer fills the cache for the next caller; a critical child keeps the caller's
own timeout, since a late answer is of no use to a fit check. When the caller's wait
expires the helper answers, in order, from:

1. the last good answer, for ``static`` (any age) and ``display`` (up to 60 s) only. A
   fit check never gets an old reading: another process may have allocated since;
2. NVML, through ``studio/nvidia_probe.py`` in a child with a deadline, for
   ``--query-gpu`` lists it can answer. This is a new reading (memory.used is reported as
   total - free, which can only understate free memory);
3. otherwise the original ``TimeoutExpired`` is raised and the caller's own fallback (for
   some callers, torch) runs exactly as before.

A non-zero exit, a missing nvidia-smi and any other ``OSError`` are passed straight
through and never cached, so hosts without NVIDIA hardware behave exactly as before.
``UNSLOTH_GPU_QUERY_CACHE=0`` turns all of this off and calls ``subprocess.run`` directly.
"""

from __future__ import annotations

import contextlib
import copy
import contextvars
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

from loggers import get_logger
from utils import gpu_memory_events as _events

logger = get_logger(__name__)

# Bound at import: the worker thread is how a FOREGROUND call enforces its own deadline, not
# a background job, so code that stubs threading.Thread to stop background refreshes (as
# several tests do) must not also stop every nvidia-smi read from completing.
_Thread = threading.Thread

STATIC = "static"
DISPLAY = "display"
CRITICAL = "critical"

# Fields that cannot change while the driver stays loaded. Anything else is live.
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
# How long a caller waits on a driver already known to be slow before taking a fallback.
_SLOW_WAIT_S = 1.0
_NVML_TIMEOUT_S = 5.0


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
    if kind == DISPLAY:
        return _env_float("UNSLOTH_GPU_QUERY_DISPLAY_TTL", 3.0)
    return _env_float("UNSLOTH_GPU_QUERY_CRITICAL_TTL", 1.0)


def _background_timeout() -> float:
    return _env_float("UNSLOTH_GPU_QUERY_BACKGROUND_TIMEOUT", 120.0)


# ── Read modes ────────────────────────────────────────────────────────────────

_display_mode: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unsloth_gpu_query_display", default = False
)
_fresh_mode: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unsloth_gpu_query_fresh", default = False
)


@contextlib.contextmanager
def display_reads() -> Iterator[None]:
    """Live reads inside this block are for display only and may be a few seconds old.

    Context variables follow ``asyncio.to_thread``, so wrapping the ``to_thread`` call in a
    route is enough. Never use this around anything that decides placement or fit."""
    token = _display_mode.set(True)
    try:
        yield
    finally:
        _display_mode.reset(token)


@contextlib.contextmanager
def fresh_reads() -> Iterator[None]:
    """Every live read inside this block runs the CLI, bypassing cache and coalescing.

    For loops that compare consecutive samples (VRAM settle) and for readings paired with
    another measurement under a fence, where a reused sample would read as "stable" or
    pair two different instants. Only the non-blocking timeout still applies."""
    token = _fresh_mode.set(True)
    try:
        yield
    finally:
        _fresh_mode.reset(token)


def classify(argv: Sequence[str]) -> str:
    """The category of an nvidia-smi command line (see the module docstring)."""
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


# ── State ─────────────────────────────────────────────────────────────────────


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
    nvml_served: int = 0
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
    """Studio changed what is resident on the GPUs: no live sample taken before now may
    decide a fit. Cheap, callable from any thread."""
    _events.invalidate_gpu_memory(reason)
    if reason:
        logger.debug("GPU memory readings invalidated: %s", reason)


def invalidate_static(reason: str = "") -> None:
    """Forget the static inventory too (hardware re-detection)."""
    global _static_gen
    _events.invalidate_gpu_memory(reason)
    with _lock:
        _static_gen += 1
    if reason:
        logger.debug("GPU inventory readings invalidated: %s", reason)


def reset() -> None:
    """Drop every cached answer and counter. For tests."""
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


def _copy(result: Any) -> Any:
    """A shallow copy, so one caller rewriting ``stdout`` cannot change what the next one
    reads. Type-agnostic: whatever ``subprocess.run`` returned comes back unchanged."""
    return copy.copy(result)


def _entry_fresh(entry: _Entry, kind: str, now: float) -> bool:
    if now - entry.at > ttl_for(kind):
        return False
    if kind == STATIC:
        return entry.static_gen == _static_gen
    return entry.gen == _events.generation()


# ── Child processes ───────────────────────────────────────────────────────────


def _run_child(flight: _Flight, argv: list, kind: str, kwargs: dict) -> None:
    try:
        with _lock:
            _stats.spawned += 1
        # Looked up at call time so a patched subprocess.run (tests) is honoured.
        result = subprocess.run(argv, **kwargs)
        flight.result = result
        stdout = getattr(result, "stdout", None)
        if getattr(result, "returncode", None) == 0 and isinstance(stdout, str) and stdout.strip():
            with _lock:
                if flight.epoch != _reset_epoch:
                    return
                existing = _cache.get(flight.key)
                # A slow child that began before the current entry's must not replace it.
                if existing is None or existing.started <= flight.started:
                    # Age counts from when the query was issued, not when a slow CLI
                    # finally answered: the numbers describe the earlier moment.
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


def _start_or_join(
    key: tuple, argv: list, kind: str, kwargs: dict, timeout: float
) -> Optional[_Flight]:
    """The in-flight child for this key, or a new one. A non-static caller never joins a
    child started before the last invalidation: it could report pre-load memory.

    None for a critical caller when the child in flight is older than the critical TTL:
    its answer may describe memory from long before the fit check, and starting another
    child behind one the driver is still holding only piles more onto a stuck driver."""
    with _lock:
        flight = _inflight.get(key)
        if flight is not None and (kind == STATIC or flight.gen == _events.generation()):
            if kind == CRITICAL and time.monotonic() - flight.started > ttl_for(CRITICAL):
                return None
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
    child_kwargs["timeout"] = (
        float(timeout) if kind == CRITICAL else max(float(timeout), _background_timeout())
    )
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


# ── Fallbacks when the CLI did not answer in time ─────────────────────────────


def _nvml_rows(timeout: float) -> Optional[list[dict]]:
    if sys.platform == "darwin" or os.environ.get("UNSLOTH_NVIDIA_LIBRARY_PROBE", "1") == "0":
        return None
    try:
        from utils.prebuilt.update_flow import find_installer_script
        script = find_installer_script(
            env_var = "UNSLOTH_NVIDIA_PROBE", script_name = "nvidia_probe.py"
        )
    except Exception:
        return None
    if script is None:
        return None
    try:
        from utils.native_path_leases import child_env_without_native_path_secret
        from utils.subprocess_compat import windows_hidden_subprocess_kwargs

        env = child_env_without_native_path_secret()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        hidden = windows_hidden_subprocess_kwargs()
    except Exception:
        env, hidden = None, {}
    argv = [sys.executable, "-I", str(script), "--json"]
    key = (subprocess.run, "nvml", tuple(argv))
    kwargs = dict(
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        env = env,
        **hidden,
    )
    now = time.monotonic()
    with _lock:
        entry = _cache.get(key)
        cached = (
            entry is not None
            and entry.gen == _events.generation()
            and now - entry.at <= ttl_for(CRITICAL)
        )
    if cached:
        result = entry.result
    else:
        flight = _start_or_join(key, argv, CRITICAL, kwargs, timeout)
        if flight is None or not flight.done.wait(timeout):
            return None
        if flight.exc is not None or flight.result is None:
            return None
        result = flight.result
    try:
        payload = json.loads(getattr(result, "stdout", None) or "null")
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("source") != "nvml":
        return None
    rows = [r for r in payload.get("devices") or [] if isinstance(r, dict) and not r.get("mig")]
    return rows or None


def _nvml_csv(argv: list, timeout: float) -> Optional[str]:
    """Answer a --query-gpu from NVML in nvidia-smi's own CSV shape, or None."""
    fields = _query_fields(argv)
    if not fields:
        return None
    rows = _nvml_rows(timeout)
    if not rows:
        return None
    units = "" if _nounits(argv) else " MiB"
    lines = []
    for row in rows:
        try:
            total = int(row.get("memory_total_mib") or 0)
            free = int(row.get("memory_free_mib") or 0)
        except (TypeError, ValueError):
            return None
        # The probe writes a failed memory read as total 0: no answer beats a wrong one.
        if total <= 0 and any(f.startswith("memory.") for f in fields):
            return None
        cells = []
        for name in fields:
            if name == "index":
                cells.append(str(row.get("index", "")))
            elif name in ("uuid", "gpu_uuid"):
                cells.append(str(row.get("uuid", "")))
            elif name in ("name", "gpu_name"):
                cells.append(str(row.get("name", "")))
            elif name == "compute_cap":
                cells.append(str(row.get("compute_cap") or "[N/A]"))
            elif name == "memory.total":
                cells.append(f"{total}{units}")
            elif name == "memory.free":
                cells.append(f"{free}{units}")
            elif name == "memory.used":
                # NVML v1 used = total - free, which includes the driver reservation that
                # nvidia-smi leaves out: larger, so free derived from it is never higher.
                cells.append(f"{max(total - free, 0)}{units}")
            else:
                cells.append("[N/A]")
        lines.append(", ".join(cells))
    return "\n".join(lines) + "\n"


def _fallback(
    key: tuple, argv: list, kind: str, timeout: float, nvml_fallback: bool
) -> Optional[subprocess.CompletedProcess]:
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
    if nvml_fallback and kind != STATIC:
        stdout = _nvml_csv(argv, min(timeout, _NVML_TIMEOUT_S))
        if stdout is not None:
            with _lock:
                _stats.nvml_served += 1
            logger.info("nvidia-smi did not answer in time; answered from NVML")
            return subprocess.CompletedProcess(list(argv), 0, stdout, "")
    return None


# ── Entry point ───────────────────────────────────────────────────────────────


def run_nvidia_smi(
    argv: Sequence[str],
    *,
    timeout: float,
    kind: Optional[str] = None,
    nvml_fallback: bool = True,
    cache: bool = True,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """Drop-in for ``subprocess.run(argv, timeout=timeout, **kwargs)`` on an nvidia-smi
    command line. See the module docstring for caching, coalescing and fallbacks.

    ``cache=False`` is for a caller that already keeps its own answer and has an explicit
    refresh (the NVLink topology): the CLI always runs, only the bounded wait applies."""
    argv = list(argv)
    if not enabled():
        return subprocess.run(argv, timeout = timeout, **kwargs)
    kind = kind or classify(argv)
    text_mode = bool(
        kwargs.get("text") or kwargs.get("encoding") or kwargs.get("universal_newlines")
    )
    # The runner itself (not its id, which a collected object can hand to the next one) is
    # part of the key: constant in production, and a test that swaps in a different fake
    # CLI mid-test must not be answered by the previous fake.
    key = (subprocess.run, tuple(argv), text_mode)

    if not cache or (_fresh_mode.get() and kind != STATIC):
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
        # No cached answer may stand in here, so on a driver that just timed out give up
        # after the short wait and let the caller's own fallback (NVML, for llama.cpp) run.
        wait = min(timeout, _SLOW_WAIT_S) if driver_slow() else timeout + 1.0
        if not flight.done.wait(wait):
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
        # Stale-while-revalidate: the panel gets the last answer now and one child refreshes it.
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
    if flight is None:
        # A critical read and the CLI has been busy on this query for longer than its TTL:
        # behave as a timeout (a new NVML reading, else the caller's own fallback).
        answer = _fallback(key, argv, kind, timeout, nvml_fallback)
        if answer is not None:
            return answer
        raise subprocess.TimeoutExpired(argv, timeout)
    # On a driver that just timed out, do not hold the caller for the whole timeout when a
    # safe answer already exists; if none does, keep waiting as before.
    first_wait = min(timeout, _SLOW_WAIT_S) if driver_slow() else timeout
    if not flight.done.wait(first_wait):
        if first_wait < timeout:
            answer = _fallback(key, argv, kind, timeout, nvml_fallback)
            if answer is not None:
                return answer
            flight.done.wait(max(timeout - first_wait, 0.0))
    if not flight.done.is_set():
        _mark_slow()
        answer = _fallback(key, argv, kind, timeout, nvml_fallback)
        if answer is not None:
            return answer
        raise subprocess.TimeoutExpired(argv, timeout)
    if isinstance(flight.exc, subprocess.TimeoutExpired):
        answer = _fallback(key, argv, kind, timeout, nvml_fallback)
        if answer is not None:
            return answer
    return _flight_outcome(flight, argv, timeout)
