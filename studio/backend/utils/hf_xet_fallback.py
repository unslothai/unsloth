# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unsloth shim over the shared ``unsloth_zoo.hf_xet_fallback`` Xet -> HTTP stall fallback.

Re-exports the shared API and injects Unsloth's marker-aware cache purge
(``prepare_cache_for_transport``) so the download manager keeps its ``.transport``
marker semantics on the HTTP retry.

Import discipline: ``unsloth_zoo``'s ``__init__`` eagerly imports ``transformers``. The workers
import this shim at startup (to decide the per-worker Xet env flip) *before* activating the model's
``transformers`` sidecar. Activation only prepends the sidecar to ``sys.path``, so a ``transformers``
already cached in ``sys.modules`` (via an eager ``unsloth_zoo`` import here) wins -- pinning the
default 4.57.x and regressing Qwen3.5 / GLM-4.7 / gemma-4 training with
``Tokenizer class TokenizersBackend does not exist``. So the shared backend is loaded **lazily**
(``_load_shared``), only on first use of a heavy download helper, i.e. after the sidecar is active.
``child_should_disable_xet`` and the ``DEFAULT_*`` constants are defined locally so importing them
never triggers the heavy load.
"""

from __future__ import annotations

import os
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

# Defaults mirror unsloth_zoo.hf_xet_fallback; plain literals so they resolve (including as
# default args below) without importing unsloth_zoo/transformers.
DEFAULT_GRACE_PERIOD = 10.0
DEFAULT_HEARTBEAT_INTERVAL = 30.0
# Xet gets 30s of zero progress before the HTTP retry; HTTP, the last resort, keeps 180s. The
# wrappers pass None so the shared layer picks per transport; these literals are for callers that
# want an explicit value without the heavy import.
DEFAULT_STALL_TIMEOUT = 30.0
DEFAULT_CONNECT_TIMEOUT = 90.0
DEFAULT_HTTP_STALL_TIMEOUT = 180.0
# Xet workers spent per download before the transport changes. A wedged transfer usually clears on a
# fresh process, and the retry replays only the in-flight file: the worker runs
# snapshot_download(max_workers=1), so every finished shard is already a blob and is skipped.
DEFAULT_XET_ATTEMPTS = 2

# --- lazy shared-backend loader ----------------------------------------------------------------
_shared: Any = None
_shared_available: Optional[bool] = None  # None = not yet attempted
_shared_import_error: Optional[BaseException] = None
# Guards _shared_available AND every UNSLOTH_ZOO_DISABLE_GPU_INIT save/set/restore here. Both
# loaders mutate that one process-wide variable, so they must serialize against each other: two
# locks would still allow A-saves-unset / B-saves-"1" / A-restores-unset / B-restores-"1", leaving
# it set for the life of the process. RLock because child_environment_for_spawn holds it across a
# spawn and legitimately nests (its own _spawn_env_lock is an RLock for the same reason).
_load_lock = threading.RLock()


def _gpu_present() -> bool:
    """Whether this host has a usable accelerator, decided WITHOUT importing unsloth_zoo.

    Only torch is consulted (already imported by the time any download helper runs), and any
    failure answers False so a genuinely torch-less host keeps the light-init retry below.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 -- no torch at all: the light path is the right one
        return False
    for probe in (
        lambda: torch.cuda.is_available(),
        lambda: torch.backends.mps.is_available(),
        lambda: torch.xpu.is_available(),
    ):
        try:
            if probe():
                return True
        except Exception:  # noqa: BLE001 -- a missing backend is just "not this one"
            continue
    return False


def _load_shared() -> bool:
    """Import ``unsloth_zoo.hf_xet_fallback`` on demand; return True if available. Deferred so
    importing this module at worker startup does not pull transformers in before the sidecar is
    activated. Degrades (returns False) rather than crashing when unsloth_zoo is unavailable."""
    global _shared, _shared_available, _shared_import_error
    if _shared_available is not None:
        return _shared_available
    with _load_lock:
        if _shared_available is not None:
            return _shared_available
        try:
            import unsloth_zoo.hf_xet_fallback as shared

            _shared = shared
            _shared_available = True
            _shared_import_error = None
            return True
        except Exception as exc:  # noqa: BLE001 - any import failure must degrade, not crash
            # unsloth_zoo's __init__ runs torch/GPU detection, which raises on a torch-less/GPU-less
            # host. The download helper needs none of it, so retry via UNSLOTH_ZOO_DISABLE_GPU_INIT.
            _shared_import_error = exc
            import os as _os

            # ...but ONLY on a host that really has no accelerator. That flag makes unsloth_zoo take its MLX/CPU path, injecting triton and bitsandbytes
            # STUBS into sys.modules for the process. On a working GPU box those stubs raise from the first CUDA-only kernel, turning a healthy GPU into 500s.
            if _gpu_present():
                _shared_available = False
                import logging as _logging

                _logging.getLogger(__name__).warning(
                    "unsloth_zoo.hf_xet_fallback unavailable (%s); the Xet stall watchdog is "
                    "disabled. Not retrying under UNSLOTH_ZOO_DISABLE_GPU_INIT because this host "
                    "has an accelerator and that path would stub out triton/bitsandbytes for the "
                    "whole process.",
                    exc,
                )
                return False

            global _gpu_init_override_depth
            _prev_gpu_init = _os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT")
            _ours = _prev_gpu_init != "1"
            _gpu_init_override_depth += _ours  # claimed before the write, released after
            _os.environ["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
            try:
                import unsloth_zoo.hf_xet_fallback as shared

                _shared = shared
                _shared_available = True
                _shared_import_error = None
                return True
            except Exception as exc2:  # noqa: BLE001 - degrade so Unsloth still boots with plain HF
                _shared_import_error = exc2
                _shared_available = False
                import logging as _logging

                _logging.getLogger(__name__).warning(
                    "unsloth_zoo.hf_xet_fallback unavailable (%s); the Xet stall watchdog is "
                    "disabled. Install/upgrade unsloth_zoo (and its torch dependency) to "
                    "re-enable automatic Xet -> HTTP download recovery.",
                    _shared_import_error,
                )
                return False
            finally:
                if _prev_gpu_init is None:
                    _os.environ.pop("UNSLOTH_ZOO_DISABLE_GPU_INIT", None)
                else:
                    _os.environ["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = _prev_gpu_init
                _gpu_init_override_depth -= _ours


# _load_optional results by module name. Memoising the FAILURE is the point: on a zoo that predates
# these modules the import can never start succeeding, so without this every xet_health /
# record_xet_outcome / xet_env_overrides call re-ran the GPU-init retry, re-opening the
# process-wide env window on every download. With it the window opens once per module per process.
_UNTRIED = object()
_optional_modules: "dict[str, Any]" = {}


def _reset_optional_module_cache() -> None:
    """Forget memoised optional-module results (tests that install or remove a zoo module)."""
    with _load_lock:
        _optional_modules.clear()


def _load_optional(module_name: str) -> Any:
    """Import an optional shared Xet helper module (health / tuning), or return ``None``.

    Separate from ``_load_shared``: these modules exist only in newer unsloth_zoo, and an Unsloth
    pinned to an older one must keep downloading without the preflight verdict or buffer caps.
    The GPU-init retry matters most here: ``unsloth_zoo.__init__`` runs torch accelerator detection
    and raises ``NotImplementedError`` on a CPU-only host, which is precisely the small machine
    whose RAM these caps protect, so without the retry they switch off where they are needed.
    """
    import importlib
    import os as _os

    cached = _optional_modules.get(module_name, _UNTRIED)
    if cached is not _UNTRIED:
        return cached

    try:
        module = importlib.import_module(module_name)
        _optional_modules[module_name] = module
        return module
    except Exception as exc:  # noqa: BLE001 - an older/absent unsloth_zoo must degrade, not crash
        first_error = exc

    # Deliberately the SAME lock _load_shared uses: interleaved save/set/restore would leave
    # UNSLOTH_ZOO_DISABLE_GPU_INIT set for the life of the process (see _load_lock).
    with _load_lock:
        cached = _optional_modules.get(module_name, _UNTRIED)
        if cached is not _UNTRIED:
            return cached
        global _gpu_init_override_depth
        previous = _os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT")
        ours = previous != "1"
        # Claim BEFORE the write, release AFTER the restore, so the set window sits strictly inside
        # the window where a spawning thread can see the flag is ours. The other order leaves a gap
        # at each end where a child inherits an unclaimed flag and never clears it.
        _gpu_init_override_depth += ours
        try:
            _os.environ["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001
                import logging as _logging
                _logging.getLogger(__name__).debug(
                    "%s unavailable (%s; with GPU init disabled: %s)", module_name, first_error, exc
                )
                module = None
            finally:
                if previous is None:
                    _os.environ.pop("UNSLOTH_ZOO_DISABLE_GPU_INIT", None)
                else:
                    _os.environ["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = previous
        finally:
            _gpu_init_override_depth -= ours
        _optional_modules[module_name] = module
        return module


def _xet_health_from(module: Any, **kwargs: Any) -> Any:
    if module is None:
        return None
    try:
        return module.xet_health(**kwargs)
    except Exception as exc:  # noqa: BLE001
        import logging as _logging
        _logging.getLogger(__name__).debug("xet_health failed: %s", exc)
        return None


def cached_xet_health(**kwargs: Any) -> Any:
    """Return Zoo's Xet verdict only when its health module is already loaded.

    Capability reads use this path so opening Hub cannot initialize Unsloth Zoo. A real
    download calls :func:`xet_health`, which loads the optional module and populates this cache.
    """
    with _load_lock:
        module = _optional_modules.get("unsloth_zoo.hf_xet_health", _UNTRIED)
    return None if module is _UNTRIED else _xet_health_from(module, **kwargs)


def xet_health(**kwargs: Any) -> Any:
    """Load and query Zoo's Xet verdict for an actual download decision.

    ``None`` means "no opinion": callers keep their default (Xet), they do not downgrade.
    Read-only capability requests use :func:`cached_xet_health` instead.
    """
    module = _load_optional("unsloth_zoo.hf_xet_health")
    return _xet_health_from(module, **kwargs)


def xet_health_is_forced(health: Any) -> bool:
    """Is *health* an operator override rather than a measurement of this machine?

    ``unsloth_zoo.hf_xet_health`` stamps ``source = "forced"`` on exactly the two env-var verdicts:
    ``UNSLOTH_DISABLE_XET`` / ``UNSLOTH_STABLE_DOWNLOADS`` / ``HF_HUB_DISABLE_XET`` turning Xet OFF,
    and ``UNSLOTH_FORCE_XET`` turning it ON. Callers already honour the off switches by returning
    early, so this exists for the on switch: the free-RAM gate must stand down for it, or Unsloth
    ships an escape hatch that only works in one direction.

    Anything unreadable (an older zoo whose verdict has no ``source``, a test double) answers False,
    which leaves the RAM gate in force -- the safe default."""
    return health is not None and str(getattr(health, "source", "")) == "forced"


def record_xet_outcome(ok: bool, reason: str = "") -> None:
    """Record a finished Xet attempt so a repeatedly-failing machine stops starting on Xet."""
    module = _load_optional("unsloth_zoo.hf_xet_health")
    if module is None:
        return
    try:
        module.record_xet_outcome(ok, reason)
    except Exception as exc:  # noqa: BLE001
        import logging as _logging
        _logging.getLogger(__name__).debug("record_xet_outcome failed: %s", exc)


def xet_env_overrides() -> "dict[str, str]":
    """RAM/CPU-derived ``HF_XET_*`` caps for a download worker's environment; ``{}`` if unavailable."""
    module = _load_optional("unsloth_zoo.hf_xet_tuning")
    if module is None:
        return {}
    try:
        return dict(module.xet_env_overrides())
    except Exception as exc:  # noqa: BLE001
        import logging as _logging
        _logging.getLogger(__name__).debug("xet_env_overrides failed: %s", exc)
        return {}


def apply_xet_env(env: dict, cache_dir: "Optional[str]" = None) -> "Optional[dict[str, str]]":
    """Let unsloth_zoo size a download worker's ``HF_XET_*`` in *env*, in place.

    Returns what it wrote, or ``None`` when the installed zoo has no opinion, which is the caller's
    signal to fall back. ``fail_fast`` suits a supervised child: our Xet -> HTTP ladder acts on the
    failure, so short Xet timeouts are right here and wrong process-wide.

    *env* is a copy of this process's environment, which already carries the zoo's import-time
    sizing, and applying is setdefault: on a zoo that can resize we recompute for *cache_dir*
    instead, so a backend whose cache has since moved does not hand the worker the old volume's
    numbers. Older zoos keep the previous behaviour.

    The zoo sizes from TOTAL RAM, which cannot see a model already loaded, so the result passes
    through :func:`clamp_to_available_ram` before it reaches the worker."""
    module = _load_optional("unsloth_zoo.hf_xet_tuning")
    if module is None or not hasattr(module, "apply_xet_env"):
        return None
    try:
        resize = getattr(module, "resize_for_cache_dir", None)
        if resize is not None:
            sized = dict(resize(env, cache_dir))
        else:
            sized = dict(module.apply_xet_env(env, fail_fast = True))
    except Exception as exc:  # noqa: BLE001
        import logging as _logging
        _logging.getLogger(__name__).debug("apply_xet_env failed: %s", exc)
        return None
    return clamp_to_available_ram(env, sized, cache_dir = cache_dir, module = module)


# Share of free RAM a download may turn into buffers. A quarter of AVAILABLE always exceeds the
# zoo's eighth of TOTAL on an idle machine, so the clamp is unreachable unless RAM is actually held.
_AVAILABLE_RAM_SHARE = 4
# Integer arithmetic converges in one or two passes; the bound only guards a future non-monotonic zoo.
_CLAMP_MAX_PASSES = 3
_BUFFER_LIMIT_KEY = "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"


def _as_int(value: str) -> "Optional[int]":
    """``value`` as a plain int, or None for the unit-suffixed ones ("60s") that never scale."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


# --- concurrent-worker budget ledger -------------------------------------------------------------
# A worker allocates inside the child, after Popen returns, so free RAM does not drop until well
# after we sized it. Four downloads starting together would each read the same untouched `available`
# and each take a quarter of it, promising the whole machine. Reservations bridge that window:
# sizing subtracts what live siblings were already promised but have not yet taken. Only the
# unmaterialized remainder, because once a worker's buffers are resident `available` has ALREADY
# dropped by them: charging the whole promise on top of that reading counts the same bytes twice,
# for the worker's entire lifetime, and talks the next download out of RAM that is genuinely free.
# RLock: the clamp holds this across its whole decide-and-reserve region, and the reserve retakes it.
_budget_lock = threading.RLock()
# token -> [bytes, pid or None, monotonic stamp]
_budget_reservations: "dict[int, list]" = {}
_budget_token_seq = 0
# A reservation never bound to a pid means the spawn died between sizing and Popen.
_UNBOUND_RESERVATION_TTL = 60.0
# Backstop against pid reuse keeping a dead reservation alive; no download worker outlives this.
_BOUND_RESERVATION_TTL = 12 * 60 * 60.0
# Set by the sizing call, consumed by the spawn that follows it on the SAME thread.
_pending_reservation = threading.local()


def _pid_alive(pid: int) -> bool:
    """Is *pid* still running? Platform-aware, because this probe must not have side effects.

    NOT ``os.kill(pid, 0)``: on Windows CPython maps every signal other than ``CTRL_C_EVENT`` /
    ``CTRL_BREAK_EVENT`` onto ``TerminateProcess(handle, sig)``, so signal 0 would KILL the download
    worker this ledger is merely asking about. ``utils.process_lifetime`` already carries the
    handle-based probe (``OpenProcess`` + ``WaitForSingleObject``); reuse it rather than growing a
    second copy that can drift."""
    try:
        from utils.process_lifetime import _pid_alive as _platform_pid_alive
        return bool(_platform_pid_alive(pid))
    except Exception:  # noqa: BLE001 - fall through to the POSIX probe below
        pass
    if os.name == "nt":
        # No platform probe available: assume alive rather than reach for os.kill, so a reservation
        # is at worst held too long instead of a running download being terminated.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _worker_rss(pid: int) -> int:
    """Physical RAM *pid* already holds, or ``0`` when it cannot be read.

    ``rss`` is psutil's portable field: RES on Linux, resident size on macOS, ``WorkingSetSize`` on
    Windows. All three are physical pages, which is the same quantity ``virtual_memory().available``
    has already been reduced by, so it is the right thing to credit against a promise.

    Zero on any failure (no psutil, the worker exited between the liveness probe and here, Windows
    ``AccessDenied``), which reserves the whole promise -- the conservative pre-credit behaviour."""
    try:
        import psutil  # noqa: PLC0415 - optional, and only on the ledger path
        return max(0, int(psutil.Process(pid).memory_info().rss))
    except Exception:  # noqa: BLE001 - an unreadable worker is not evidence it allocated nothing
        return 0


def _live_reserved_locked() -> int:
    """Bytes promised to live workers that are NOT YET RESIDENT, pruning anything finished or never
    spawned.

    A promise covers the gap between sizing and allocation. The hf_xet buffer is an adjustable
    semaphore, so the worker draws on it as terms arrive rather than allocating it up front; every
    byte it has drawn is already missing from ``available``. Subtracting the whole promise from that
    reading charges the resident part a second time, which is why the credit is capped at the
    promise: a fully materialized worker contributes nothing further, a freshly spawned one still
    contributes all of it."""
    now = time.monotonic()
    total = 0
    for token, entry in list(_budget_reservations.items()):
        nbytes, pid, stamp = entry
        if pid is None:
            if now - stamp > _UNBOUND_RESERVATION_TTL:
                _budget_reservations.pop(token, None)
            else:
                total += nbytes  # nothing spawned yet, so nothing of it is resident
            continue
        if now - stamp > _BOUND_RESERVATION_TTL or not _pid_alive(pid):
            _budget_reservations.pop(token, None)
            continue
        total += max(0, nbytes - _worker_rss(pid))
    return total


def _reserve_worker_budget(nbytes: int) -> None:
    """Hold *nbytes* against this thread's imminent spawn, replacing any reservation it still owns
    (a retried sizing must not stack)."""
    global _budget_token_seq
    with _budget_lock:
        stale = getattr(_pending_reservation, "token", None)
        if stale is not None:
            _budget_reservations.pop(stale, None)
        _budget_token_seq += 1
        token = _budget_token_seq
        _budget_reservations[token] = [max(0, int(nbytes)), None, time.monotonic()]
    _pending_reservation.token = token


def bind_worker_budget(pid: "Optional[int]") -> None:
    """Attach the reservation this thread just made to *pid*, so it frees when the worker exits.

    ``None`` drops it, for a spawn that never produced a process."""
    token = getattr(_pending_reservation, "token", None)
    _pending_reservation.token = None
    if token is None:
        return
    with _budget_lock:
        entry = _budget_reservations.get(token)
        if entry is None:
            return
        if pid is None:
            _budget_reservations.pop(token, None)
        else:
            entry[1], entry[2] = int(pid), time.monotonic()


def clamp_to_available_ram(
    env: dict,
    sized: "dict[str, str]",
    *,
    cache_dir: "Optional[str]" = None,
    module: Any = None,
) -> "dict[str, str]":
    """Shrink a zoo-sized ``HF_XET_*`` budget that free RAM cannot afford. Returns what *env* holds.

    hf_xet's reconstruction buffers are the worker's RSS, not reclaimable page cache. Sized from
    total RAM, a download started while a 27B GGUF is resident asks for the same multi-GB budget it
    would on an idle box, and the two together are the swap (issue #9032).

    A clamp, not a second sizing formula: the zoo keeps deciding, this only hands it a smaller
    machine, so the two cannot drift. Three properties:

    - Free when there is headroom: a budget that fits returns untouched.
    - Only keys the zoo wrote are rewritten, so an explicit user setting survives. A user-set
      ``HF_XET_HIGH_PERFORMANCE`` makes the zoo drop its caps, leaving no budget key to clamp.
    - Unmeasurable RAM, or a zoo too old to report it, leaves the download alone.

    Whatever budget ends up in force is reserved for this thread's imminent spawn, so siblings
    starting in the same window size against the remainder instead of against the same snapshot.
    ``bind_worker_budget`` ties that reservation to the worker's pid.
    """
    if module is None:
        module = _load_optional("unsloth_zoo.hf_xet_tuning")
    overrides = getattr(module, "xet_env_overrides", None)
    profile_of = getattr(module, "system_profile", None)
    if overrides is None or profile_of is None or _BUFFER_LIMIT_KEY not in sized:
        return sized
    try:
        import dataclasses

        profile = profile_of(cache_dir)
        available = int(getattr(profile, "available_ram_bytes", 0) or 0)
        total = int(getattr(profile, "total_ram_bytes", 0) or 0)
        if available <= 0 or total <= 0:
            return sized
        floor = int(getattr(module, "_MIN_BUFFER_LIMIT", 1_000_000_000))
        limit = int(sized[_BUFFER_LIMIT_KEY])
        # Reading the ledger and reserving against it is ONE decision. Split across two critical
        # sections, concurrent workers all read the same total before any of them wrote, which is
        # the very overcommit the ledger exists to stop. The recompute inside is pure arithmetic on
        # a frozen profile, so holding the lock across it costs microseconds; the RAM/disk reading
        # above stays outside. `_budget_lock` is an RLock because `_reserve_worker_budget` retakes
        # it here.
        with _budget_lock:
            unclaimed = max(0, available - _live_reserved_locked())
            budget = max(floor, unclaimed // _AVAILABLE_RAM_SHARE)
            if limit <= budget:
                # Still reserved: four unclamped workers would otherwise promise four full budgets.
                _reserve_worker_budget(limit)
                return sized

            # Re-ask the zoo about a machine the download can afford, so buffer, per-file and file
            # count all scale together instead of the limit moving on its own.
            fraction = int(getattr(module, "_RAM_FRACTION", 8)) or 8
            synthetic = max(floor, budget * fraction)
            clamped = sized
            for _ in range(_CLAMP_MAX_PASSES):
                candidate = dict(
                    overrides(
                        dataclasses.replace(
                            profile,
                            total_ram_bytes = min(total, synthetic),
                            available_ram_bytes = available,
                        ),
                        fail_fast = True,
                    )
                )
                clamped = candidate
                new_limit = int(candidate[_BUFFER_LIMIT_KEY])
                if new_limit <= budget:
                    break
                # Monotonic in total RAM, so scaling by the overshoot converges.
                synthetic = max(floor, synthetic * budget // new_limit)

            # Reduce-only: keep a value the recompute would RAISE. `xet_env_overrides` is called raw
            # here, without the throttled flag `apply_xet_env` threads through after a 429, so an
            # un-throttled recompute could otherwise hand back the stream ceiling that backoff
            # lowered. Every derived number is monotonic in total RAM, so taking the smaller of the
            # two is always a coherent config.
            written = {}
            for key, value in clamped.items():
                if key not in sized:
                    continue
                before, after = _as_int(sized[key]), _as_int(value)
                written[key] = (
                    sized[key]
                    if before is not None and after is not None and after > before
                    else value
                )
            env.update(written)
            effective = _as_int(written.get(_BUFFER_LIMIT_KEY, "")) or budget
            _reserve_worker_budget(effective)
        import logging as _logging

        _logging.getLogger(__name__).info(
            "Xet download buffers clamped to free RAM: %.2fGB -> %.2fGB "
            "(%.1fGB free of %.1fGB total, %.2fGB promised to running downloads and not yet taken)",
            limit / 1e9,
            effective / 1e9,
            available / 1e9,
            total / 1e9,
            (available - unclaimed) / 1e9,
        )
        return written
    except Exception as exc:  # noqa: BLE001 - a clamp must never be what breaks a download
        import logging as _logging
        _logging.getLogger(__name__).debug("clamp_to_available_ram failed: %s", exc)
        return sized


def available_ram_bytes() -> "tuple[Optional[int], int]":
    """``(free RAM right now, the floor Xet wants)``; ``(None, floor)`` when RAM is unmeasurable.

    Both numbers are the zoo's. It just compares its floor against TOTAL RAM, which cannot see a
    loaded model; exposing them here lets the transport choice apply the same rule to free RAM."""
    module = _load_optional("unsloth_zoo.hf_xet_tuning")
    floor = int(getattr(module, "MIN_XET_RAM_BYTES", 4_000_000_000) or 4_000_000_000)
    profile_of = getattr(module, "system_profile", None)
    if profile_of is None:
        return (None, floor)
    try:
        available = int(getattr(profile_of(), "available_ram_bytes", 0) or 0)
    except Exception as exc:  # noqa: BLE001
        import logging as _logging
        _logging.getLogger(__name__).debug("available_ram_bytes failed: %s", exc)
        return (None, floor)
    return (available if available > 0 else None, floor)


def free_ram_pressure_reason() -> "Optional[str]":
    """Why a download should take HTTP right now, or ``None`` to leave Xet alone.

    The zoo refuses Xet below ``MIN_XET_RAM_BYTES`` but measures TOTAL RAM, so the check passes on a
    32 GB box down to 2 GB free because a 27B GGUF is loaded (issue #9032). Same rule and threshold,
    asked of free RAM. Buffers are clamped separately; this catches the host where even the clamped
    floor will not fit.

    One rule with two callers, which must agree: the capabilities probe resolves what the UI submits
    as an explicit transport, and ``resolve_auto_use_xet`` covers an API caller that sends "auto".
    Unmeasurable RAM is not evidence of pressure, so anything unreadable keeps Xet.

    RAM promised to running downloads but not yet resident is subtracted, so the Nth concurrent
    download is sent to HTTP rather than handed Xet's floor. The clamp alone cannot bound that: its
    budget bottoms out at the floor, so enough simultaneous workers would still add up past free
    RAM. Only the unclaimed remainder, since whatever a worker has already taken is missing from
    this reading already (see ``_live_reserved_locked``)."""
    try:
        available, floor = available_ram_bytes()
        if available is not None:
            with _budget_lock:
                available = max(0, available - _live_reserved_locked())
    except Exception as exc:  # noqa: BLE001 - a probe must not decide the transport by crashing
        import logging as _logging
        _logging.getLogger(__name__).debug("free_ram_pressure_reason failed: %s", exc)
        return None
    if available is None or available >= floor:
        return None
    return (
        f"HTTP: only {available / 1e9:.1f}GB RAM free (Xet wants {floor / 1e9:.0f}GB); "
        "close a loaded model or wait for running downloads to use Xet"
    )


def child_should_disable_xet(config: dict) -> bool:
    """Single source of truth for the per-worker Xet env flip (mirrors
    ``unsloth_zoo.hf_xet_fallback.child_should_disable_xet``). Deliberately lightweight: importing or
    calling it must NOT pull in unsloth_zoo/transformers, so the worker can decide before activating
    the transformers sidecar (see the module docstring)."""
    return bool(config.get("disable_xet"))


def is_data_phase_stall(message: str) -> bool:
    """Whether a watchdog verdict fired AFTER bytes had flowed (mirrors
    ``unsloth_zoo.hf_xet_fallback.is_data_phase_stall``).

    "did not start" is the pre-first-byte trip, as likely slow metadata or a cache lock as a broken
    Xet; the others mean the transfer moved and then wedged, which a fresh worker recovers from. The
    lifecycle decides both whether to spend another Xet worker and whether to charge a health
    failure on this one rule, so the two cannot disagree. Local for the same reason as
    ``child_should_disable_xet``: the stall path must not depend on the heavy import."""
    return "did not start" not in (message or "")


def xet_attempts() -> int:
    """Xet workers a download may spend before HTTP (mirrors
    ``unsloth_zoo.hf_xet_fallback.xet_attempts``): ``UNSLOTH_XET_ATTEMPTS``, default 2, clamped to 8;
    junk or non-positive falls back to the default. ``1`` restores the straight-to-HTTP ladder."""
    raw = os.environ.get("UNSLOTH_XET_ATTEMPTS")
    if not raw:
        return DEFAULT_XET_ATTEMPTS
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_XET_ATTEMPTS
    if value <= 0:
        return DEFAULT_XET_ATTEMPTS
    return min(value, 8)


# --- degraded stubs (used only when unsloth_zoo is unavailable) -------------------------------
class _DegradedDownloadStallError(RuntimeError):
    """Stub mirror so callers' ``except`` clauses resolve; never raised in degraded mode."""


def _degraded_get_hf_download_state(*args: Any, **kwargs: Any) -> None:
    return None  # unmeasurable -> the (absent) watchdog never fires


def _degraded_start_watchdog(
    *,
    on_heartbeat: "Optional[Callable[[str], None]]" = None,
    interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    xet_disabled: bool = False,
    **kwargs: Any,
) -> "threading.Event":
    # No stall detection, but keep emitting heartbeats so the orchestrator's inactivity deadline
    # is not tripped during a long download.
    stop = threading.Event()
    if on_heartbeat is None:
        return stop
    transport = "https" if xet_disabled else "xet"

    def _beat() -> None:
        while not stop.wait(interval):
            try:
                on_heartbeat(f"Downloading ({transport} transport)...")
            except Exception:
                pass

    threading.Thread(
        target = _beat,
        daemon = True,
        name = "hf-xet-degraded-heartbeat",
    ).start()
    return stop


def _degraded_cancelled(cancel_event: "Optional[threading.Event]") -> bool:
    return cancel_event is not None and cancel_event.is_set()


def _degraded_hf_hub_download_with_xet_fallback(
    repo_id: str,
    filename: str,
    token: Optional[str],
    *,
    repo_type: str = "model",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    cancel_event: "Optional[threading.Event]" = None,
    **_ignored: Any,
) -> str:
    # Keep the cancellation contract: do not start or return a download once cancelled.
    if _degraded_cancelled(cancel_event):
        raise RuntimeError("Cancelled")

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id = repo_id,
        filename = filename,
        token = token,
        repo_type = repo_type,
        revision = revision,
        cache_dir = cache_dir,
        force_download = force_download,
    )
    if _degraded_cancelled(cancel_event):
        raise RuntimeError("Cancelled")
    return path


def _degraded_snapshot_download_with_xet_fallback(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    repo_type: str = "model",
    cache_dir: Optional[str] = None,
    allow_patterns: Optional[Any] = None,
    ignore_patterns: Optional[Any] = None,
    force_download: bool = False,
    cancel_event: "Optional[threading.Event]" = None,
    **_ignored: Any,
) -> str:
    if _degraded_cancelled(cancel_event):
        raise RuntimeError("Cancelled")

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id = repo_id,
        repo_type = repo_type,
        revision = revision,
        token = token,
        cache_dir = cache_dir,
        allow_patterns = allow_patterns,
        ignore_patterns = ignore_patterns,
        force_download = force_download,
    )
    if _degraded_cancelled(cancel_event):
        raise RuntimeError("Cancelled")
    return path


# --- lazy attribute access for the heavy shared API -------------------------------------------
# ``DownloadStallError`` (class identity matters for ``except``), ``start_watchdog`` and
# ``get_hf_download_state`` come from the shared backend when available, else the degraded stubs.
# Resolved via PEP 562 ``__getattr__`` so ``from utils.hf_xet_fallback import X`` triggers the load
# only for these heavy names, not for ``child_should_disable_xet`` / ``DEFAULT_*``.
_DEGRADED_ATTRS = {
    "DownloadStallError": _DegradedDownloadStallError,
    "get_hf_download_state": _degraded_get_hf_download_state,
}


# Nonzero while a loader has UNSLOTH_ZOO_DISABLE_GPU_INIT set process-wide for its retry. Read by
# utf8_child_env so a child spawned in that window does not inherit it: unsloth_zoo injects triton
# and bitsandbytes STUBS when it is set, so a training child would silently run against no-ops.
# Only counted when the loader introduced the value; an operator who exported it keeps it.
_gpu_init_override_depth = 0


def gpu_init_override_active() -> bool:
    """Is a loader currently holding UNSLOTH_ZOO_DISABLE_GPU_INIT set for its own import?"""
    return _gpu_init_override_depth > 0


def env_override_barrier() -> Any:
    """Context manager a caller holds across a spawn so no loader can be mid-override.

    Spawn children inherit the parent's live ``os.environ`` and there is no env dict to filter, so
    the only way to keep UNSLOTH_ZOO_DISABLE_GPU_INIT out of a worker is that no loader has it set
    when the child is created. Loaders never spawn, so holding this with the spawn lock cannot
    deadlock, and ``_load_optional`` memoises so the window opens at most once per module per
    process.
    """
    return _load_lock


def _supported_kwargs(fn: Any, kwargs: "dict[str, Any]") -> "dict[str, Any]":
    """Drop kwargs *fn* does not accept; pass everything through if it takes ``**kwargs``.

    Uninspectable callables (C functions, some test doubles) also pass through unchanged.
    """
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def start_watchdog(**kwargs: Any) -> Any:
    """Shared stall watchdog, minus any kwarg the INSTALLED unsloth_zoo does not accept.

    Load-bearing version-skew adapter: the supported floor (2026.8.1) has no ``connect_timeout`` or
    ``heartbeat_interval`` and no ``**kwargs``, so passing one raises TypeError into the caller's
    ``except Exception`` -- the watchdog then never starts and a stalled Xet worker is never killed
    or retried over HTTP. That is the feature entirely off, not degraded. Filtering keeps newer
    knobs live on a newer zoo and makes the NEXT new kwarg a no-op instead of a repeat of this bug.

    Dropping the pre-byte budget on 2026.8.1 costs little: huggingface_hub opens the ``.incomplete``
    BEFORE calling ``xet_get`` (``file_download.py`` opens ``incomplete_path`` and calls ``xet_get``
    inside that ``with``) and the floor counts a partial by presence, not size, so a hf_xet hang
    still trips the floor's 180s data clock. Verified against the released wheel: wedged inside
    ``xet_get`` trips, wedged before the open does not. The uncovered window is the metadata phase,
    where ``snapshot_download`` calls ``repo_info`` with no timeout. That gap predates this shim;
    the connect clock closes it only once a zoo carrying it ships, and passing the kwarg early would
    not close it, it would disable the watchdog outright.
    """
    impl = _shared.start_watchdog if _load_shared() else _degraded_start_watchdog
    return impl(**_supported_kwargs(impl, kwargs))


# Annotation-only declarations for the three names above: they bind NO value, so lookup still misses
# and PEP 562 ``__getattr__`` resolves them lazily -- but ruff/pyflakes see them as defined, so listing
# them in ``__all__`` does not trip F822 (while F822 still catches a real typo elsewhere in the list).
DownloadStallError: type
get_hf_download_state: Any


def __getattr__(name: str) -> Any:
    if name in _DEGRADED_ATTRS:
        if _load_shared():
            return getattr(_shared, name)
        return _DEGRADED_ATTRS[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Indirection seam the public wrappers call (and tests monkeypatch): lazy-load the shared backend,
# then dispatch to it or the degraded stub. The ``_shared_*`` names preserve the pre-refactor contract.
def _shared_hf_hub_download_with_xet_fallback(*args: Any, **kwargs: Any) -> str:
    impl = (
        _shared.hf_hub_download_with_xet_fallback
        if _load_shared()
        else _degraded_hf_hub_download_with_xet_fallback
    )
    return impl(*args, **kwargs)


def _shared_snapshot_download_with_xet_fallback(*args: Any, **kwargs: Any) -> str:
    impl = (
        _shared.snapshot_download_with_xet_fallback
        if _load_shared()
        else _degraded_snapshot_download_with_xet_fallback
    )
    return impl(*args, **kwargs)


__all__ = [
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_GRACE_PERIOD",
    "DEFAULT_HEARTBEAT_INTERVAL",
    "DEFAULT_HTTP_STALL_TIMEOUT",
    "DEFAULT_STALL_TIMEOUT",
    "DEFAULT_XET_ATTEMPTS",
    "DownloadStallError",
    "child_should_disable_xet",
    "cached_xet_health",
    "is_data_phase_stall",
    "xet_attempts",
    "get_hf_download_state",
    "record_xet_outcome",
    "start_watchdog",
    "xet_env_overrides",
    "apply_xet_env",
    "clamp_to_available_ram",
    "available_ram_bytes",
    "free_ram_pressure_reason",
    "bind_worker_budget",
    "xet_health",
    "xet_health_is_forced",
    "hf_hub_download_with_xet_fallback",
    "snapshot_download_with_xet_fallback",
]


def _studio_prepare_for_http(
    repo_type: str,
    repo_id: str,
    *,
    cache_dir: Optional[str] = None,
) -> None:
    """Unsloth's marker-aware purge before an HTTP resume, keeping the download manager's ``.transport``
    accounting consistent (vs unsloth_zoo's generic default). Guarded: a purge failure is logged,
    not fatal to the retry."""
    try:
        from hub.utils.download_registry import prepare_cache_for_transport
        prepare_cache_for_transport(
            repo_type,
            repo_id,
            "http",
            root = Path(cache_dir) if cache_dir else None,
        )
    except Exception as exc:
        try:
            from loggers import get_logger
            get_logger(__name__).debug(
                "Unsloth prepare_cache_for_transport failed for %s: %s", repo_id, exc
            )
        except ModuleNotFoundError as logger_exc:
            if logger_exc.name != "loggers":
                raise


def hf_hub_download_with_xet_fallback(
    repo_id: str,
    filename: str,
    token: Optional[str],
    *,
    cancel_event: Optional[threading.Event] = None,
    repo_type: str = "model",
    revision: Optional[str] = None,
    stall_timeout: Optional[float] = None,
    interval: Optional[float] = None,
    grace_period: float = DEFAULT_GRACE_PERIOD,
    on_status: Optional[Callable[[str], None]] = None,
    force_download: bool = False,
    cache_dir: Optional[str] = None,
    reuse_other_cache_root: bool = False,
    local_files_only: bool = False,
) -> str:
    """Single-file download via the shared fallback with Unsloth's marker-aware HTTP-retry prep.
    ``force_download`` re-fetches a newer blob over a cached one (Unsloth's model-update path).

    ``local_files_only`` resolves from the cache and never from the network, raising
    huggingface_hub's ``LocalEntryNotFoundError`` on a miss. It deliberately BYPASSES the shared
    fallback rather than forwarding the kwarg: that ladder exists only to recover a wedged
    network transfer, so with no transfer permitted there is nothing to watch, and -- decisively
    -- ``start_watchdog``-style version skew means an older installed ``unsloth_zoo`` could drop
    an unrecognised kwarg on the floor. A dropped ``local_files_only`` DOWNLOADS, which is the one
    outcome this parameter exists to prevent, so it must not depend on the installed zoo.

    ``reuse_other_cache_root`` (opt-in) resolves a file cached ONLY under huggingface_hub's
    import-time root through that root. Unsloth's cache folder is a setting, so after it changes every
    cached asset is invisible to a call pinned to the new root: GBs re-download, and a gated base with
    no valid token 401s even though the bytes are there and the preflight (which checks both roots)
    already cleared it. Routed THROUGH the other root rather than returned raw, so the ref still
    resolves and a republished file is picked up; the blob is reused, and offline/401
    hf_hub_download keeps the failed HEAD and serves the cached pointer. Off for
    ``force_download``, whose point is to re-fetch."""
    if cache_dir is None:
        from utils.hf_cache_settings import get_hf_cache_paths
        cache_dir = str(get_hf_cache_paths().hub_cache)
    if reuse_other_cache_root and not force_download and cache_dir is not None:
        try:
            from huggingface_hub import try_to_load_from_cache

            # Only a str is a cached path; a miss is None and a known-absent file is a sentinel.
            here = try_to_load_from_cache(
                repo_id, filename, repo_type = repo_type, revision = revision, cache_dir = cache_dir
            )
            if not isinstance(here, str):
                elsewhere = try_to_load_from_cache(
                    repo_id, filename, repo_type = repo_type, revision = revision, cache_dir = None
                )
                if isinstance(elsewhere, str) and Path(elsewhere).is_file():
                    cache_dir = None
        except Exception:  # noqa: BLE001 — a cache we cannot read just keeps the live root
            pass
    if local_files_only:
        # Straight to huggingface_hub, after the root switch above (which is pure cache lookups and
        # is exactly what lets an offline caller reach a file left under the import-time root).
        # Cancellation is still honoured either side, as the fallback path does it. ``force_download``
        # is not forwarded: there is nothing to re-fetch offline, and huggingface_hub rejects the pair.
        from huggingface_hub import hf_hub_download

        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Cancelled")
        path = hf_hub_download(
            repo_id = repo_id,
            filename = filename,
            token = token,
            repo_type = repo_type,
            revision = revision,
            cache_dir = cache_dir,
            local_files_only = True,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Cancelled")
        return path
    # Omit rather than forward None: an older unsloth_zoo hands `interval` straight to Event.wait(),
    # where None blocks forever and a hung Xet download never falls back. Omitting also lets the
    # shared layer pick its per-transport defaults.
    optional: dict[str, Any] = {}
    if stall_timeout is not None:
        optional["stall_timeout"] = stall_timeout
    if interval is not None:
        optional["interval"] = interval
    return _shared_hf_hub_download_with_xet_fallback(
        repo_id,
        filename,
        token,
        cancel_event = cancel_event,
        repo_type = repo_type,
        revision = revision,
        **optional,
        grace_period = grace_period,
        on_status = on_status,
        force_download = force_download,
        cache_dir = cache_dir,
        prepare_for_http_fn = partial(_studio_prepare_for_http, cache_dir = cache_dir),
    )


def snapshot_download_with_xet_fallback(repo_id: str, **kwargs: Any) -> str:
    """Whole-repo download via the shared fallback with Unsloth's marker-aware HTTP-retry prep."""
    if kwargs.get("cache_dir") is None:
        from utils.hf_cache_settings import get_hf_cache_paths
        kwargs["cache_dir"] = str(get_hf_cache_paths().hub_cache)
    kwargs.setdefault(
        "prepare_for_http_fn",
        partial(_studio_prepare_for_http, cache_dir = kwargs["cache_dir"]),
    )
    return _shared_snapshot_download_with_xet_fallback(repo_id, **kwargs)
