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
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

# Mirror unsloth_zoo.hf_xet_fallback; literals avoid importing zoo/transformers.
DEFAULT_GRACE_PERIOD = 10.0
DEFAULT_HEARTBEAT_INTERVAL = 30.0
# Xet gets 30s of no progress before HTTP retry; HTTP (last resort) keeps 180s.
DEFAULT_STALL_TIMEOUT = 30.0
DEFAULT_CONNECT_TIMEOUT = 90.0
DEFAULT_HTTP_STALL_TIMEOUT = 180.0
# Xet worker attempts per download; finished shards are skipped on retry.
DEFAULT_XET_ATTEMPTS = 2

_shared: Any = None
_shared_available: Optional[bool] = None
_shared_import_error: Optional[BaseException] = None
# One RLock for loading and every UNSLOTH_ZOO_DISABLE_GPU_INIT save/set/restore,
# so interleaved restores cannot leave it set; reentrant across spawns.
_load_lock = threading.RLock()


def _gpu_present() -> bool:
    """Torch only, with no unsloth_zoo import; MPS does not count, since zoo's full init rejects it."""
    try:
        import torch
    except Exception:  # noqa: BLE001 -- no torch at all: the light path is the right one
        return False
    for probe in (
        lambda: torch.cuda.is_available(),
        lambda: torch.xpu.is_available(),
    ):
        try:
            if probe():
                return True
        except Exception:  # noqa: BLE001 -- a missing backend is just "not this one"
            continue
    return False


def _gate_torch_stack(reason: str) -> None:
    """Let the torch warm finish ``import torch._dynamo`` before an ``unsloth_zoo`` import (never fatal)."""
    try:
        from utils.torch_warmup import gate_torch_stack_import
        gate_torch_stack_import(reason)
    except Exception:  # noqa: BLE001, S110 - the gate is a safety net, never a new failure
        pass


def _media_import_window(reason: str):
    try:
        from utils.torch_warmup import hold_media_import_window
    except Exception:  # noqa: BLE001
        return nullcontext()
    return hold_media_import_window(reason)


def _import_light(module_name: str) -> Any:
    """``unsloth_zoo.<name>`` from the init-free ``unsloth_zoo_downloads`` (unsloth-zoo#1752), else ``None``.

    An already-loaded ``unsloth_zoo.<name>`` wins, keeping its identity (and test doubles).
    """
    import sys as _sys

    module = _sys.modules.get(module_name)
    if module is not None and not getattr(
        getattr(module, "__spec__", None), "_initializing", False
    ):
        return module
    try:
        from unsloth_zoo_downloads import hf_xet_fallback, hf_xet_health, hf_xet_tuning
    except Exception:  # noqa: BLE001 - absent on older zoo releases: take the full-init path
        return None
    return {
        "unsloth_zoo.hf_xet_fallback": hf_xet_fallback,
        "unsloth_zoo.hf_xet_health": hf_xet_health,
        "unsloth_zoo.hf_xet_tuning": hf_xet_tuning,
    }.get(module_name)


def _load_shared() -> bool:
    """Import ``unsloth_zoo.hf_xet_fallback`` on demand; return True if available. Deferred so
    importing this module at worker startup does not pull transformers in before the sidecar is
    activated. Degrades (returns False) rather than crashing when unsloth_zoo is unavailable."""
    global _shared, _shared_available, _shared_import_error
    if _shared_available is not None:
        return _shared_available
    light = _import_light("unsloth_zoo.hf_xet_fallback")
    if light is not None:
        with _load_lock:
            if _shared_available is None:
                _shared, _shared_available, _shared_import_error = light, True, None
            return _shared_available
    # Outside _load_lock: a thread waiting on the warm must not hold up the env-var bookkeeping.
    _gate_torch_stack("unsloth_zoo.hf_xet_fallback import")
    with _media_import_window("unsloth_zoo.hf_xet_fallback import"):
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

                # ...but NOT on a CUDA/XPU host (see _gpu_present). That flag makes unsloth_zoo take its MLX/CPU path,
                # injecting triton and bitsandbytes STUBS into sys.modules for the process. On a working GPU box those stubs
                # raise from the first CUDA-only kernel, turning a healthy GPU into 500s.
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
                _gpu_init_override_depth += _ours
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


# Memoize failure too, so an old zoo does not reopen the GPU-init env window every call.
_UNTRIED = object()
_optional_modules: "dict[str, Any]" = {}


def _reset_optional_module_cache() -> None:
    """Forget memoised optional-module results (tests that install or remove a zoo module)."""
    with _load_lock:
        _optional_modules.clear()


def _load_optional(module_name: str) -> Any:
    """Newer-zoo only; the GPU-init retry matters since zoo raises NotImplementedError on CPU-only hosts."""
    import importlib
    import os as _os

    cached = _optional_modules.get(module_name, _UNTRIED)
    if cached is not _UNTRIED:
        return cached

    light = _import_light(module_name)
    if light is not None:
        _optional_modules[module_name] = light
        return light

    _gate_torch_stack(f"{module_name} import")
    with _media_import_window(f"{module_name} import"):
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
            # Same rule as _load_shared: the retry's triton/bitsandbytes stubs stay in sys.modules for good.
            if _gpu_present():
                import sys as _sys

                # A zoo __init__ failing late leaves the submodules it already ran (hf_xet_tuning) in sys.modules.
                module = _sys.modules.get(module_name)
                # Still executing in another thread: never memoise a half-built module.
                if getattr(getattr(module, "__spec__", None), "_initializing", False):
                    module = None
                if module is None:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "%s unavailable (%s); not retrying under UNSLOTH_ZOO_DISABLE_GPU_INIT because this host has an "
                        "accelerator and that path would stub out triton/bitsandbytes for the whole process.",
                        module_name,
                        first_error,
                    )
                _optional_modules[module_name] = module
                return module
            global _gpu_init_override_depth
            previous = _os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT")
            ours = previous != "1"
            # Claim BEFORE the write, release AFTER the restore.
            _gpu_init_override_depth += ours
            try:
                _os.environ["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
                try:
                    module = importlib.import_module(module_name)
                except Exception as exc:  # noqa: BLE001
                    import logging as _logging
                    _logging.getLogger(__name__).debug(
                        "%s unavailable (%s; with GPU init disabled: %s)",
                        module_name,
                        first_error,
                        exc,
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


@dataclass(frozen = True)
class _EnvXetHealth:
    use_xet: bool
    reason: str
    source: str = "forced"

    def __bool__(self) -> bool:
        return self.use_xet


def _env_xet_health() -> Any:
    """The env checks atop ``unsloth_zoo.hf_xet_health.xet_health``, for when that module cannot load: without
    them Auto picks Xet and the worker gets ``HF_HUB_DISABLE_XET=0`` over the operator's ``1``."""

    def _on(name: str) -> bool:
        return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")

    if _on("UNSLOTH_DISABLE_XET") or _on("UNSLOTH_STABLE_DOWNLOADS") or _on("HF_HUB_DISABLE_XET"):
        return _EnvXetHealth(False, "Xet disabled by environment")
    if _on("UNSLOTH_FORCE_XET"):
        return _EnvXetHealth(True, "Xet forced by environment")
    return None


def _xet_health_from(module: Any, **kwargs: Any) -> Any:
    if module is None:
        return _env_xet_health()
    try:
        return module.xet_health(**kwargs)
    except Exception as exc:  # noqa: BLE001
        import logging as _logging
        _logging.getLogger(__name__).debug("xet_health failed: %s", exc)
        return None


def cached_xet_health(**kwargs: Any) -> Any:
    """Peeks at Zoo's Xet verdict without loading it or taking _load_lock, which a slow import holds."""
    module = _optional_modules.get("unsloth_zoo.hf_xet_health", _UNTRIED)
    return None if module is _UNTRIED else _xet_health_from(module, **kwargs)


def xet_health(**kwargs: Any) -> Any:
    """Loads Zoo's Xet verdict to decide a download; None means no opinion and keeps Xet."""
    module = _load_optional("unsloth_zoo.hf_xet_health")
    return _xet_health_from(module, **kwargs)


def xet_health_is_forced(health: Any) -> bool:
    """True only for an env override stamped source=forced (e.g. UNSLOTH_FORCE_XET), so the free-RAM gate stands down; unreadable answers False."""
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
    """Zoo sizes HF_XET_* in env in place from total RAM, so clamp_to_available_ram must follow; returns what it wrote, or None when the zoo has no opinion."""
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


# Share of available RAM a download may use for buffers.
_AVAILABLE_RAM_SHARE = 4
# Integer arithmetic converges in 1-2 passes; the bound only guards a future non-monotonic zoo.
_CLAMP_MAX_PASSES = 3
_BUFFER_LIMIT_KEY = "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"


def _as_int(value: str) -> "Optional[int]":
    """``value`` as a plain int, or None for the unit-suffixed ones ("60s") that never scale."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


# Reservations cover the gap before a child allocates, counting only unmaterialized bytes.
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
    """Never os.kill(pid, 0): on Windows it terminates the process; uses process_lifetime's probe."""
    try:
        from utils.process_lifetime import _pid_alive as _platform_pid_alive
        return bool(_platform_pid_alive(pid))
    except Exception:  # noqa: BLE001 - fall through to the POSIX probe below
        pass
    if os.name == "nt":
        # Assume alive: holding a reservation too long beats killing a download.
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
    """Physical RAM the pid already holds via psutil rss; 0 on failure, which reserves the whole promise."""
    try:
        import psutil  # noqa: PLC0415 - optional, and only on the ledger path
        return max(0, int(psutil.Process(pid).memory_info().rss))
    except Exception:  # noqa: BLE001 - an unreadable worker is not evidence it allocated nothing
        return 0


def _live_reserved_locked() -> int:
    """Promised bytes not yet resident; capped at the promise so resident RAM is not charged twice."""
    now = time.monotonic()
    total = 0
    for token, entry in list(_budget_reservations.items()):
        nbytes, pid, stamp = entry
        if pid is None:
            if now - stamp > _UNBOUND_RESERVATION_TTL:
                _budget_reservations.pop(token, None)
            else:
                total += nbytes
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
    """Shrinks zoo-written HF_XET_* budgets to free RAM; user-set keys survive (HF_XET_HIGH_PERFORMANCE drops the caps); bind_worker_budget ties reservation to pid."""
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
        # Reading the ledger and reserving against it is ONE decision.
        with _budget_lock:
            unclaimed = max(0, available - _live_reserved_locked())
            budget = max(floor, unclaimed // _AVAILABLE_RAM_SHARE)
            if limit <= budget:
                # Still reserved: four unclamped workers would otherwise promise four full budgets.
                _reserve_worker_budget(limit)
                return sized

            # Re-ask the zoo with a synthetic RAM so all derived limits scale together.
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

            # Reduce-only: never raise a value; derived numbers are monotonic in RAM.
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
    """Free RAM and Xet's floor; free RAM is None when unmeasurable, unlike the zoo's total-RAM check."""
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
    """Tests the Xet RAM floor of the zoo against free RAM minus unresident promises; None keeps Xet. Capabilities probe and resolve_auto_use_xet must agree."""
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
    """Per-worker Xet flip; must not import unsloth_zoo or transformers, so the worker decides early."""
    return bool(config.get("disable_xet"))


def is_data_phase_stall(message: str) -> bool:
    """Whether the watchdog fired after bytes flowed; a pre-first-byte trip is not a data-phase stall."""
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
    # Keep heartbeats so the orchestrator's inactivity deadline is not tripped.
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


# Resolved lazily via PEP 562 so only these names trigger the heavy load; DownloadStallError identity matters for except clauses.
_DEGRADED_ATTRS = {
    "DownloadStallError": _DegradedDownloadStallError,
    "get_hf_download_state": _degraded_get_hf_download_state,
}


# Nonzero while a loader sets UNSLOTH_ZOO_DISABLE_GPU_INIT; utf8_child_env strips it from children, since zoo injects triton and bitsandbytes stubs when set.
_gpu_init_override_depth = 0


def gpu_init_override_active() -> bool:
    """Is a loader currently holding UNSLOTH_ZOO_DISABLE_GPU_INIT set for its own import?"""
    return _gpu_init_override_depth > 0


def env_override_barrier() -> Any:
    """Hold across a spawn so no loader is mid-override, since children inherit the live os.environ."""
    return _load_lock


def _supported_kwargs(fn: Any, kwargs: "dict[str, Any]") -> "dict[str, Any]":
    """Filters kwargs to fn's signature; uninspectable callables get them all unchanged."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def start_watchdog(**kwargs: Any) -> Any:
    """Drops kwargs the zoo lacks: passing one raises TypeError into the except clause, so the watchdog never starts and stalled Xet is never retried over HTTP."""
    impl = _shared.start_watchdog if _load_shared() else _degraded_start_watchdog
    return impl(**_supported_kwargs(impl, kwargs))


# Annotation-only so ruff sees these in __all__ while PEP 562 still resolves them lazily.
DownloadStallError: type
get_hf_download_state: Any


def __getattr__(name: str) -> Any:
    if name in _DEGRADED_ATTRS:
        if _load_shared():
            return getattr(_shared, name)
        return _DEGRADED_ATTRS[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Seam the public wrappers call and tests monkeypatch.
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
    gguf_header_delta: bool = False,
) -> str:
    """local_files_only skips the fallback (an old zoo could drop it into a download); reuse_other_cache_root avoids re-downloads and 401s; gguf_header_delta opt-in."""
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
        except Exception:  # noqa: BLE001 - a cache we cannot read just keeps the live root
            pass
    if local_files_only:
        # Straight to huggingface_hub after the root switch so an offline caller reaches the import-time root; force_download is not forwarded, hub rejects the pair.
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
    if gguf_header_delta and str(filename).lower().endswith(".gguf"):
        # A rebuilt file has the Hub's sha256, so it already is the newer blob a forced fetch wants.
        try:
            from hub.utils.gguf_header_delta import prepare_media_gguf
            if prepare_media_gguf(
                repo_id,
                filename,
                token,
                repo_type = repo_type,
                revision = revision,
                cache_dir = cache_dir,
                cancel_event = cancel_event,
            ).placed:
                force_download = False
        except Exception:  # noqa: BLE001 - an optimisation only: the normal download follows
            pass
    # Omit rather than forward None: an older unsloth_zoo hands `interval` straight to Event.wait()
    optional: dict[str, Any] = {}
    if stall_timeout is not None:
        optional["stall_timeout"] = stall_timeout
    if interval is not None:
        optional["interval"] = interval
    from hub.utils.hf_tokens import call_with_anonymous_retry

    # The 401 comes on the metadata HEAD, before any byte is written.
    return call_with_anonymous_retry(
        lambda token: _shared_hf_hub_download_with_xet_fallback(
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
        ),
        token,
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
