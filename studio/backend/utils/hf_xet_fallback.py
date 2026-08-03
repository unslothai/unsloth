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

import threading
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

# Defaults mirror unsloth_zoo.hf_xet_fallback; plain literals so they resolve (including as
# default args below) without importing unsloth_zoo/transformers.
DEFAULT_GRACE_PERIOD = 10.0
DEFAULT_HEARTBEAT_INTERVAL = 30.0
# Xet gets 30s of zero progress before the HTTP retry; HTTP, as the last resort, keeps 180s. The
# wrappers below pass None so the shared layer picks per transport -- these literals exist for
# callers that want an explicit value without triggering the heavy import.
DEFAULT_STALL_TIMEOUT = 30.0
DEFAULT_CONNECT_TIMEOUT = 90.0
DEFAULT_HTTP_STALL_TIMEOUT = 180.0

# --- lazy shared-backend loader ----------------------------------------------------------------
_shared: Any = None
_shared_available: Optional[bool] = None  # None = not yet attempted
_shared_import_error: Optional[BaseException] = None
# Guards the memoized _shared_available AND every UNSLOTH_ZOO_DISABLE_GPU_INIT save/set/restore in
# this module. Both loaders below mutate that one process-wide variable, so they must serialize
# against each other, not merely against themselves: two locks would still allow A-saves-unset /
# B-saves-"1" / A-restores-unset / B-restores-"1", leaving it set for the life of the process.
# Reentrant on purpose. It still excludes OTHER threads, which is all the env save/set/restore
# sequence needs, but child_environment_for_spawn now holds it across a spawn and that context
# manager legitimately nests -- its own _spawn_env_lock is an RLock for exactly the same reason.
_load_lock = threading.RLock()


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


# Result cache for _load_optional, keyed by module name. Memoising the FAILURE is the point: on a
# zoo that predates these modules the import can never start succeeding, and without this every
# xet_health / record_xet_outcome / xet_env_overrides call re-ran the whole GPU-init retry --
# re-opening the process-wide env window on every single download.
_UNTRIED = object()
_optional_modules: "dict[str, Any]" = {}


def _reset_optional_module_cache() -> None:
    """Forget memoised optional-module results (tests that install or remove a zoo module)."""
    with _load_lock:
        _optional_modules.clear()


def _load_optional(module_name: str) -> Any:
    """Import an optional shared Xet helper module, or return ``None``.

    Separate from ``_load_shared`` because these modules (health / tuning) are pure stdlib and
    exist only in newer unsloth_zoo: a Studio pinned to an older zoo must keep downloading, just
    without the preflight verdict or the buffer caps.

    The GPU-init retry matters more here than anywhere else. ``unsloth_zoo.__init__`` runs torch
    accelerator detection and raises ``NotImplementedError`` on a CPU-only host -- and a CPU-only
    host is precisely the small machine whose RAM these caps exist to protect. Without the retry
    the caps would silently switch themselves off exactly where they are needed.
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

    # The retry mutates process-wide state, so it must not run concurrently with itself. Two
    # requests could otherwise interleave save/set/restore -- A saves unset and sets 1, B saves 1,
    # A restores unset, B restores 1 -- and leave UNSLOTH_ZOO_DISABLE_GPU_INIT set for the life of
    # the process, so every later worker inherits it and skips Zoo's GPU init. Deliberately the
    # SAME lock _load_shared holds while it runs its own copy of this sequence.
    with _load_lock:
        cached = _optional_modules.get(module_name, _UNTRIED)
        if cached is not _UNTRIED:
            return cached
        global _gpu_init_override_depth
        previous = _os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT")
        ours = previous != "1"
        # Claim ownership BEFORE the write and release it AFTER the restore, so the window in which
        # the variable is set sits strictly inside the window in which a spawning thread can see
        # that it is ours. The other order leaves a gap at each end where a child is handed a flag
        # nobody is claiming, and a child that inherits it never clears it.
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


def xet_health(**kwargs: Any) -> Any:
    """The machine's Xet verdict, or ``None`` when unsloth_zoo cannot answer.

    ``None`` means "no opinion": callers keep their existing default (Xet) rather than treating a
    missing health module as a reason to downgrade.
    """
    module = _load_optional("unsloth_zoo.hf_xet_health")
    if module is None:
        return None
    try:
        return module.xet_health(**kwargs)
    except Exception as exc:  # noqa: BLE001
        import logging as _logging
        _logging.getLogger(__name__).debug("xet_health failed: %s", exc)
        return None


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


def child_should_disable_xet(config: dict) -> bool:
    """Single source of truth for the per-worker Xet env flip (mirrors
    ``unsloth_zoo.hf_xet_fallback.child_should_disable_xet``). Deliberately lightweight: importing or
    calling it must NOT pull in unsloth_zoo/transformers, so the worker can decide before activating
    the transformers sidecar (see the module docstring)."""
    return bool(config.get("disable_xet"))


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


# Nonzero while a loader is inside its UNSLOTH_ZOO_DISABLE_GPU_INIT retry, during which that
# variable is set process-wide. Read by utf8_child_env so a child spawned in that window does not
# inherit it: unsloth_zoo injects triton and bitsandbytes STUBS when it is set, so a training child
# that inherited it would silently run against no-ops. Only counted when the loader actually
# introduced the value -- an operator who exported it themselves keeps it.
_gpu_init_override_depth = 0


def gpu_init_override_active() -> bool:
    """Is a loader currently holding UNSLOTH_ZOO_DISABLE_GPU_INIT set for its own import?"""
    return _gpu_init_override_depth > 0


def env_override_barrier() -> Any:
    """Context manager a caller holds across a spawn so no loader can be mid-override.

    ``multiprocessing`` spawn children inherit the parent's live ``os.environ`` -- there is no env
    dict to filter -- so the only way to keep UNSLOTH_ZOO_DISABLE_GPU_INIT out of a worker is to
    make sure no loader has it set at the moment the child is created. The loaders never spawn, so
    holding this alongside the spawn lock cannot deadlock. It is uncontended in practice because
    ``_load_optional`` memoises: the window opens at most once per module per process.
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

    This is a version-skew adapter, and it is load-bearing: the supported floor (2026.8.1) has no
    ``connect_timeout`` or ``heartbeat_interval`` and no ``**kwargs`` to absorb them, so passing one
    raises TypeError -- and every caller wraps this in ``except Exception``, so the watchdog would
    silently never start and a stalled Xet worker would never be killed or retried over HTTP. That
    is the feature being entirely off, not degraded. Filtering here keeps newer knobs live on a
    newer zoo without breaking the floor, and makes the NEXT new kwarg a no-op rather than a repeat
    of this bug.

    Dropping the pre-byte budget on 2026.8.1 costs less than it looks. That release does reset its
    timer whenever the child owns no ``.incomplete``, but huggingface_hub opens the partial BEFORE
    it calls ``xet_get`` (``file_download.py`` opens ``incomplete_path`` and invokes ``xet_get``
    inside that ``with``), and the floor counts a partial by presence rather than size -- so a
    hf_xet hang still sits behind an open zero-byte partial and still trips the floor's 180s data
    clock. Verified against the released wheel: wedged inside ``xet_get`` trips, wedged before the
    open does not. What stays uncovered there is the metadata phase, which is where
    ``snapshot_download`` calls ``repo_info`` with no timeout. That gap predates this shim and the
    connect clock closes it only once a zoo carrying it is released; passing the kwarg through
    early would not close it, it would disable the watchdog outright.
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
    "DownloadStallError",
    "child_should_disable_xet",
    "get_hf_download_state",
    "record_xet_outcome",
    "start_watchdog",
    "xet_env_overrides",
    "xet_health",
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
) -> str:
    """Single-file download via the shared fallback with Unsloth's marker-aware HTTP-retry prep.
    ``force_download`` re-fetches a newer blob over a cached one (Unsloth's model-update path)."""
    if cache_dir is None:
        from utils.hf_cache_settings import get_hf_cache_paths
        cache_dir = str(get_hf_cache_paths().hub_cache)
    # Omit rather than forward None. No production caller passes these, and an older unsloth_zoo
    # takes the value literally: its watchdog hands `interval` straight to Event.wait(), where None
    # blocks forever, so a hung Xet download would never fall back. Omitting them also lets the
    # shared layer pick its own per-transport defaults instead of freezing one here.
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
