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
_load_lock = threading.Lock()


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

            _prev_gpu_init = _os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT")
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

    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - an older/absent unsloth_zoo must degrade, not crash
        first_error = exc

    previous = _os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT")
    _os.environ["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        import logging as _logging
        _logging.getLogger(__name__).debug(
            "%s unavailable (%s; with GPU init disabled: %s)", module_name, first_error, exc
        )
        return None
    finally:
        if previous is None:
            _os.environ.pop("UNSLOTH_ZOO_DISABLE_GPU_INIT", None)
        else:
            _os.environ["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = previous


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
    "start_watchdog": _degraded_start_watchdog,
    "get_hf_download_state": _degraded_get_hf_download_state,
}

# Annotation-only declarations for the three names above: they bind NO value, so lookup still misses
# and PEP 562 ``__getattr__`` resolves them lazily -- but ruff/pyflakes see them as defined, so listing
# them in ``__all__`` does not trip F822 (while F822 still catches a real typo elsewhere in the list).
DownloadStallError: type
start_watchdog: Any
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
    return _shared_hf_hub_download_with_xet_fallback(
        repo_id,
        filename,
        token,
        cancel_event = cancel_event,
        repo_type = repo_type,
        revision = revision,
        stall_timeout = stall_timeout,
        interval = interval,
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
