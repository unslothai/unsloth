# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio shim over the shared ``unsloth_zoo.hf_xet_fallback`` Xet -> HTTP stall fallback.

Re-exports the shared API and injects Studio's marker-aware cache purge
(``prepare_cache_for_transport``) so the download manager keeps its ``.transport``
marker semantics on the HTTP retry.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

_shared_import_error = None
try:
    import unsloth_zoo.hf_xet_fallback as _shared
    _shared_available = True
except Exception as _exc:  # noqa: BLE001 - any import failure must degrade, not crash
    # unsloth_zoo's __init__ runs torch/GPU detection, which raises on a torch-less/GPU-less Studio
    # host. The download helper needs none of it, so retry via the light UNSLOTH_ZOO_DISABLE_GPU_INIT
    # path before giving up.
    _shared_import_error = _exc
    import os as _os

    _prev_gpu_init = _os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT")
    _os.environ["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    try:
        import unsloth_zoo.hf_xet_fallback as _shared
        _shared_available = True
        _shared_import_error = None
    except Exception as _exc2:  # noqa: BLE001 - degrade so Studio still boots with plain HF downloads
        _shared_import_error = _exc2
        _shared_available = False
    finally:
        if _prev_gpu_init is None:
            _os.environ.pop("UNSLOTH_ZOO_DISABLE_GPU_INIT", None)
        else:
            _os.environ["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = _prev_gpu_init

if _shared_available:
    # Bind by assignment so each public name shares one module-level binding with the degraded branch.
    DEFAULT_GRACE_PERIOD = _shared.DEFAULT_GRACE_PERIOD
    DEFAULT_HEARTBEAT_INTERVAL = _shared.DEFAULT_HEARTBEAT_INTERVAL
    DEFAULT_STALL_TIMEOUT = _shared.DEFAULT_STALL_TIMEOUT
    DownloadStallError = _shared.DownloadStallError
    child_should_disable_xet = _shared.child_should_disable_xet
    get_hf_download_state = _shared.get_hf_download_state
    start_watchdog = _shared.start_watchdog
    _shared_hf_hub_download_with_xet_fallback = _shared.hf_hub_download_with_xet_fallback
    _shared_snapshot_download_with_xet_fallback = _shared.snapshot_download_with_xet_fallback
else:
    # Degrade instead of crashing Studio: plain HF downloads, stall watchdog disabled. Thin stubs,
    # not a second copy of the orchestration; recovery returns once unsloth_zoo is upgraded.
    import logging as _logging

    _logging.getLogger(__name__).warning(
        "unsloth_zoo.hf_xet_fallback unavailable (%s); the Xet stall watchdog is "
        "disabled. Install/upgrade unsloth_zoo (and its torch dependency) to "
        "re-enable automatic Xet -> HTTP download recovery.",
        _shared_import_error,
    )

    DEFAULT_HEARTBEAT_INTERVAL = 30.0
    DEFAULT_STALL_TIMEOUT = 180.0
    DEFAULT_GRACE_PERIOD = 10.0

    class DownloadStallError(RuntimeError):
        """Stub mirror so callers' ``except`` clauses resolve; never raised in degraded mode."""

    def child_should_disable_xet(config: dict) -> bool:
        return bool(config.get("disable_xet"))

    def get_hf_download_state(*args: Any, **kwargs: Any) -> None:
        return None  # unmeasurable -> the (absent) watchdog never fires

    def start_watchdog(
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

    def _shared_hf_hub_download_with_xet_fallback(
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

    def _shared_snapshot_download_with_xet_fallback(
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


__all__ = [
    "DEFAULT_GRACE_PERIOD",
    "DEFAULT_HEARTBEAT_INTERVAL",
    "DEFAULT_STALL_TIMEOUT",
    "DownloadStallError",
    "child_should_disable_xet",
    "get_hf_download_state",
    "start_watchdog",
    "hf_hub_download_with_xet_fallback",
    "snapshot_download_with_xet_fallback",
]


def _studio_prepare_for_http(repo_type: str, repo_id: str) -> None:
    """Studio's marker-aware purge before an HTTP resume, keeping the download manager's ``.transport``
    accounting consistent (vs unsloth_zoo's generic default). Guarded: a purge failure is logged,
    not fatal to the retry."""
    try:
        from hub.utils.download_registry import prepare_cache_for_transport
        prepare_cache_for_transport(repo_type, repo_id, "http")
    except Exception as exc:
        try:
            from loggers import get_logger
            get_logger(__name__).debug(
                "Studio prepare_cache_for_transport failed for %s: %s", repo_id, exc
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
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    grace_period: float = DEFAULT_GRACE_PERIOD,
    on_status: Optional[Callable[[str], None]] = None,
    force_download: bool = False,
) -> str:
    """Single-file download via the shared fallback with Studio's marker-aware HTTP-retry prep.
    ``force_download`` re-fetches a newer blob over a cached one (Studio's model-update path)."""
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
        prepare_for_http_fn = _studio_prepare_for_http,
    )


def snapshot_download_with_xet_fallback(repo_id: str, **kwargs: Any) -> str:
    """Whole-repo download via the shared fallback with Studio's marker-aware HTTP-retry prep."""
    kwargs.setdefault("prepare_for_http_fn", _studio_prepare_for_http)
    return _shared_snapshot_download_with_xet_fallback(repo_id, **kwargs)
