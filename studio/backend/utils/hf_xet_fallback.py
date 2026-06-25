# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio shim over the shared Xet -> HTTP stall fallback.

The no-progress watchdog, the spawn-child download, and the single Xet -> HTTP
retry now live once in ``unsloth_zoo.hf_xet_fallback`` (so Unsloth main and Studio
share one implementation). This module re-exports that API and injects Studio's
marker-aware cache purge (``prepare_cache_for_transport``) so the hub download
manager keeps its ``.transport`` marker semantics on the HTTP retry. Call sites
(core/inference/llama_cpp.py, core/training/worker.py) and the orchestrator's
``DownloadStallError`` import are unchanged.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

try:
    from unsloth_zoo.hf_xet_fallback import (
        DEFAULT_GRACE_PERIOD,
        DEFAULT_HEARTBEAT_INTERVAL,
        DEFAULT_STALL_TIMEOUT,
        DownloadStallError,
        child_should_disable_xet,
        get_hf_download_state,
        start_watchdog,
    )
    from unsloth_zoo.hf_xet_fallback import (
        hf_hub_download_with_xet_fallback as _shared_hf_hub_download_with_xet_fallback,
        snapshot_download_with_xet_fallback as _shared_snapshot_download_with_xet_fallback,
    )
except ModuleNotFoundError as exc:
    if exc.name != "unsloth_zoo.hf_xet_fallback":
        raise

    # The shared helper lives in a newer unsloth_zoo. Rather than crash Studio at
    # startup on an older (but dependency-satisfying) unsloth_zoo, degrade
    # gracefully: plain HF downloads with the no-progress stall watchdog disabled
    # -- the same best-effort posture core Unsloth uses in from_pretrained. The
    # automatic Xet -> HTTP recovery returns as soon as unsloth_zoo is upgraded.
    # These are thin stubs, not a second copy of the orchestration.
    import logging as _logging

    _logging.getLogger(__name__).warning(
        "unsloth_zoo.hf_xet_fallback not found; the Xet stall watchdog is "
        "disabled. Upgrade unsloth_zoo to re-enable automatic Xet -> HTTP "
        "download recovery."
    )

    DEFAULT_HEARTBEAT_INTERVAL = 30.0
    DEFAULT_STALL_TIMEOUT = 180.0
    DEFAULT_GRACE_PERIOD = 10.0

    class DownloadStallError(RuntimeError):
        """Stub mirror of the shared type so callers and ``except`` clauses still
        resolve when the shared helper is unavailable (it is simply never raised
        in degraded mode, since there is no watchdog to detect a stall)."""

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
        # No stall detection without the shared helper, but keep emitting heartbeat
        # statuses so the orchestrator's inactivity deadline is not tripped during a
        # legitimately long load/download in this degraded mode.
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
            target = _beat, daemon = True, name = "hf-xet-degraded-heartbeat",
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
        # No subprocess to interrupt mid-call here, but keep the cancellation
        # contract: do not start, and do not return, a download once cancelled.
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
    """Make the partial safe for an HTTP resume using Studio's marker-aware purge,
    so the download manager's ``.transport`` marker accounting stays consistent
    (vs the generic delete-incompletes default in unsloth_zoo).

    The shared orchestrator already wraps this hook, but guard it here too so a
    purge failure (locked file, missing dir) is logged rather than aborting the
    HTTP retry that is the whole point of the fallback."""
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
) -> str:
    """Single-file download with the shared Xet -> HTTP stall fallback, using
    Studio's marker-aware cache prep on the HTTP retry."""
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
        prepare_for_http_fn = _studio_prepare_for_http,
    )


def snapshot_download_with_xet_fallback(repo_id: str, **kwargs: Any) -> str:
    """Whole-repo download with the shared Xet -> HTTP stall fallback, using Studio's
    marker-aware cache prep on the HTTP retry (same injection as the single-file path)."""
    kwargs.setdefault("prepare_for_http_fn", _studio_prepare_for_http)
    return _shared_snapshot_download_with_xet_fallback(repo_id, **kwargs)
