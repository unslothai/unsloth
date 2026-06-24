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
    # The shared helper lives in a newer unsloth_zoo; fail with an actionable
    # message instead of a bare ModuleNotFoundError at Studio startup.
    if exc.name == "unsloth_zoo.hf_xet_fallback":
        raise RuntimeError(
            "Unsloth Studio requires an unsloth_zoo that provides "
            "unsloth_zoo.hf_xet_fallback. Upgrade unsloth_zoo alongside unsloth "
            "(pip install -U unsloth_zoo)."
        ) from exc
    raise

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
    (vs the generic delete-incompletes default in unsloth_zoo)."""
    from hub.utils.download_registry import prepare_cache_for_transport
    prepare_cache_for_transport(repo_type, repo_id, "http")


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
