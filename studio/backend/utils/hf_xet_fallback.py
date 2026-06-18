# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Xet-primary HF downloads with an automatic HTTP fallback on a no-progress stall.

Xet (``hf_xet``) is the fast default but can hang with no progress and no
exception, and a blocked native thread cannot be killed. Keep Xet primary; fall
back to plain HTTP only when the parent observes a stall. ``HF_HUB_DISABLE_XET``
is read at import time, so the fallback runs in a fresh ``spawn`` child (not a
thread) that sets the env before importing ``huggingface_hub``. Cached files
short-circuit with no child; deterministic errors (401/403/404/disk-full) and
cancellation propagate without a fallback. Mirrors the safetensors inference
recovery in core/inference/{orchestrator,worker}.py.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
from typing import Any, Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

_CTX = mp.get_context("spawn")

# Defaults match the existing inference watchdog and hub shutdown deadline.
DEFAULT_HEARTBEAT_INTERVAL = 30.0
DEFAULT_STALL_TIMEOUT = 180.0
DEFAULT_GRACE_PERIOD = 10.0
_POLL_INTERVAL = 0.5


class DownloadStallError(RuntimeError):
    """Raised when no download progress is observed for too long.

    Canonical home; orchestrator.py re-imports it so all paths share one type.
    """


def child_should_disable_xet(config: dict) -> bool:
    """Single source of truth for the per-worker Xet env flip."""
    return bool(config.get("disable_xet"))


def get_hf_download_state(
    repo_ids: Optional[list[str]] = None, *, repo_type: str = "model"
) -> Optional[tuple[int, bool]]:
    """Return ``(total_on_disk_bytes, has_incomplete)`` for the active HF cache.

    Sparse-aware (st_blocks based) so a sparse Xet/``hf_transfer`` ``.incomplete``
    is not mistaken for full-size progress. ``None`` means the state could not be
    measured, so callers skip stall logic for that tick.
    """
    try:
        from hub.utils.hf_cache_state import (
            blob_bytes_present,
            has_active_incomplete_blobs,
            hf_cache_root,
            iter_active_repo_cache_dirs,
        )

        if hf_cache_root() is None:
            return (0, False)

        total = 0
        has_incomplete = False
        for repo_id in repo_ids or []:
            # Skip local paths: HF IDs never start with / . ~ or contain "\".
            if not repo_id or repo_id.startswith(("/", ".", "~")) or "\\" in repo_id:
                continue
            for entry in iter_active_repo_cache_dirs(repo_type, repo_id):
                blobs_dir = entry / "blobs"
                if not blobs_dir.is_dir():
                    continue
                for blob in blobs_dir.iterdir():
                    try:
                        if blob.is_file():
                            total += blob_bytes_present(blob)
                    except OSError:
                        pass
            if has_active_incomplete_blobs(repo_type, repo_id):
                has_incomplete = True
        return (total, has_incomplete)
    except Exception as e:
        logger.debug("Failed to determine HF download state: %s", e)
        return None


def start_watchdog(
    *,
    repo_ids: list[str],
    on_stall: Callable[[str], None],
    repo_type: str = "model",
    interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    xet_disabled: bool = False,
    on_heartbeat: Optional[Callable[[str], None]] = None,
) -> threading.Event:
    """Start a daemon thread that fires ``on_stall(message)`` exactly once iff a
    ``*.incomplete`` is present AND the on-disk size is unchanged for
    *stall_timeout* seconds. The timer resets while no ``*.incomplete`` exists, so
    post-download init is never misread as a stall. Returns a stop event the
    caller sets when the download phase ends.
    """
    stop = threading.Event()
    transport = "https" if xet_disabled else "xet"
    fired = False

    def _beat() -> None:
        nonlocal fired
        state = get_hf_download_state(repo_ids, repo_type = repo_type)
        last_size = state[0] if state is not None else 0
        last_change = time.monotonic()

        while not stop.wait(interval):
            state = get_hf_download_state(repo_ids, repo_type = repo_type)
            now = time.monotonic()

            if state is None:
                if on_heartbeat is not None:
                    on_heartbeat(f"Downloading ({transport} transport)...")
                continue

            current_size, has_incomplete = state
            if current_size != last_size:
                last_size = current_size
                last_change = now

            # Reset unless .incomplete confirms an active download, so model init
            # and lock waits are not counted as a stall.
            if not has_incomplete:
                last_change = now
            elif now - last_change >= stall_timeout:
                if not fired:
                    fired = True
                    on_stall(
                        f"Download appears stalled ({transport} transport) "
                        f"-- no progress for {int(now - last_change)}s"
                    )
                return

            if on_heartbeat is not None:
                on_heartbeat(f"Downloading ({transport} transport)...")

    threading.Thread(target = _beat, daemon = True, name = "hf-xet-watchdog").start()
    return stop


def _download_child_entry(
    *,
    repo_id: str,
    filename: str,
    token: Optional[str],
    repo_type: str,
    disable_xet: bool,
    result_queue: Any,
) -> None:
    """Spawn-child entrypoint: download one file and report the result.

    Top-level and picklable. Sets the Xet env BEFORE importing huggingface_hub,
    forms its own process group so the parent can kill the whole transfer, and
    never logs the token or signed URLs.
    """
    # Die with Studio on Linux (this mp child gets no parent-set preexec_fn).
    try:
        from utils.process_lifetime import bind_current_process_to_parent_lifetime
        bind_current_process_to_parent_lifetime()
    except Exception:
        pass

    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass

    if disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        # Keep the HTTP writer sequential and resumable (hf_transfer leaves sparse
        # partials a sequential resume cannot safely continue).
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    # Test-only fault injection (never set in production): stall the Xet attempt
    # so the watchdog + HTTP fallback can be exercised against a real repo.
    if not disable_xet and os.environ.get("UNSLOTH_HF_XET_FORCE_STALL") == "1":
        import time as _t
        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            blobs = os.path.join(HF_HUB_CACHE, "models--" + repo_id.replace("/", "--"), "blobs")
            os.makedirs(blobs, exist_ok = True)
            with open(os.path.join(blobs, "xet-force-stall.incomplete"), "wb") as fh:
                fh.write(b"\0" * 4096)
        except OSError:
            pass
        while True:
            _t.sleep(3600)

    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id = repo_id,
            filename = filename,
            repo_type = repo_type,
            token = token,
        )
        result_queue.put({"ok": True, "path": path})
    except BaseException as e:  # noqa: BLE001 - report every failure to the parent
        error = f"{type(e).__name__}: {e}"
        try:
            from hub.utils.download_registry import scrub_secrets
            error = scrub_secrets(error, hf_token = token)
        except Exception:
            pass
        result_queue.put({"ok": False, "error": error})


def _terminate_process_group(proc: "mp.process.BaseProcess", grace_period: float) -> None:
    """Kill *proc* and its whole process group (Xet may spawn helper procs).

    The child calls ``os.setsid()`` so its pgid equals its pid; signal via
    ``os.killpg(pid, ...)`` -- NOT ``getpgid``, which before the child becomes a
    group leader resolves to OUR group. SIGTERM, then SIGKILL after *grace_period*.
    """
    pid = proc.pid

    def _signal_group(sig: int) -> None:
        if pid is not None and hasattr(os, "killpg"):
            try:
                os.killpg(pid, sig)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        # Windows or pre-setsid: best effort on the single process.
        try:
            proc.terminate() if sig != getattr(signal, "SIGKILL", -9) else proc.kill()
        except Exception:
            pass

    _signal_group(getattr(signal, "SIGTERM", signal.SIGINT))
    proc.join(timeout = grace_period)
    if proc.is_alive():
        _signal_group(getattr(signal, "SIGKILL", signal.SIGTERM))
        proc.join(timeout = 5.0)


def _run_download_attempt(
    repo_id: str,
    filename: str,
    token: Optional[str],
    *,
    repo_type: str,
    disable_xet: bool,
    cancel_event: Optional[threading.Event],
    stall_timeout: float,
    interval: float,
    grace_period: float,
    on_status: Optional[Callable[[str], None]],
) -> tuple[str, Optional[str]]:
    """Run one download in a spawn child supervised by the no-progress watchdog.

    Returns ``("ok", path)``, ``("stall", None)``, ``("cancelled", None)``, or
    ``("error", message)``. This is the seam tests monkeypatch to avoid spawning.
    """
    result_queue: Any = _CTX.Queue()
    proc = _CTX.Process(
        target = _download_child_entry,
        kwargs = dict(
            repo_id = repo_id,
            filename = filename,
            token = token,
            repo_type = repo_type,
            disable_xet = disable_xet,
            result_queue = result_queue,
        ),
        daemon = True,
    )
    proc.start()
    from utils.process_lifetime import adopt_pid

    adopt_pid(proc.pid)  # bind to parent lifetime (Windows job / sweep)

    stalled = threading.Event()
    stop_watchdog = start_watchdog(
        repo_ids = [repo_id],
        on_stall = lambda msg: stalled.set(),
        repo_type = repo_type,
        interval = interval,
        stall_timeout = stall_timeout,
        xet_disabled = disable_xet,
        on_heartbeat = on_status,
    )

    result: Optional[dict] = None
    try:
        while proc.is_alive():
            if cancel_event is not None and cancel_event.is_set():
                _terminate_process_group(proc, grace_period)
                return ("cancelled", None)
            if stalled.is_set():
                _terminate_process_group(proc, grace_period)
                return ("stall", None)
            try:
                result = result_queue.get(timeout = _POLL_INTERVAL)
                break
            except queue.Empty:
                continue
        else:
            # Process exited; drain any result it enqueued.
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                result = None
    finally:
        stop_watchdog.set()
        proc.join(timeout = grace_period)

    if result is None:
        return (
            "error",
            f"download process for '{repo_id}/{filename}' exited "
            f"(code={proc.exitcode}) without a result",
        )
    if result.get("ok"):
        return ("ok", result["path"])
    return ("error", result.get("error") or "unknown download error")


def hf_hub_download_with_xet_fallback(
    repo_id: str,
    filename: str,
    token: Optional[str],
    *,
    cancel_event: Optional[threading.Event] = None,
    repo_type: str = "model",
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    interval: float = DEFAULT_HEARTBEAT_INTERVAL,
    grace_period: float = DEFAULT_GRACE_PERIOD,
    on_status: Optional[Callable[[str], None]] = None,
) -> str:
    """Download a single file with Xet primary and HTTP as a stall-only fallback.

    Returns the local cache path. Raises ``RuntimeError("Cancelled")`` if
    *cancel_event* is set, re-raises a deterministic child error unchanged (no
    fallback), and raises ``DownloadStallError`` only if BOTH transports stall.
    """
    # Finalized blob already cached: return it with no child and no network.
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(repo_id, filename, repo_type = repo_type)
        if isinstance(cached, str) and os.path.exists(cached):
            return cached
    except Exception as e:
        logger.debug("Cached probe failed for %s/%s: %s", repo_id, filename, e)

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Cancelled")

    disable_xet = False
    for attempt in range(2):
        if disable_xet:
            # Purge a non-HTTP partial before resuming over HTTP: an HTTP resume
            # over a sparse Xet/hf_transfer partial silently corrupts the blob.
            try:
                from hub.utils.download_registry import prepare_cache_for_transport
                prepare_cache_for_transport(repo_type, repo_id, "http")
            except Exception as e:
                logger.debug("prepare_cache_for_transport failed for %s: %s", repo_id, e)

        kind, payload = _run_download_attempt(
            repo_id,
            filename,
            token,
            repo_type = repo_type,
            disable_xet = disable_xet,
            cancel_event = cancel_event,
            stall_timeout = stall_timeout,
            interval = interval,
            grace_period = grace_period,
            on_status = on_status,
        )

        if kind == "ok":
            return payload  # type: ignore[return-value]
        if kind == "cancelled":
            raise RuntimeError("Cancelled")
        if kind == "error":
            # Deterministic failure: the other transport would fail identically.
            raise RuntimeError(payload)
        # kind == "stall"
        if attempt == 0 and not disable_xet:
            logger.warning(
                "Download stalled for '%s/%s' -- retrying with HF_HUB_DISABLE_XET=1",
                repo_id,
                filename,
            )
            if on_status is not None:
                on_status(f"{repo_id}/{filename}: Xet stalled, retrying over HTTP")
            disable_xet = True
            continue
        raise DownloadStallError(
            f"Download stalled for '{repo_id}/{filename}' even with "
            f"HF_HUB_DISABLE_XET=1 -- check your network connection"
        )

    # Unreachable: the loop either returns or raises on each attempt.
    raise DownloadStallError(f"Download failed for '{repo_id}/{filename}'")
