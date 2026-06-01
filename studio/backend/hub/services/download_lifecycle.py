# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

from fastapi import HTTPException

from hub.schemas.downloads import ActiveDownload, DownloadJobState
from hub.utils import download_manifest
from hub.utils import download_registry
from hub.utils import inventory_scan as hf_cache_scan
from hub.utils.hf_cache_state import EXIT_CANCELLED
from hub.utils.state_dir import RepoType


def backend_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_transport(use_xet: bool) -> str:
    transport = (
        download_registry.TRANSPORT_XET if use_xet else download_registry.TRANSPORT_HTTP
    )
    unavailable_reason = download_registry.download_transport_unavailable_reason(
        transport
    )
    if unavailable_reason is not None:
        raise HTTPException(status_code = 400, detail = unavailable_reason)
    return transport


def spawn_worker(
    args: list[str],
    hf_token: Optional[str],
    *,
    use_xet: bool,
    protected_blob_hashes: Optional[frozenset[str]] = None,
) -> subprocess.Popen:
    """Spawn the download worker.

    Both the XET client and the ``hf_transfer`` Rust extension write file
    chunks out of order, which makes their partial blobs unsafe to resume
    under a sequential writer (see the ``download_registry`` module docstring).
    The HTTP path stays on the built-in sequential downloader so SIGKILL →
    resume produces a byte-identical final file.

    ``protected_blob_hashes`` are blobs a concurrent same-repo peer is already
    writing; the worker excludes them from its cache-preparation purge so it
    never deletes an ``.incomplete`` shared with that peer (e.g. a bundled
    mmproj). Passed via env so it works identically on every platform.
    """
    cwd = backend_dir()
    mode = (
        download_registry.TRANSPORT_XET if use_xet else download_registry.TRANSPORT_HTTP
    )
    env = os.environ.copy()
    if protected_blob_hashes:
        env["UNSLOTH_PROTECTED_BLOB_HASHES"] = ",".join(sorted(protected_blob_hashes))
    else:
        env.pop("UNSLOTH_PROTECTED_BLOB_HASHES", None)
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_DISABLE_XET"] = "0" if use_xet else "1"
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "0" if hf_token else "1"
    # hf_transfer writes parallel HTTP Range chunks. Even within "http"
    # mode it can leave sparse partials. Disable unconditionally so that
    # the writer used by the worker is always single-stream sequential
    # when transport=http.
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    for token_key in (
        "HF_TOKEN",
        "HF_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
    ):
        env.pop(token_key, None)
    if hf_token:
        env["HF_TOKEN"] = hf_token
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{cwd}{os.pathsep}{existing_path}" if existing_path else str(cwd)
    )
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hub.workers.hf_download",
            *args,
            "--parent-pid",
            str(os.getpid()),
            "--transport",
            mode,
        ],
        env = env,
        cwd = str(cwd),
        stdout = subprocess.DEVNULL,
        stderr = subprocess.PIPE,
        start_new_session = sys.platform != "win32",
    )


def drain_stderr_excerpt(stream, edge_bytes: int = 500) -> bytes:
    """Drain a worker's stderr to EOF, retaining the first and last bytes.

    Reading incrementally keeps the pipe from filling while bounding memory:
    short messages are preserved whole; long messages keep context from both
    ends because worker stderr prefixes often identify the failing repo/phase."""
    if stream is None:
        return b""
    edge_bytes = max(1, edge_bytes)
    max_bytes = edge_bytes * 2
    full = bytearray()
    head = bytearray()
    tail = bytearray()
    truncated = False
    for chunk in iter(lambda: stream.read(4096), b""):
        if not truncated:
            full.extend(chunk)
            if len(full) <= max_bytes:
                continue
            truncated = True
            head.extend(full[:edge_bytes])
            tail.extend(full[-edge_bytes:])
            full.clear()
            continue
        tail.extend(chunk)
        if len(tail) > edge_bytes:
            del tail[:-edge_bytes]
    if not truncated:
        return bytes(full)
    return bytes(head + b"\n...[stderr truncated]...\n" + tail)


def _cancellation_return_codes() -> frozenset[int]:
    """Negative ``Popen.returncode`` values that mean intentional cancellation:
    the signal we send to stop a worker (SIGKILL) plus the ones it traps
    (SIGTERM/SIGINT). Crash signals (SIGSEGV, SIGABRT, ...) are deliberately
    excluded so they surface as errors. ``getattr`` keeps this valid on
    Windows, where these POSIX signals may be absent."""
    codes: set[int] = set()
    for name in ("SIGKILL", "SIGTERM", "SIGINT"):
        sig = getattr(signal, name, None)
        if sig is not None:
            codes.add(-int(sig))
    return frozenset(codes)


_CANCELLATION_RETURN_CODES = _cancellation_return_codes()


def _sigpipe_return_codes() -> frozenset[int]:
    sig = getattr(signal, "SIGPIPE", None)
    if sig is None:
        return frozenset()
    value = int(sig)
    return frozenset({-value, 128 + value})


_SIGPIPE_RETURN_CODES = _sigpipe_return_codes()


def classify_exit(rc: int, *, cancel_requested: bool = False) -> str:
    """Map a worker process exit code to a job state.

    - rc == 0: clean completion.
    - rc == EXIT_CANCELLED (130): the worker trapped a stop signal and exited
      its own cancellation path, so this is unambiguously a cancel.
    - rc killed by a cancellation signal (SIGKILL/SIGTERM/SIGINT): a cancel
      only when *we* asked for it. The OOM killer also sends SIGKILL, so an
      unrequested signal kill is surfaced as an error rather than masquerading
      as a user cancel.
    - rc killed by SIGPIPE or shell-style 128+SIGPIPE: parent pipe is gone, so
      the worker can no longer report progress and is treated as cancelled.
    - any other non-zero rc, including crash signals (SIGSEGV, SIGABRT):
      worker errored out.

    Windows has no POSIX signal exit encoding: ``Popen.kill`` calls
    ``TerminateProcess`` and the child exits with a positive code (typically
    1), so a user cancel cannot be told apart from a genuine error by the code
    alone. There ``cancel_requested`` is the only available signal, so any
    non-clean exit after we asked to stop is treated as a cancel.
    """
    if rc == 0:
        return "complete"
    if rc == EXIT_CANCELLED:
        return "cancelled"
    if rc in _SIGPIPE_RETURN_CODES:
        return "cancelled"
    if rc in _CANCELLATION_RETURN_CODES:
        return "cancelled" if cancel_requested else "error"
    if cancel_requested and sys.platform == "win32":
        return "cancelled"
    return "error"


def finalize_worker_exit(
    registry: download_registry.DownloadRegistry,
    key: str,
    proc: subprocess.Popen,
    *,
    hf_token: Optional[str],
    label: str,
    log_prefix: str,
    logger,
    repo_type: Optional[RepoType] = None,
    repo_id: Optional[str] = None,
    transport: Optional[str] = None,
) -> None:
    """Block until *proc* exits, then record the job's terminal state in
    *registry*. Drains stderr first so the pipe never fills, scrubs secrets
    from it, and classifies the exit code. A no-op when the process was
    already dropped (e.g. superseded by a newer job for the same key).

    There is no stall/liveness watchdog here: huggingface_hub already bounds
    every chunk read with a socket timeout, retries transient errors a few
    times (resetting the budget on any byte received), and raises a clean,
    resumable error on a genuinely dead connection. A custom watchdog could
    only ever kill a download hf_hub would itself have kept alive. We let the
    worker's own exit code be the single source of truth."""
    stderr_data = drain_stderr_excerpt(proc.stderr)
    rc = proc.wait()
    if not registry.drop_process(key, proc):
        return
    stderr_text = download_registry.scrub_secrets(
        (stderr_data or b"").decode("utf-8", "replace").strip(),
        hf_token = hf_token,
    )
    state = classify_exit(rc, cancel_requested = registry.cancel_requested(key))
    if state == "complete":
        registry.set_job(key, "complete")
        if stderr_text:
            if download_manifest.MANIFEST_DEGRADED_MARKER in stderr_text:
                logger.warning(
                    f"{log_prefix} complete with degraded diagnostics for "
                    f"{label}: {stderr_text}"
                )
            else:
                logger.info(
                    f"{log_prefix} worker diagnostics for {label}: {stderr_text}"
                )
        logger.info(f"{log_prefix} complete: {label}")
        # Defensive cleanup: canonical clear is at download-start in the
        # worker. This catches the rare case where the start clear failed
        # (transient disk error) but the download itself succeeded — no
        # stale marker should outlive a successful completion.
        if repo_type and repo_id:
            try:
                download_manifest.clear_cancel_marker(
                    repo_type,
                    repo_id,
                    download_registry.variant_from_key(key),
                )
            except Exception as exc:
                logger.debug(f"clear_cancel_marker failed for {repo_id} (rc=0): {exc}")
    elif state == "cancelled":
        registry.set_job(key, "cancelled")
        logger.info(f"{log_prefix} cancelled by user: {label} (rc={rc})")
        # Persist the original variant casing from metadata so the marker matches
        # the manifest; the lowercased job key is only a fallback once metadata is
        # gone. Mismatched casing would otherwise double-list the variant offline.
        metadata = registry.get_job_metadata(key)
        download_registry.persist_cancel_marker(
            repo_type,
            repo_id,
            metadata.variant
            if metadata is not None and metadata.variant
            else download_registry.variant_from_key(key),
            transport,
            logger = logger,
        )
    else:
        registry.set_job(
            key,
            "error",
            stderr_text or f"worker exited with code {rc}",
        )
        logger.error(
            f"{log_prefix} failed for {label} (rc={rc}): {stderr_text}",
        )


def kill_and_reap_process(
    proc: subprocess.Popen,
    *,
    label: str,
    logger,
    timeout: float = 10.0,
) -> None:
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    except Exception as exc:
        logger.warning(f"Cancel SIGKILL for {label} failed: {exc}")
    try:
        proc.wait(timeout = timeout)
    except subprocess.TimeoutExpired:
        logger.warning(f"Cancelled worker for {label} did not exit after SIGKILL")
    except Exception:
        pass


def register_worker(
    registry: download_registry.DownloadRegistry,
    key: str,
    proc: subprocess.Popen,
    *,
    hf_token: Optional[str],
    label: str,
    log_prefix: str,
    logger,
    repo_type: RepoType,
    repo_id: str,
    transport: str,
    watch_name: str,
) -> bool:
    if not registry.register_process(key, proc):
        kill_and_reap_process(proc, label = label, logger = logger)
        return False

    worker_token = hf_token

    def _watch() -> None:
        finalize_worker_exit(
            registry,
            key,
            proc,
            hf_token = worker_token,
            label = label,
            log_prefix = log_prefix,
            logger = logger,
            repo_type = repo_type,
            repo_id = repo_id,
            transport = transport,
        )
        if registry.get_job(key).state in ("error", "cancelled"):
            download_registry.purge_empty_marker_dir(
                repo_type,
                repo_id,
                download_registry.variant_from_key(key),
            )
        hf_cache_scan.invalidate_hf_cache_scans()

    threading.Thread(target = _watch, name = watch_name, daemon = True).start()
    return True


def launch_worker(
    registry: download_registry.DownloadRegistry,
    key: str,
    *,
    spawn: Callable[[], subprocess.Popen],
    hf_token: Optional[str],
    label: str,
    log_prefix: str,
    logger,
    repo_type: RepoType,
    repo_id: str,
    transport: str,
    watch_name: str,
) -> str:
    try:
        proc = spawn()
    except Exception as e:
        scrubbed = download_registry.scrub_secrets(str(e), hf_token = hf_token)
        logger.error(
            f"Failed to spawn {log_prefix.lower()} worker for {label}: {scrubbed}",
            exc_info = True,
        )
        registry.set_job(key, "error", scrubbed)
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to start {log_prefix.lower()}: {scrubbed}",
        ) from e
    register_worker(
        registry,
        key,
        proc,
        hf_token = hf_token,
        label = label,
        log_prefix = log_prefix,
        logger = logger,
        repo_type = repo_type,
        repo_id = repo_id,
        transport = transport,
        watch_name = watch_name,
    )
    return registry.get_job(key).state


def cancel_worker(
    registry: download_registry.DownloadRegistry,
    key: str,
    *,
    generation: Optional[int],
    repo_type: RepoType,
    repo_id: str,
    variant: Optional[str],
    label: str,
    logger,
) -> str:
    proc = registry.get_process(key)
    # No worker process yet: arm a pending cancel for the claim-to-register
    # window so register_process kills it on arrival.
    if proc is None:
        if registry.mark_pending_cancel(key, generation):
            return "cancelling"
        return registry.get_job(key).state
    # Worker already exited; its watcher will classify the real return code.
    # Arming a pending cancel here would let a genuine failure (or an external
    # signal kill) be mislabeled as a user cancel and persist a spurious marker.
    if proc.poll() is not None:
        return registry.get_job(key).state

    if not registry.request_cancel(key, proc, generation):
        return registry.get_job(key).state
    registry.persist_cancel_for_key(
        key,
        repo_type = repo_type,
        repo_id = repo_id,
        variant = variant,
    )

    try:
        proc.kill()
    except ProcessLookupError:
        pass
    except Exception as e:
        logger.warning(f"Cancel SIGKILL for {label} failed: {e}")

    return "cancelling"


def idle_status(
    registry: download_registry.DownloadRegistry,
    key: str,
    *,
    repo_type: RepoType,
    repo_id: Optional[str],
    variant: Optional[str],
) -> tuple[DownloadJobState, Optional[str], int]:
    state = registry.get_job(key)
    generation = registry.current_generation(key)
    if (
        state.state == "idle"
        and repo_id
        and download_manifest.has_cancel_marker(
            repo_type,
            repo_id,
            variant,
        )
    ):
        return ("cancelled", None, generation)
    return (state.state, state.error, generation)


def active_download_refs(
    registry: download_registry.DownloadRegistry,
    repo_id: Optional[str],
    *,
    with_variant: bool,
) -> list[ActiveDownload]:
    downloads: list[ActiveDownload] = []
    for ref in registry.active_job_refs(repo_id):
        metadata = ref.metadata
        if with_variant:
            ref_repo_id = (
                metadata.repo_id if metadata is not None else ref.key.split("::", 1)[0]
            )
            if metadata is not None:
                variant = metadata.variant
            else:
                _repo, sep, raw_variant = ref.key.partition("::")
                variant = raw_variant if sep and raw_variant else None
        else:
            ref_repo_id = metadata.repo_id if metadata is not None else ref.key
            variant = None
        downloads.append(
            ActiveDownload(
                repo_id = ref_repo_id,
                variant = variant,
                transport = metadata.transport if metadata is not None else None,
                state = ref.state,
                generation = ref.generation,
            )
        )
    return downloads
