# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
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

logger = logging.getLogger(__name__)


def backend_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_effective_use_xet(use_xet: bool) -> bool:
    """Downgrade an Xet request to HTTP when hf_xet is unavailable, so a defaulted
    or explicit Xet request never hard-fails on installs without the Xet extra."""
    if not use_xet:
        return False
    reason = download_registry.download_transport_unavailable_reason(
        download_registry.TRANSPORT_XET
    )
    if reason is None:
        return True
    logger.warning("Xet transport unavailable, falling back to HTTP: %s", reason)
    return False


def resolve_transport(use_xet: bool) -> str:
    transport = download_registry.TRANSPORT_XET if use_xet else download_registry.TRANSPORT_HTTP
    unavailable_reason = download_registry.download_transport_unavailable_reason(transport)
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

    XET and ``hf_transfer`` write chunks out of order, so their partials can't
    resume under a sequential writer; the HTTP path stays sequential so
    SIGKILL -> resume is byte-identical. ``protected_blob_hashes`` are blobs a
    concurrent same-repo peer is writing, excluded from the cache-prep purge so a
    shared ``.incomplete`` (e.g. bundled mmproj) is never deleted.
    """
    cwd = backend_dir()
    mode = download_registry.TRANSPORT_XET if use_xet else download_registry.TRANSPORT_HTTP
    env = os.environ.copy()
    if protected_blob_hashes:
        env["UNSLOTH_PROTECTED_BLOB_HASHES"] = ",".join(sorted(protected_blob_hashes))
    else:
        env.pop("UNSLOTH_PROTECTED_BLOB_HASHES", None)
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_DISABLE_XET"] = "0" if use_xet else "1"
    # No token in Studio settings: fall back to the backend's own HF_TOKEN so
    # private repos stay downloadable (needed while inkling repos are private).
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN") or None
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "0" if hf_token else "1"
    # hf_transfer's parallel Range chunks can leave sparse partials even in
    # "http" mode; disable so the worker's writer is always sequential.
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
    env["PYTHONPATH"] = f"{cwd}{os.pathsep}{existing_path}" if existing_path else str(cwd)
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

    Incremental reads keep the pipe from filling while bounding memory; long
    messages keep both ends since stderr prefixes often name the failing repo."""
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
    """Returncodes for intentional cancellation only (SIGKILL/SIGTERM/SIGINT); crash signals stay errors, and ``getattr`` tolerates Windows where these signals are absent."""
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
      cleanly with a resumable partial. In-app cancel uses untrappable SIGKILL
      and the OOM killer never produces 130, so 130 is always a resumable cancel.
    - rc killed by SIGKILL/SIGTERM/SIGINT: a cancel only when *we* asked for it.
      The OOM killer also sends SIGKILL, so an unrequested kill surfaces as error.
    - rc killed by SIGPIPE (or 128+SIGPIPE): parent pipe is gone; treated as
      cancelled.
    - any other non-zero rc (incl. crash signals): worker errored out.

    Windows has no POSIX signal exit encoding, so a user cancel can't be told from
    an error by code alone; there ``cancel_requested`` decides.
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
    cancel_marker_transport: Optional[str] = None,
    defer_error: bool = False,
) -> str:
    """Block until *proc* exits, then record the job's terminal state in
    *registry*. Drains and scrubs stderr first, then classifies the exit code.
    A no-op when the process was already dropped (e.g. superseded).

    No stall watchdog: huggingface_hub already times out chunk reads and raises
    a resumable error on a dead connection, so the worker's exit code is the
    single source of truth."""
    stderr_data = drain_stderr_excerpt(proc.stderr)
    rc = proc.wait()
    cancel_requested = registry.cancel_requested(key)
    if not registry.drop_process(key, proc):
        return "idle"
    stderr_text = download_registry.scrub_secrets(
        (stderr_data or b"").decode("utf-8", "replace").strip(),
        hf_token = hf_token,
    )
    state = classify_exit(rc, cancel_requested = cancel_requested)
    if state == "complete":
        registry.set_job(key, "complete")
        if transport == download_registry.TRANSPORT_HTTP:
            registry.update_job_transport(key, download_registry.TRANSPORT_HTTP)
        if stderr_text:
            if download_manifest.MANIFEST_DEGRADED_MARKER in stderr_text:
                logger.warning(
                    f"{log_prefix} complete with degraded diagnostics for "
                    f"{label}: {stderr_text}"
                )
            else:
                logger.info(f"{log_prefix} worker diagnostics for {label}: {stderr_text}")
        logger.info(f"{log_prefix} complete: {label}")
        # Defensive cleanup: the canonical clear is at download-start; this
        # catches the rare case where that failed but the download succeeded.
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
        # Read metadata before the terminal set_job so a concurrent eviction
        # can't drop it; the job key is the fallback variant label.
        metadata = registry.get_job_metadata(key)
        registry.set_job(key, "cancelled")
        logger.info(f"{log_prefix} cancelled: {label} (rc={rc})")
        download_registry.persist_cancel_marker(
            repo_type,
            repo_id,
            metadata.variant
            if metadata is not None and metadata.variant
            else download_registry.variant_from_key(key),
            cancel_marker_transport or transport,
            logger = logger,
        )
    else:
        if not defer_error:
            registry.set_job(
                key,
                "error",
                stderr_text or f"worker exited with code {rc}",
            )
        logger.error(
            f"{log_prefix} failed for {label} (rc={rc}): {stderr_text}",
        )
    return state


def _set_retry_failure_state(
    registry: download_registry.DownloadRegistry,
    key: str,
    error: str,
    *,
    repo_type: RepoType,
    repo_id: str,
    fallback_variant: Optional[str],
    fallback_transport: Optional[str],
    logger,
) -> str:
    state, metadata = registry.set_error_unless_cancelled(key, error)
    if state == "cancelled":
        download_registry.persist_cancel_marker(
            repo_type,
            repo_id,
            metadata.variant if metadata is not None and metadata.variant else fallback_variant,
            metadata.transport
            if metadata is not None and metadata.transport
            else fallback_transport,
            logger = logger,
        )
    return state


def _try_http_retry(
    registry: download_registry.DownloadRegistry,
    key: str,
    *,
    hf_token: Optional[str],
    label: str,
    log_prefix: str,
    logger,
    repo_type: RepoType,
    repo_id: str,
    watch_name: str,
) -> bool:
    """Reclaim *key* with HTTP transport and spawn a recovery worker.

    Returns ``True`` when the HTTP worker was successfully registered.
    Caller is responsible for ensuring this is only called when: the job is
    in ``"error"`` state, the original transport was XET, and HTTP is available.

    Derives variant and blob-hash metadata from the registry entry written by
    the original XET claim so callers do not re-construct worker arguments.
    Re-queries peer protection hashes at spawn time to reflect any concurrent
    sibling changes between the XET failure and this call.
    """
    original_metadata = registry.get_job_metadata(key)
    if original_metadata is None:
        logger.debug("%s XET retry skipped for %s; metadata unavailable", log_prefix, label)
        _set_retry_failure_state(
            registry,
            key,
            "XET retry skipped: metadata unavailable",
            repo_type = repo_type,
            repo_id = repo_id,
            fallback_variant = download_registry.variant_from_key(key),
            fallback_transport = download_registry.TRANSPORT_XET,
            logger = logger,
        )
        return False
    if original_metadata.transport != download_registry.TRANSPORT_XET:
        logger.debug(
            "%s XET retry skipped for %s; original transport was %s",
            log_prefix,
            label,
            original_metadata.transport,
        )
        _set_retry_failure_state(
            registry,
            key,
            f"XET retry skipped: original transport was {original_metadata.transport}",
            repo_type = repo_type,
            repo_id = repo_id,
            fallback_variant = original_metadata.variant,
            fallback_transport = original_metadata.transport,
            logger = logger,
        )
        return False
    variant = original_metadata.variant
    blob_hashes = original_metadata.blob_hashes
    progress_blob_hashes = original_metadata.progress_blob_hashes
    completed_baseline_bytes = (
        download_registry.completed_blob_bytes(
            repo_type,
            repo_id,
            progress_blob_hashes,
        )
        if progress_blob_hashes
        else 0
    )
    generation = registry.current_generation(key)
    registry.release_active_slot(key)
    while True:
        if registry.cancel_requested(key):
            _set_retry_failure_state(
                registry,
                key,
                "HTTP retry cancelled before reclaiming the download slot",
                repo_type = repo_type,
                repo_id = repo_id,
                fallback_variant = variant,
                fallback_transport = original_metadata.transport,
                logger = logger,
            )
            return False

        claimed, conflict_state = registry.claim(
            key,
            download_registry.TRANSPORT_HTTP,
            repo_type = repo_type,
            repo_id = repo_id,
            variant = variant,
            blob_hashes = blob_hashes,
            progress_blob_hashes = progress_blob_hashes,
            completed_baseline_bytes = completed_baseline_bytes,
            generation = generation,
            replace_active = True,
            cancel_marker_transport = original_metadata.transport,
        )
        if claimed:
            break
        if conflict_state == "deleting":
            logger.debug(
                "%s XET retry claim rejected for %s; repo is being deleted",
                log_prefix,
                label,
            )
            _set_retry_failure_state(
                registry,
                key,
                "HTTP retry could not reclaim the download slot",
                repo_type = repo_type,
                repo_id = repo_id,
                fallback_variant = variant,
                fallback_transport = original_metadata.transport,
                logger = logger,
            )
            return False
        logger.debug(
            "%s XET retry claim blocked for %s by active sibling state %s; waiting",
            log_prefix,
            label,
            conflict_state,
        )
        time.sleep(0.05)

    args: list[str] = ["--repo-id", repo_id]
    if repo_type == "dataset":
        args.append("--dataset")
    elif variant:
        args.extend(["--variant", variant])

    # Re-query at spawn time: sibling state may have changed since XET failed.
    peer_hashes = registry.peer_blob_hashes(key) if variant else frozenset()

    logger.warning(
        "%s XET worker failed for %s; retrying over HTTP",
        log_prefix,
        label,
    )
    try:
        proc = spawn_worker(
            args,
            hf_token,
            use_xet = False,
            protected_blob_hashes = peer_hashes or None,
        )
    except Exception as exc:
        scrubbed = download_registry.scrub_secrets(str(exc), hf_token = hf_token)
        logger.error(
            "%s HTTP retry spawn failed for %s: %s",
            log_prefix,
            label,
            scrubbed,
        )
        registry.update_job_transport(key, original_metadata.transport)
        _set_retry_failure_state(
            registry,
            key,
            scrubbed,
            repo_type = repo_type,
            repo_id = repo_id,
            fallback_variant = variant,
            fallback_transport = original_metadata.transport,
            logger = logger,
        )
        return False

    return register_worker(
        registry,
        key,
        proc,
        hf_token = hf_token,
        label = label,
        log_prefix = log_prefix,
        logger = logger,
        repo_type = repo_type,
        repo_id = repo_id,
        transport = download_registry.TRANSPORT_HTTP,
        cancel_marker_transport = original_metadata.transport,
        watch_name = watch_name,
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
    cancel_marker_transport: Optional[str] = None,
    watch_name: str,
) -> bool:
    if not registry.register_process(key, proc):
        kill_and_reap_process(proc, label = label, logger = logger)
        return False

    worker_token = hf_token

    def _watch() -> None:
        try:
            can_retry_http = (
                transport == download_registry.TRANSPORT_XET
                and download_registry.download_transport_unavailable_reason(
                    download_registry.TRANSPORT_HTTP
                )
                is None
            )
            state = finalize_worker_exit(
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
                cancel_marker_transport = cancel_marker_transport,
                defer_error = can_retry_http,
            )
            # XET-to-HTTP recovery: when a non-cancelled XET worker fails and
            # HTTP is available, attempt one automatic retry over HTTP.  The
            # transport check is the recursion guard: an HTTP worker that errors
            # never satisfies `transport == TRANSPORT_XET`, so it stays terminal.
            if can_retry_http and state == "error":
                _try_http_retry(
                    registry,
                    key,
                    hf_token = worker_token,
                    label = label,
                    log_prefix = log_prefix,
                    logger = logger,
                    repo_type = repo_type,
                    repo_id = repo_id,
                    watch_name = watch_name,
                )
        except Exception:
            # finalize_worker_exit is the only thing that clears running/cancelling;
            # if it raises, force a terminal state so claim() isn't blocked until restart.
            logger.exception("download watcher crashed for %s", key)
            # finalize may have raised before reaping the worker; terminate the
            # still-registered Popen first, else the terminal set_job clears the
            # repo guard and a live worker would race a retry on the same repo.
            try:
                kill_and_reap_process(proc, label = label, logger = logger)
            except Exception:
                logger.exception("failed to reap worker after watcher crash for %s", key)
            try:
                registry.drop_process(key, proc)
            except Exception:
                logger.exception("failed to drop worker after watcher crash for %s", key)
            try:
                registry.set_job(key, "error", "download watcher crashed")
            except Exception:
                logger.exception("failed to mark %s errored after watcher crash", key)
        finally:
            try:
                if registry.get_job(key).state in ("error", "cancelled"):
                    download_registry.purge_empty_marker_dir(
                        repo_type,
                        repo_id,
                        download_registry.variant_from_key(key),
                    )
            except Exception:
                logger.exception("post-finalize marker cleanup failed for %s", key)
            finally:
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
    label: str,
    logger,
) -> str:
    proc = registry.get_process(key)
    # No worker process yet: arm a pending cancel so register_process kills it on
    # arrival during the claim-to-register window.
    if proc is None:
        if registry.mark_pending_cancel(key, generation):
            return "cancelling"
        return registry.get_job(key).state
    # Worker already exited; let its watcher classify the real return code.
    if proc.poll() is not None:
        get_metadata = getattr(registry, "get_job_metadata", None)
        metadata = get_metadata(key) if get_metadata is not None else None
        can_retry_http = (
            metadata is not None
            and metadata.transport == download_registry.TRANSPORT_XET
            and download_registry.download_transport_unavailable_reason(
                download_registry.TRANSPORT_HTTP
            )
            is None
        )
        if can_retry_http and registry.mark_pending_cancel(key, generation):
            return "cancelling"
        return registry.get_job(key).state

    if not registry.request_cancel(key, proc, generation):
        return registry.get_job(key).state
    # No eager marker: finalize_worker_exit writes it on a "cancelled" exit.
    # Persisting before the kill races a clean completion and strands a stale marker.
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
    registry: download_registry.DownloadRegistry, repo_id: Optional[str], *, with_variant: bool
) -> list[ActiveDownload]:
    downloads: list[ActiveDownload] = []
    for ref in registry.active_job_refs(repo_id):
        metadata = ref.metadata
        if with_variant:
            ref_repo_id = metadata.repo_id if metadata is not None else ref.key.split("::", 1)[0]
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
