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
from typing import Callable, Mapping, Optional, Sequence

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


def resolve_requested_use_xet(transport_mode: Optional[str], use_xet: bool) -> tuple[bool, str]:
    """Turn a download request's transport preference into ``(use_xet, reason)``.

    ``transport_mode`` is the current field; ``use_xet`` is the older boolean, still honoured so an
    older frontend or scripted caller keeps working. An explicit "xet" is respected even on a
    machine the health check dislikes, but still gets the memory caps and the stall fallback.
    """
    mode = (transport_mode or "").strip().lower()
    if mode == download_registry.TRANSPORT_HTTP:
        return (False, "HTTP (requested)")
    if mode == download_registry.TRANSPORT_XET:
        return (resolve_effective_use_xet(True), "Xet (requested)")
    if mode == download_registry.TRANSPORT_AUTO:
        return resolve_auto_use_xet()
    resolved = resolve_effective_use_xet(use_xet)
    return (resolved, "Xet" if resolved else "HTTP")


def resolve_auto_use_xet() -> tuple[bool, str]:
    """Pick a transport for a download the user left on "Auto". Returns ``(use_xet, reason)``.

    Server-side on purpose: only the backend can see this machine's RAM, its hf_xet build, and
    whether Xet has been failing here. Probing IS allowed here, unlike in the capabilities endpoint
    the UI polls on render, because this runs once per download request and the verdict is memoized.
    """
    if not resolve_effective_use_xet(True):
        return (False, "hf_xet is not installed")
    try:
        from utils.hf_xet_fallback import xet_health
        health = xet_health()
    except Exception as exc:  # noqa: BLE001 - never let a health probe block a download
        logger.debug("Xet health probe failed, defaulting to Xet: %s", exc)
        health, reason = None, "Xet (health check unavailable)"
    else:
        # Older unsloth_zoo without the health module: no opinion, keep the existing default.
        reason = "Xet" if health is None else str(health.reason)
    if health is not None and not health.use_xet:
        return (False, reason)
    if _health_is_forced(health):
        # UNSLOTH_FORCE_XET=1 has to win here, since the zoo's off switches already win above and the pair
        # would otherwise be asymmetric; buffers are still clamped to free RAM.
        return (True, reason)
    # Free RAM is separate evidence, read from a different zoo module: a missing health module says
    # nothing about whether this machine can afford Xet right now.
    pressure = _memory_pressure_reason()
    if pressure is not None:
        return (False, pressure)
    return (True, reason)


def _health_is_forced(health) -> bool:
    """``UNSLOTH_FORCE_XET``-style verdict? Delegates so this and the capabilities probe stand down
    on exactly the same evidence; a degraded shim answers False, which keeps the RAM gate on."""
    try:
        from utils.hf_xet_fallback import xet_health_is_forced
        return xet_health_is_forced(health)
    except Exception as exc:  # noqa: BLE001 - an unreadable verdict is not an override
        logger.debug("Could not read the Xet health source: %s", exc)
        return False


def _memory_pressure_reason() -> Optional[str]:
    """Free-RAM verdict for an API caller that sends "auto".

    The Unsloth UI never reaches here: it resolves Auto through the capabilities probe and submits
    the concrete xet/http, so that probe applies the same rule. Shared helper, so the two agree."""
    try:
        from utils.hf_xet_fallback import free_ram_pressure_reason
        return free_ram_pressure_reason()
    except Exception as exc:  # noqa: BLE001 - a probe must not decide the transport by crashing
        logger.debug("Free-RAM probe failed, leaving the Xet verdict alone: %s", exc)
        return None


def _allow_high_performance() -> bool:
    """Legacy opt-in, still honoured for installs whose unsloth_zoo cannot size the worker itself."""
    return os.environ.get("UNSLOTH_XET_ALLOW_HIGH_PERFORMANCE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def resolve_transport(use_xet: bool) -> str:
    transport = download_registry.TRANSPORT_XET if use_xet else download_registry.TRANSPORT_HTTP
    unavailable_reason = download_registry.download_transport_unavailable_reason(transport)
    if unavailable_reason is not None:
        raise HTTPException(status_code = 400, detail = unavailable_reason)
    return transport


def write_files_manifest(files: Sequence[str]) -> str:
    """Stage a scoped job's file list in a temp JSON file and return its path.

    The worker deletes it after reading. A pipeline repo lists hundreds of files, well past
    what is comfortable on a command line."""
    import json
    import tempfile

    handle = tempfile.NamedTemporaryFile(
        mode = "w", suffix = ".json", prefix = "unsloth-dl-files-", delete = False, encoding = "utf-8"
    )
    with handle:
        json.dump(list(files), handle)
    return handle.name


def spawn_worker(
    args: list[str],
    hf_token: Optional[str],
    *,
    use_xet: bool,
    protected_blob_hashes: Optional[frozenset[str]] = None,
    cache_env: Optional[Mapping[str, str]] = None,
    allow_ambient_token: bool = True,
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
    from utils.hf_cache_settings import get_hf_cache_paths

    env = get_hf_cache_paths().child_env()
    if cache_env is not None:
        env.update(cache_env)
    if protected_blob_hashes:
        env["UNSLOTH_PROTECTED_BLOB_HASHES"] = ",".join(sorted(protected_blob_hashes))
    else:
        env.pop("UNSLOTH_PROTECTED_BLOB_HASHES", None)
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_DISABLE_XET"] = "0" if use_xet else "1"
    if use_xet:
        # hf_xet sizes its buffers from constants and reads its config natively at import, so the worker's
        # env is sized here, and by unsloth_zoo so there is one rule and not two (UNSLOTH_XET_FORCE_CAPS=1
        # bounds the machine regardless): the hand-rolled copies drifted, and on a 2TB host the worker got
        # a laptop's 24GB buffer and ran 3.4x slower.
        from utils import hf_xet_fallback

        # The worker's own cache, which the sizing measures: this backend's env may still name the one it
        # started with, since moving the cache in Settings does not rewrite the live process.
        sized = hf_xet_fallback.apply_xet_env(env, env.get("HF_HUB_CACHE"))
        if sized is None and not _allow_high_performance():
            # No tuning module: that unsloth_zoo is also the one setting HF_XET_HIGH_PERFORMANCE=1 at import,
            # and the inherited "1" would hand the worker a 64GB ceiling, since xet-core applies the preset
            # AFTER reading the environment.
            for key in ("HF_XET_HIGH_PERFORMANCE", "HF_XET_HP"):
                env[key] = "0"
    # Fall back to the backend's own HF_TOKEN so private repos stay downloadable, but never for a repo
    # an API caller named: that would lend them the owner's identity.
    if not hf_token and allow_ambient_token:
        hf_token = os.environ.get("HF_TOKEN") or None
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "0" if hf_token else "1"
    # hf_transfer's parallel Range chunks can leave sparse partials even in "http" mode, so disable it
    # and keep the worker's writer sequential.
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
    proc = None
    try:
        proc = subprocess.Popen(
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
        return proc
    finally:
        if use_xet:
            # Tie the sizing's RAM reservation to the worker so it frees when the worker exits and a sibling
            # sizes against the remainder; a spawn that raised passes None, dropping the reservation.
            from utils import hf_xet_fallback
            hf_xet_fallback.bind_worker_budget(proc.pid if proc is not None else None)


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
    metadata = registry.get_job_metadata(key)
    state = classify_exit(rc, cancel_requested = cancel_requested)
    if state == "complete":
        hf_cache_scan.invalidate_hf_cache_scans()
        registry.set_job(key, "complete")
        # Where /v1 learns a new model exists: its resolver answers from a cached scan with no watcher.
        # Models only, since noting a dataset id as a local model would refuse a bare request naming it.
        if repo_type == "model":
            try:
                from core.inference.local_model_resolver import (
                    invalidate_index,
                    note_downloaded,
                    warm_index_soon,
                )

                note_downloaded(repo_id)
                invalidate_index(additions_only = True)
                # Rebuild here, not on the first request, to keep the scan off the request path.
                warm_index_soon()
            except Exception:
                pass
        if transport == download_registry.TRANSPORT_HTTP:
            registry.update_job_transport(key, download_registry.TRANSPORT_HTTP)
        if stderr_text:
            if download_manifest.MANIFEST_DEGRADED_MARKER in stderr_text:
                logger.warning(
                    f"{log_prefix} complete with degraded diagnostics for {label}: {stderr_text}"
                )
            else:
                logger.info(f"{log_prefix} worker diagnostics for {label}: {stderr_text}")
        logger.info(f"{log_prefix} complete: {label}")
        # Defensive cleanup: the canonical clear is at download-start; this catches the rare case where that
        # failed but the download succeeded.
        if repo_type and repo_id:
            try:
                download_manifest.clear_cancel_marker(
                    repo_type,
                    repo_id,
                    download_registry.variant_from_key(key),
                    hub_cache = metadata.hub_cache if metadata is not None else None,
                )
            except Exception as exc:
                logger.debug(f"clear_cancel_marker failed for {repo_id} (rc=0): {exc}")
    elif state == "cancelled":
        # Read metadata before the terminal set_job so a concurrent eviction cannot drop it; the job key is
        # the fallback variant label.
        registry.set_job(key, "cancelled")
        logger.info(f"{log_prefix} cancelled: {label} (rc={rc})")
        download_registry.persist_cancel_marker(
            repo_type,
            repo_id,
            metadata.variant
            if metadata is not None and metadata.variant
            else download_registry.variant_from_key(key),
            cancel_marker_transport or transport,
            hub_cache = metadata.hub_cache if metadata is not None else None,
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
            hub_cache = metadata.hub_cache if metadata is not None else None,
            logger = logger,
        )
    return state


# "no byte baseline was handed to us, sample one" -- distinct from a sampled None (unmeasurable).
_UNSAMPLED = object()


def _is_data_phase_stall(message: str) -> bool:
    """Did the watchdog trip AFTER bytes had flowed? Delegates to the shared rule so "worth
    retrying" and "worth recording against the machine" can never disagree, with the same literal
    check inline as the degraded fallback."""
    try:
        from utils.hf_xet_fallback import is_data_phase_stall
        return is_data_phase_stall(message)
    except Exception:  # noqa: BLE001
        return "did not start" not in (message or "")


def _xet_attempt_budget() -> int:
    """How many XET workers one download may spend before HTTP. Degrades to 1 (the pre-retry
    ladder) if the shared helper cannot be imported, so a broken unsloth_zoo never turns into an
    extra stall the user waits through."""
    try:
        from utils.hf_xet_fallback import xet_attempts
        return xet_attempts()
    except Exception:  # noqa: BLE001
        return 1


def _try_transport_retry(
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
    retry_transport: str = download_registry.TRANSPORT_HTTP,
    xet_attempt: int = 1,
    pending_xet_failure: Optional[str] = None,
    bytes_before: "Optional[int]" = _UNSAMPLED,
    allow_ambient_token: bool = True,
) -> bool:
    """Reclaim *key* under *retry_transport* and spawn a recovery worker.

    Returns ``True`` when the recovery worker was successfully registered.
    Caller is responsible for ensuring this is only called when: the job is
    in ``"error"`` state, the original transport was XET, and the target
    transport is available.

    Two directions share this body. ``TRANSPORT_HTTP`` is terminal: the
    transport changes, so the worker's own ``prepare_cache_for_transport``
    purges the XET partial an HTTP resume would corrupt. ``TRANSPORT_XET`` is
    the stall retry: same transport, same marker, one more child, bounded by
    *xet_attempt* rather than by the transport check that stops the HTTP one.

    *allow_ambient_token* rides along unchanged, so a job that started anonymous cannot pick
    the backend's own HF_TOKEN up on a lower rung.

    *pending_xet_failure* is a stall verdict held back from the health tracker
    and carried into the next worker, so a download that recovers on its second
    Xet attempt reports nothing and one that does not reports exactly one
    failure. *bytes_before* is the ORIGINAL pre-Xet baseline: resampling would
    fold the killed worker's partial writes in and make a recovered attempt
    read as a cached no-op.

    Derives variant and blob-hash metadata from the registry entry written by
    the original XET claim so callers do not re-construct worker arguments.
    Re-queries peer protection hashes at spawn time to reflect any concurrent
    sibling changes between the XET failure and this call.
    """
    retry_over_xet = retry_transport == download_registry.TRANSPORT_XET
    retry_name = "XET" if retry_over_xet else "HTTP"

    def _give_up() -> None:
        """Ladder ends here, so a held-back verdict must still reach the health tracker: otherwise a
        stall followed by a failed reclaim is silently forgiven."""
        if pending_xet_failure:
            _record_xet_failure(pending_xet_failure, logger)

    original_metadata = registry.get_job_metadata(key)
    if original_metadata is None:
        logger.debug("%s XET retry skipped for %s; metadata unavailable", log_prefix, label)
        _give_up()
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
        _give_up()
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
            root = Path(original_metadata.hub_cache) if original_metadata.hub_cache else None,
        )
        if progress_blob_hashes
        else 0
    )
    generation = registry.current_generation(key)
    registry.release_active_slot(key)
    while True:
        if registry.cancel_requested(key):
            # A user cancel is not evidence against Xet: a held-back stall is dropped, not charged.
            _set_retry_failure_state(
                registry,
                key,
                f"{retry_name} retry cancelled before reclaiming the download slot",
                repo_type = repo_type,
                repo_id = repo_id,
                fallback_variant = variant,
                fallback_transport = original_metadata.transport,
                logger = logger,
            )
            return False

        claimed, conflict_state = registry.claim(
            key,
            # A XET retry re-claims the SAME transport, the permissive case in claim(): only a cross-transport
            # claim can mix an HTTP resume with a XET rewrite over one shared blob.
            retry_transport,
            repo_type = repo_type,
            repo_id = repo_id,
            variant = variant,
            blob_hashes = blob_hashes,
            progress_blob_hashes = progress_blob_hashes,
            completed_baseline_bytes = completed_baseline_bytes,
            generation = generation,
            replace_active = True,
            cancel_marker_transport = original_metadata.transport,
            hub_cache = original_metadata.hub_cache,
            xet_cache = original_metadata.xet_cache,
            # Carry the scoped file list across the reclaim: the record it overwrites is what a later start is
            # compared against, so dropping it makes an identical start 409 instead of adopting.
            scoped_files = original_metadata.scoped_files or None,
        )
        if claimed:
            break
        # An STT owner is not an active job, so mark_pending_cancel cannot reach this loop and waiting it
        # out would be an uninterruptible spin.
        if conflict_state in ("deleting", "repository_owned"):
            logger.debug(
                "%s XET retry claim rejected for %s; blocked by %s",
                log_prefix,
                label,
                conflict_state,
            )
            _give_up()
            _set_retry_failure_state(
                registry,
                key,
                f"{retry_name} retry could not reclaim the download slot",
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
    # A scoped job must retry as the SAME scoped download; without its file list the recovery worker
    # would fall through to a full snapshot.
    if original_metadata.scoped_files:
        args.extend(["--files-json", write_files_manifest(original_metadata.scoped_files)])

    # Re-query at spawn time: sibling state may have changed since XET failed.
    peer_hashes = registry.peer_blob_hashes(key) if variant else frozenset()

    if retry_over_xet:
        logger.warning(
            "%s XET worker stalled for %s; retrying on XET (attempt %d of %d)",
            log_prefix,
            label,
            xet_attempt,
            _xet_attempt_budget(),
        )
    else:
        logger.warning(
            "%s XET worker failed for %s; retrying over HTTP",
            log_prefix,
            label,
        )
    try:
        cache_env = (
            {
                "HF_HUB_CACHE": original_metadata.hub_cache,
                "HF_XET_CACHE": original_metadata.xet_cache,
            }
            if original_metadata.hub_cache and original_metadata.xet_cache
            else None
        )
        spawn_kwargs = {
            "use_xet": retry_over_xet,
            "protected_blob_hashes": peer_hashes or None,
            "allow_ambient_token": allow_ambient_token,
        }
        if cache_env is not None:
            spawn_kwargs["cache_env"] = cache_env
        proc = spawn_worker(
            args,
            hf_token,
            **spawn_kwargs,
        )
    except Exception as exc:
        scrubbed = download_registry.scrub_secrets(str(exc), hf_token = hf_token)
        logger.error(
            "%s %s retry spawn failed for %s: %s",
            log_prefix,
            retry_name,
            label,
            scrubbed,
        )
        registry.update_job_transport(key, original_metadata.transport)
        if retry_over_xet:
            # A failed EXTRA worker spawn is a local failure, not a verdict on the download, so drop to the
            # rung this job would have taken without the retry instead of stranding it in "error".
            logger.warning(
                "%s XET retry could not be spawned for %s; falling back to HTTP",
                log_prefix,
                label,
            )
            return _try_transport_retry(
                registry,
                key,
                hf_token = hf_token,
                label = label,
                log_prefix = log_prefix,
                logger = logger,
                repo_type = repo_type,
                repo_id = repo_id,
                watch_name = watch_name,
                retry_transport = download_registry.TRANSPORT_HTTP,
                xet_attempt = xet_attempt,
                pending_xet_failure = pending_xet_failure,
                bytes_before = bytes_before,
                allow_ambient_token = allow_ambient_token,
            )
        _give_up()
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
        transport = retry_transport,
        cancel_marker_transport = original_metadata.transport,
        watch_name = watch_name,
        xet_attempt = xet_attempt,
        pending_xet_failure = pending_xet_failure,
        bytes_before = bytes_before,
        allow_ambient_token = allow_ambient_token,
    )


def _try_http_retry(registry: download_registry.DownloadRegistry, key: str, **kwargs) -> bool:
    """Reclaim *key* over HTTP: the terminal rung of the recovery ladder. Thin alias kept because
    "retry over HTTP" is what most call sites mean and read better than the transport keyword."""
    return _try_transport_retry(
        registry, key, retry_transport = download_registry.TRANSPORT_HTTP, **kwargs
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


def _record_xet_failure(reason: str, logger) -> None:
    """Tell the health tracker a Xet transfer failed here; best-effort, never fatal to a download."""
    try:
        from utils.hf_xet_fallback import record_xet_outcome
        record_xet_outcome(False, reason)
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not record Xet outcome: %s", exc)


def _repo_bytes_on_disk(repo_type, repo_id: str, cache_dir) -> "Optional[int]":
    """Bytes present for this repo, or None when unmeasurable.

    Tells an actual Xet transfer from a job that found everything cached: the worker reports only an
    exit code, and the .transport marker is written before the transfer, so there is no other signal.
    """
    try:
        from utils.hf_xet_fallback import get_hf_download_state
        state = get_hf_download_state([repo_id], repo_type = repo_type, cache_dir = cache_dir)
    except Exception:  # noqa: BLE001 - a missing measurement must never fail a download
        return None
    return None if state is None else int(state[0])


def _sweep_ownership(metadata, own_blob_hashes, owned_for_sweep, repo_type: str, repo_id: str):
    """(owned hashes, owns-everything) for a job that has just reached a terminal state.

    A variant job whose API-side hash pre-resolution failed carries a non-null variant with an
    EMPTY hash set, so neither claim applies and its own fresh partial waits out a grace that
    exists to guess at a writer we already know is dead. The worker wrote a manifest naming the
    files it fetched, so read the ownership back off that.
    """
    if own_blob_hashes is None:
        return None, True
    if owned_for_sweep:
        return frozenset(owned_for_sweep), False
    from hub.utils import download_manifest

    manifest = download_manifest.read_manifest(
        repo_type,
        repo_id,
        getattr(metadata, "variant", None),
        hub_cache = getattr(metadata, "hub_cache", None),
    )
    recovered = {
        expected.sha256
        for expected in getattr(manifest, "expected_files", ()) or ()
        if getattr(expected, "sha256", None)
    }
    return frozenset(recovered), False


def _job_bytes_on_disk(repo_type, repo_id: str, cache_dir, blob_hashes) -> "Optional[int]":
    """Bytes THIS job owns, or None when unmeasurable.

    Scoped to the variant's own blobs when the claim resolved them: the registry lets two
    same-transport GGUF variants of one repo run concurrently over one blobs/ dir, so a repo-wide
    measure would credit a cached no-op worker with its sibling's bytes, clearing a legitimate stall
    streak and flipping an already demoted verdict back to Xet. Non-variant model jobs and dataset
    jobs cannot have a concurrent same-repo sibling (claim() rejects those), so they stay repo-wide.
    """
    if blob_hashes is None:
        return _repo_bytes_on_disk(repo_type, repo_id, cache_dir)
    if not blob_hashes:
        return None  # variant job with no resolved hashes: unmeasurable, so never clear the streak
    try:
        return download_registry.completed_blob_bytes(
            repo_type,
            repo_id,
            blob_hashes,
            root = Path(cache_dir) if cache_dir else None,
        )
    except Exception:  # noqa: BLE001 - a missing measurement must never fail a download
        return None


def _record_xet_success(logger) -> None:
    """Tell the health tracker a Xet transfer completed here, which resets the failure streak."""
    try:
        from utils.hf_xet_fallback import record_xet_outcome
        record_xet_outcome(True, "Xet download completed")
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not record Xet outcome: %s", exc)


def _start_stall_watchdog(
    registry: download_registry.DownloadRegistry,
    key: str,
    proc: subprocess.Popen,
    *,
    repo_type: RepoType,
    repo_id: str,
    label: str,
    log_prefix: str,
    logger,
    on_stall: Callable[[str], None],
):
    """Kill *proc* if its download stops making byte-level progress. Returns a stop event, or
    ``None`` when no watchdog could be started.

    SIGKILL rather than a polite signal: the worker traps SIGTERM and exits 130 ("cancelled"), which
    would record a user cancel and skip the recovery ladder. An untrapped kill lands as "error",
    which is what triggers the retry.

    What survives the kill depends on the transport the retry picks. Over HTTP the writer is
    sequential and the partial genuinely resumes. Over XET it does not: hf_xet reconstructs a file
    from offset zero rather than resuming it, so the recovery worker's own
    ``prepare_cache_for_transport(..., "xet", ...)`` purges the partial on startup and re-fetches the
    in-flight file. Every file already finalized into a blob is kept either way, and the worker runs
    ``snapshot_download(max_workers=1)``, so at most one file is ever replayed.
    """
    try:
        from utils.hf_xet_fallback import start_watchdog
    except Exception as exc:  # noqa: BLE001 - degraded unsloth_zoo: keep the old behaviour
        logger.debug("%s stall watchdog unavailable for %s: %s", log_prefix, label, exc)
        return None

    metadata = registry.get_job_metadata(key)
    cache_dir = getattr(metadata, "hub_cache", None) if metadata is not None else None

    def _on_stall(message: str) -> None:
        logger.warning(
            "%s %s for %s; killing the worker to retry over HTTP", log_prefix, message, label
        )
        on_stall(message)
        try:
            # Kill only: the _watch thread is already reaping this process, so a wait() here would race it for
            # the exit status.
            proc.kill()
        except ProcessLookupError:
            pass
        except Exception:
            logger.exception("%s failed to kill stalled worker for %s", log_prefix, label)

    try:
        return start_watchdog(
            repo_ids = [repo_id],
            repo_type = repo_type,
            cache_dir = cache_dir,
            on_stall = _on_stall,
            child_pid = proc.pid,
            # Scope the measurement to partials this worker holds open; otherwise the shared helper stays
            # repo-wide, child_pid does nothing, and two concurrent same-transport GGUF variants of one repo
            # reset each other's stall timer. The DATA clock only: before its first byte a variant is still
            # covered by the repo-wide peer-progress check.
            watch_new_partials_only = True,
            # The shared 90s zero-byte default assumes a single-file download whose pre-byte phase is one
            # HEAD; snapshot_download(max_workers=1) does a model_info lookup plus one sequential HEAD per
            # file, which for an already-cached repo is the entire job with no byte written.
            connect_timeout = 600.0,
            xet_disabled = False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("%s could not start stall watchdog for %s: %s", log_prefix, label, exc)
        return None


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
    bytes_before: "Optional[int]" = _UNSAMPLED,
    xet_attempt: int = 1,
    pending_xet_failure: Optional[str] = None,
    allow_ambient_token: bool = True,
) -> bool:
    """Watch *proc* to completion and drive the recovery ladder off its exit.

    *xet_attempt* (1-based) bounds the XET->XET stall retry, the way ``transport == TRANSPORT_XET``
    bounds the terminal XET->HTTP one. *pending_xet_failure* is an earlier attempt's stall verdict,
    held back from the health tracker until the XET phase ends so one download can never spend the
    two consecutive failures that demote a machine. *allow_ambient_token* is the token policy this
    job was started under, carried onto every rung of the ladder.
    """
    if not registry.register_process(key, proc):
        kill_and_reap_process(proc, label = label, logger = logger)
        return False

    worker_token = hf_token
    # getattr, not a direct call: test doubles and older registries do not all implement this.
    _get_metadata = getattr(registry, "get_job_metadata", None)
    _metadata = _get_metadata(key) if callable(_get_metadata) else None
    _cache_dir = getattr(_metadata, "hub_cache", None) if _metadata is not None else None
    # The variant's own blobs, so a sibling variant writing into the same repo is not counted as this job's progress.
    _own_blob_hashes = (
        getattr(_metadata, "blob_hashes", frozenset())
        if getattr(_metadata, "variant", None)
        else None
    )
    # Companions such as a shared mmproj live only in the progress set, and this worker was writing one
    # as much as it was writing the main quant.
    _owned_for_sweep = (
        getattr(_metadata, "progress_blob_hashes", None) or _own_blob_hashes
        if _own_blob_hashes is not None
        else None
    )
    # Sampled before the worker can write, as launch_worker samples BEFORE spawn(): sampling later
    # would race a fast child that already finalized its blobs, making a real transfer look like a
    # no-op and leaving the streak uncleared.
    _bytes_before = (
        _job_bytes_on_disk(repo_type, repo_id, _cache_dir, _own_blob_hashes)
        if bytes_before is _UNSAMPLED
        else bytes_before
    )

    def _watch() -> None:
        stalled: list[str] = []
        watchdog_stop = None
        try:
            can_retry_http = (
                transport == download_registry.TRANSPORT_XET
                and download_registry.download_transport_unavailable_reason(
                    download_registry.TRANSPORT_HTTP
                )
                is None
            )
            # A Xet transfer that hangs with no progress and no error never produces an exit code, so watch
            # the cache for byte-level progress and kill it; the SIGKILL surfaces as "error".
            if can_retry_http:
                watchdog_stop = _start_stall_watchdog(
                    registry,
                    key,
                    proc,
                    repo_type = repo_type,
                    repo_id = repo_id,
                    label = label,
                    log_prefix = log_prefix,
                    logger = logger,
                    on_stall = stalled.append,
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
            if watchdog_stop is not None:
                # Stop measuring once the worker is reaped: post-download symlinking and verification make no byte-
                # level progress and must not read as a stall.
                watchdog_stop.set()
            # Spend one more XET worker only on a DATA-phase verdict: retrying a pre-byte trip would buy a
            # second full 600s connect window, and that trip is as likely slow metadata as a broken Xet.
            retry_xet = (
                can_retry_http
                and state == "error"
                and bool(stalled)
                and _is_data_phase_stall(stalled[0])
                and xet_attempt < _xet_attempt_budget()
            )
            # Evidence for the WHOLE Xet phase, reported once when the phase ends.
            xet_failure = pending_xet_failure
            if stalled:
                # Exclude only the PRE-BYTE trip: "did not start" is as likely slow metadata, a queue of HEADs or a
                # cache lock as a broken Xet, and two recorded failures pin this machine to HTTP for 24h.
                # state == "error" is the other half: the watchdog appends its verdict before the kill lands, so a
                # worker that completed or was cancelled in that instant would be charged a failure it did not
                # earn.
                if state == "error" and _is_data_phase_stall(stalled[0]):
                    xet_failure = stalled[0]
                else:
                    logger.debug(
                        "%s not recording a Xet health failure (state=%s): %s",
                        log_prefix,
                        state,
                        stalled[0],
                    )
            if state == "cancelled" or (
                state == "complete" and transport == download_registry.TRANSPORT_XET
            ):
                # A XET completion proved Xet works here and a cancel was never evidence against it; an HTTP
                # completion proves nothing about Xet, so a verdict carried onto that rung is still charged.
                xet_failure = None
            if xet_failure is not None and not retry_xet:
                # Xet phase over (HTTP next, or nothing left to try): report it, once.
                _record_xet_failure(xet_failure, logger)
                xet_failure = None
            if not stalled and transport == download_registry.TRANSPORT_XET and state == "complete":
                # Clear the streak so "two failures in a row" means in a row, but only for a job that moved bytes:
                # a fully cached repo exits 0 without touching the network, and clearing an earned demotion on
                # that puts a bad machine back on Xet.
                bytes_after = _job_bytes_on_disk(repo_type, repo_id, _cache_dir, _own_blob_hashes)
                if (
                    _bytes_before is not None
                    and bytes_after is not None
                    and bytes_after > _bytes_before
                ):
                    _record_xet_success(logger)
            # One guard per direction: transport == TRANSPORT_XET makes the HTTP rung terminal, xet_attempt
            # bounds the XET one.
            if can_retry_http and state == "error":
                _try_transport_retry(
                    registry,
                    key,
                    hf_token = worker_token,
                    label = label,
                    log_prefix = log_prefix,
                    logger = logger,
                    repo_type = repo_type,
                    repo_id = repo_id,
                    watch_name = watch_name,
                    retry_transport = (
                        download_registry.TRANSPORT_XET
                        if retry_xet
                        else download_registry.TRANSPORT_HTTP
                    ),
                    xet_attempt = xet_attempt + 1 if retry_xet else xet_attempt,
                    pending_xet_failure = xet_failure,
                    # The ORIGINAL pre-Xet baseline: resampling would fold the killed worker's partial writes in, so a
                    # recovered attempt would read as a cached no-op.
                    bytes_before = _bytes_before,
                    allow_ambient_token = allow_ambient_token,
                )
        except Exception:
            if watchdog_stop is not None:
                watchdog_stop.set()
            # finalize_worker_exit is the only thing that clears running/cancelling, so if it raises, force a
            # terminal state or claim() is blocked until restart.
            logger.exception("download watcher crashed for %s", key)
            # finalize may have raised before reaping the worker, so terminate the still-registered Popen
            # first: the terminal set_job clears the repo guard and a live worker would race a retry.
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
            try:
                # The second look for anything prepare_cache_for_transport spared as too recently written. This
                # job's own blobs skip the abandonment wait, but ONLY once the job is genuinely finished: a
                # cancelled worker's partial was written seconds ago, and a retry relaunched above leaves the job
                # active, so it is not ours to assume.
                terminal = registry.get_job(key).state not in ("running", "cancelling")
                _owned, _owns_all = (
                    _sweep_ownership(
                        _metadata, _own_blob_hashes, _owned_for_sweep, repo_type, repo_id
                    )
                    if terminal
                    else (None, False)
                )
                swept = download_registry.sweep_abandoned_partials(
                    repo_type,
                    repo_id,
                    protected_blob_hashes = registry.peer_blob_hashes(key),
                    # A companion a sibling is writing right now is still held back by peer_blob_hashes above, whatever
                    # this job believes it owns.
                    owned_blob_hashes = _owned,
                    owns_all_blobs = _owns_all,
                    # The cache this worker actually wrote to: resolving the live one would miss the orphan whenever the
                    # download location changed mid-run, and sweep a cache this job never touched.
                    root = _cache_dir,
                )
                if swept:
                    logger.info(
                        "%sswept %d unresumable partial blob(s) for %s", log_prefix, swept, repo_id
                    )
            except Exception:
                logger.exception("abandoned-partial sweep failed for %s", key)
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
    allow_ambient_token: bool = True,
) -> str:
    # Only the Xet success-recording consumes this, and sampling lazy-loads unsloth_zoo, so torch and
    # transformers, on the request path.
    _baseline: Optional[int] = None
    if transport == download_registry.TRANSPORT_XET:
        # Before spawn(), deliberately: a small download can finalize its blobs while we are still
        # registering the process, and a later baseline would show no growth for a real transfer.
        _get_metadata = getattr(registry, "get_job_metadata", None)
        _metadata = _get_metadata(key) if callable(_get_metadata) else None
        _baseline = _job_bytes_on_disk(
            repo_type,
            repo_id,
            getattr(_metadata, "hub_cache", None) if _metadata is not None else None,
            (
                getattr(_metadata, "blob_hashes", frozenset())
                if getattr(_metadata, "variant", None)
                else None
            ),
        )
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
        bytes_before = _baseline,
        allow_ambient_token = allow_ambient_token,
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
    # No worker process yet: arm a pending cancel so register_process kills it on arrival during the
    # claim-to-register window.
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
    # No eager marker: finalize_worker_exit writes it on a "cancelled" exit, and persisting before the
    # kill races a clean completion and strands a stale marker.
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
        # Scoped jobs share one slot per repo, so publish the file list an adopting client needs; absent
        # metadata reports null, which the client reads as unprovable.
        scoped_files = list(metadata.scoped_files) if metadata is not None else []
        downloads.append(
            ActiveDownload(
                repo_id = ref_repo_id,
                variant = variant,
                transport = metadata.transport if metadata is not None else None,
                cancel_transport = (
                    metadata.cancel_marker_transport if metadata is not None else None
                ),
                state = ref.state,
                generation = ref.generation,
                files = scoped_files or None,
            )
        )
    return downloads
