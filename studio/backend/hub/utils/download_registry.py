# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HF cache inspection, download registry state, and orphan-worker reaping.

Worker spawning and exit handling live in
:mod:`hub.services.download_lifecycle`; this module owns the registry state
machine plus the cache/marker inspection those workers depend on.

Resume model
------------
Up to huggingface_hub 1.17 the HTTP transport supported true partial-file
resume: the resumer opened ``<etag>.incomplete`` in append mode and sent
``Range: bytes={resume_size}-`` to continue from disk. 1.18 replaced that with
a process-unique ``<etag>.<nonce>.incomplete`` opened ``"wb"`` and unlinked on
the way out, so no transport resumes within a file any more and a surviving
partial is litter (see :func:`hf_partials_are_resumable`). Resume is now
whole-file only: ``snapshot_download`` skips shards already materialized.

The XET transport CANNOT resume from a ``.incomplete`` partial:
``hf_xet.download_files`` rewrites the destination from scratch.
Network-level dedup still happens, but through the separate chunk cache at
``~/.cache/huggingface/xet/chunk-cache``, which these helpers never touch.

Cross-transport corruption: a partial written by XET (or ``hf_transfer``'s
parallel-Range writer) can be sparse — high reported size, zero-filled
gaps below. Feeding it to the HTTP resumer would produce a correct-sized
blob whose internal bytes are silently wrong. To prevent that, we keep
transport markers at the download's scope (repo for snapshots/datasets,
variant for GGUF) and refuse to inherit an HTTP partial unless the marker
proves the previous writer was the same single-stream sequential writer.

Marker writes go through tmp+rename in :func:`prepare_cache_for_transport`
before the worker hands off to ``snapshot_download``, so the next process
always sees a consistent provenance signal.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterator, Literal, NamedTuple, Optional, Sequence

from loggers import get_logger

# One floor, one name, shared: hand-written copies meant the site that got missed was missed because
# "who enforces it" had to be read rather than grepped.
from utils.process_lifetime import is_signalable_pid

from hub.utils import state_dir
from hub.utils.state_dir import RepoType

logger = get_logger(__name__)

from hub.utils.hf_cache_state import (
    TRANSPORT_AUTO,
    TRANSPORT_HTTP,
    TRANSPORT_XET,
    TRANSPORT_MARKER_NAME,
    VALID_TRANSPORTS,
    VALID_TRANSPORT_MODES,
    has_active_incomplete_blobs,
    hf_partials_are_resumable,
    iter_repo_cache_dirs,
    iter_active_repo_cache_dirs,
    repo_cache_dir_name,
    target_dir_name,
    hf_cache_root,
    ABANDONED_PARTIAL_SECONDS,
    blob_bytes_present,
    blob_download_lock_held,
    hf_cache_roots,
    incomplete_blob_hash,
    iter_destructive_repo_cache_dirs,
    partial_is_resumable,
)


@dataclass(frozen = True)
class DownloadTransportCapability:
    available: bool
    reason: Optional[str] = None


@dataclass(frozen = True)
class DownloadTransportCapabilities:
    http: DownloadTransportCapability
    xet: DownloadTransportCapability
    # What "auto" would pick right now, and why, so the picker can say "Auto (HTTP -- Xet stalled twice
    # on this machine)" instead of just "Auto".
    auto_resolves_to: str = TRANSPORT_XET
    auto_reason: Optional[str] = None
    # False on huggingface_hub >= 1.18, so the UI stops offering a byte-resume no writer can honour.
    partials_resumable: bool = True


def get_download_transport_capabilities(
    *, probe: bool = False, ram_gate: bool = False
) -> DownloadTransportCapabilities:
    """What this machine can do, and what Auto resolves to on it.

    ``probe`` runs the live Xet health check and is only for the frontend resolving Auto at
    download start. ``ram_gate`` applies the free-RAM half of that same verdict WITHOUT the
    network probe, for a surface that has to state what the next download will pick: the
    settings row said "Auto is using Xet" while the download path, which probes, chose HTTP.
    """
    xet_available = importlib.util.find_spec("hf_xet") is not None
    auto_transport = TRANSPORT_XET if xet_available else TRANSPORT_HTTP
    auto_reason: Optional[str] = None
    auto_forced = False
    if xet_available:
        try:
            from utils.hf_xet_fallback import cached_xet_health, xet_health

            # Ordinary UI polls are read-only and must not load Zoo; probe=True is the actual first-download
            # decision, and ram_gate loads it too without probing: an empty cache reads as the optimistic Xet,
            # so the row promised Xet while the next download chose HTTP.
            health_fn = xet_health if (probe or ram_gate) else cached_xet_health
            health = health_fn(probe = probe)
            if health is not None:
                auto_transport = TRANSPORT_XET if health.use_xet else TRANSPORT_HTTP
                auto_reason = str(health.reason)
                # UNSLOTH_FORCE_XET=1 is an operator override, not a measurement, so the free-RAM gate below stands
                # down exactly as resolve_auto_use_xet does.
                try:
                    from utils.hf_xet_fallback import xet_health_is_forced
                    auto_forced = bool(xet_health_is_forced(health))
                except Exception:
                    auto_forced = False
        except Exception:
            # No opinion: keep the optimistic default; the download-time ladder still recovers.
            pass
    if (
        xet_available
        and (probe or ram_gate)
        and auto_transport == TRANSPORT_XET
        and not auto_forced
    ):
        # Free RAM belongs in the same verdict, since the UI submits the answer as an explicit xet/http.
        # Read outside the health try, because a missing health module says nothing about RAM, and never
        # on an ordinary poll: only for a probe or an explicit ram_gate.
        try:
            from utils.hf_xet_fallback import free_ram_pressure_reason
            pressure = free_ram_pressure_reason()
        except Exception:
            pressure = None
        if pressure is not None:
            auto_transport = TRANSPORT_HTTP
            auto_reason = pressure
    return DownloadTransportCapabilities(
        http = DownloadTransportCapability(available = True),
        xet = DownloadTransportCapability(
            available = xet_available,
            reason = None
            if xet_available
            else "Xet transport is unavailable because hf_xet is not installed.",
        ),
        auto_resolves_to = auto_transport,
        auto_reason = auto_reason,
        partials_resumable = hf_partials_are_resumable(),
    )


def download_transport_unavailable_reason(transport: str) -> Optional[str]:
    if transport == TRANSPORT_HTTP:
        return None
    if transport == TRANSPORT_XET:
        caps = get_download_transport_capabilities().xet
        return None if caps.available else caps.reason
    return f"Unsupported download transport: {transport}"


def _worker_breadcrumb_path(key: str) -> Optional[Path]:
    parent = state_dir.workers_dir()
    if parent is None:
        return None
    safe = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return parent / f"{safe}.json"


def write_worker_breadcrumb(key: str, pid: int, metadata: Optional["DownloadMetadata"]) -> None:
    """Record a live worker's PID so a restarted backend can reap it. Best
    effort: a write failure only forfeits boot-time reaping for this worker,
    still covered by the worker's own parent-death watchdog."""
    path = _worker_breadcrumb_path(key)
    if path is None:
        return
    payload = {
        "pid": int(pid),
        "repo_type": metadata.repo_type if metadata is not None else None,
        "repo_id": metadata.repo_id if metadata is not None else None,
        "variant": metadata.variant if metadata is not None else None,
        "transport": metadata.transport if metadata is not None else None,
        "cancel_marker_transport": metadata.cancel_marker_transport
        if metadata is not None
        else None,
        "hub_cache": metadata.hub_cache if metadata is not None else None,
        "xet_cache": metadata.xet_cache if metadata is not None else None,
    }
    tmp = path.with_name(f".{path.name}.tmp-{pid}")
    try:
        tmp.write_text(json.dumps(payload), encoding = "utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug("Could not write worker breadcrumb %s: %s", path, exc)
        try:
            tmp.unlink(missing_ok = True)
        except OSError:
            pass


def remove_worker_breadcrumb(key: str) -> None:
    path = _worker_breadcrumb_path(key)
    if path is None:
        return
    _safe_unlink(path)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok = True)
    except OSError as exc:
        logger.debug("Could not remove %s: %s", path, exc)


def _process_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        SYNCHRONIZE = 0x00100000
        ERROR_INVALID_PARAMETER = 87
        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        ctypes.set_last_error(0)
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if not handle:
            return ctypes.get_last_error() != ERROR_INVALID_PARAMETER
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def _read_process_cmdline(pid: int) -> Optional[str]:
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        if proc_cmdline.exists():
            raw = proc_cmdline.read_bytes()
            return raw.replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        pass
    try:
        import psutil
        return " ".join(psutil.Process(pid).cmdline())
    except Exception:
        return None


def _cmdline_repo_id(cmdline: str) -> Optional[str]:
    try:
        args = shlex.split(cmdline)
    except ValueError:
        args = cmdline.split()
    for i, arg in enumerate(args):
        if arg == "--repo-id" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--repo-id="):
            return arg.split("=", 1)[1]
    return None


def _is_our_worker(pid: int, repo_id: Optional[str]) -> bool:
    cmdline = _read_process_cmdline(pid)
    if cmdline is None:
        return False
    if "hub.workers.hf_download" not in cmdline:
        return False
    # Exact --repo-id match: a substring match would let a stale breadcrumb for Org/Model reap a live
    # worker for Org/Model-v2.
    if isinstance(repo_id, str) and repo_id:
        return _cmdline_repo_id(cmdline) == repo_id
    return True


def _kill_orphan(pid: int) -> bool:
    """Signal the process and wait for it to actually be gone. True once it is.

    The wait is what makes the boot sweep meaningful: the signal only schedules the death, and
    a sweep that runs a microsecond later still sees the worker's Hugging Face blob lock and
    spares a partial nothing will ever finish. Bounded, because a pid we cannot reap is not a
    reason to hold up startup -- and answering False there matters: a survivor must keep its
    breadcrumb and must not have its live partial claimed as ours to delete.
    """
    # Repeated here because this one sends the signal, and a helper that kills should not depend on
    # every future caller having checked first.
    if not is_signalable_pid(pid):
        return False
    try:
        os.kill(pid, signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL)
    except OSError:
        return not _process_alive(pid)
    deadline = time.monotonic() + _ORPHAN_REAP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if not _process_alive(pid):
            return True
        time.sleep(0.05)
    logger.warning("Orphan worker pid=%s is still alive after the reap timeout.", pid)
    return False


# Long enough for a SIGKILLed worker to be torn down, short enough not to delay a boot.
_ORPHAN_REAP_TIMEOUT_SECONDS = 5.0


def _settle_orphaned_download(
    repo_type: Optional[str],
    repo_id: Optional[str],
    variant: Optional[str],
    transport: Optional[str],
    hub_cache: Optional[str] = None,
) -> None:
    """Persist a cancel marker for a reaped orphan still mid-download so the next
    launch settles it to a resumable "cancelled" state instead of a phantom-running
    row.

    Gated on surviving partial state and on the recorded manifest not already
    verifying against an active snapshot, so a download that finished before its
    breadcrumb was cleaned up is never mislabeled cancelled. For a GGUF variant
    manifest with blob hashes, the partial-state check is scoped to those hashes so
    a sibling variant cannot contaminate this orphan's state. The recorded
    transport is preserved so the resume affordance stays accurate."""
    if repo_type not in ("model", "dataset") or not repo_id:
        return
    from hub.utils import download_manifest

    cache_root = Path(hub_cache) if isinstance(hub_cache, str) and hub_cache else None

    manifest = download_manifest.read_manifest(
        repo_type,
        repo_id,
        variant,
        hub_cache = cache_root,
    )
    if repo_type == "model" and variant and manifest is None:
        return
    if manifest is None:
        if not has_active_incomplete_blobs(repo_type, repo_id, root = cache_root):
            return
    else:
        if _manifest_verifies_against_active_cache(
            repo_type,
            repo_id,
            manifest,
            root = cache_root,
        ):
            return
        if not _manifest_has_active_incomplete_blobs(
            repo_type,
            repo_id,
            manifest,
            root = cache_root,
        ):
            return
    persist_cancel_marker(
        repo_type,
        repo_id,
        variant,
        transport,
        hub_cache = hub_cache,
        logger = logger,
    )


def reap_orphan_workers() -> None:
    """Kill download workers left running by a previous backend instance.

    Verifies each breadcrumb's PID is alive AND its command line is one of our
    workers before terminating, so a recycled PID can't take down an unrelated
    process. A resumable partial is never touched, so a reaped download stays
    resumable; an interrupted one with bytes on disk is settled to a cancelled
    marker (see :func:`_settle_orphaned_download`) so its resume affordance
    survives a hard crash like a graceful shutdown's does. A partial nothing can
    resume has no affordance to preserve and is swept
    (see :func:`sweep_abandoned_partials`). Runs once at startup and never raises."""
    reaped: list[tuple[str, str, Optional[str]]] = []
    parent = state_dir.workers_dir()
    if parent is None:
        _boot_sweep(reaped)
        return
    try:
        entries = list(parent.iterdir())
    except OSError:
        # Unreadable breadcrumbs means no worker can be claimed as reaped, not that the caches go unswept:
        # they are a separate tree.
        _boot_sweep(reaped)
        return
    for entry in entries:
        if not entry.is_file() or not entry.name.endswith(".json"):
            continue
        try:
            data = json.loads(entry.read_text(encoding = "utf-8"))
        except (OSError, ValueError):
            _safe_unlink(entry)
            continue
        pid = data.get("pid") if isinstance(data, dict) else None
        repo_id = data.get("repo_id") if isinstance(data, dict) else None
        # pid 1 as well as 0 and negatives: the cmdline check below is what keeps this honest, but a
        # record naming pid 1 once slipped through the reaper's start-time check.
        signalable = is_signalable_pid(pid)
        try:
            if not signalable:
                # Its repo fields are still readable, and the settle below preserves the partial's resume marker:
                # unlinking from here would cost the user a restarted download to pay for a bug that is ours.
                pass
            elif not _process_alive(pid):
                # Already gone, which is better proof than killing it ourselves; its partial is ours to sweep even
                # though this invocation reaped nothing.
                reaped.append((data.get("repo_type") or "model", repo_id, data.get("hub_cache")))
            elif _is_our_worker(pid, repo_id):
                if not _kill_orphan(pid):
                    # Still running: keeping the breadcrumb keeps it tracked for the next boot, and claiming no
                    # ownership keeps its live partial out of the sweep.
                    logger.warning(
                        "Could not reap download worker pid=%s repo=%s; leaving its "
                        "breadcrumb and partial in place.",
                        pid,
                        repo_id,
                    )
                    continue
                # The sweep has to come after the kill, not before, or it reads the still-held blob lock and spares
                # a file nothing will ever finish.
                reaped.append((data.get("repo_type") or "model", repo_id, data.get("hub_cache")))
                logger.warning(
                    "Reaped orphaned download worker pid=%s repo=%s from a "
                    "previous backend instance.",
                    pid,
                    repo_id,
                )
            _settle_orphaned_download(
                data.get("repo_type"),
                repo_id,
                data.get("variant"),
                data.get("cancel_marker_transport") or data.get("transport"),
                data.get("hub_cache"),
            )
        except Exception as exc:
            logger.debug("Reaper failed for breadcrumb %s: %s", entry, exc)
        _safe_unlink(entry)
    _boot_sweep(reaped)


def _boot_sweep(reaped: "Sequence[tuple[str, str, Optional[str]]]") -> None:
    """Startup cleanup, run only once every surviving worker above has been killed.

    The reaped repos are swept inline: that work is bounded by the breadcrumbs and it settles
    the caches a returning user is most likely to look at. The all-caches pass is not bounded
    by anything -- it walks every repo dir, stats every partial and probes locks -- so it goes
    to a thread rather than holding the lifespan open ahead of the first request.
    """
    swept = 0
    try:
        for repo_type, repo_id, hub_cache in reaped:
            # We killed this one ourselves a moment ago, so it need not look abandoned yet.
            swept += sweep_abandoned_partials(
                repo_type,
                repo_id,
                owns_all_blobs = True,
                root = hub_cache,
            )
    except Exception as exc:
        logger.debug("Boot sweep of reaped downloads failed: %s", exc)
    if swept:
        logger.info(
            "Swept %d unresumable partial blob(s) left by a previous backend instance.", swept
        )

    def _sweep_all_caches() -> None:
        try:
            removed = sweep_abandoned_partials_in_all_caches()
        except Exception as exc:
            logger.debug("Background sweep of abandoned partials failed: %s", exc)
            return
        if removed:
            logger.info("Swept %d unresumable partial blob(s) from the HF caches.", removed)

    threading.Thread(
        target = _sweep_all_caches,
        name = "hf-abandoned-partial-sweep",
        daemon = True,
    ).start()


class _PurgeOutcome(NamedTuple):
    """Number of selected partials removed and number that could not be removed."""

    removed: int
    failed: int


# Only the unresumable sweep waits out ABANDONED_PARTIAL_SECONDS: it reclaims disk, while a marker-
# mismatch purge exists to stop a corrupt append and cannot defer.


def _purge_incomplete_blobs(
    entry: Path,
    only_hashes: Optional[frozenset[str]] = None,
    protected_hashes: Optional[frozenset[str]] = None,
    *,
    unresumable_only: bool = False,
    owned_hashes: Optional[frozenset[str]] = None,
    owns_all_blobs: bool = False,
) -> _PurgeOutcome:
    """Delete selected partials while preserving protected concurrent writes.

    Report failed deletions so sparse partials cannot receive an HTTP marker.

    ``unresumable_only`` restricts the sweep to partials no writer can reuse AND that nothing
    has touched for ``ABANDONED_PARTIAL_SECONDS``. Unlinking a live partial does not stop its
    writer on POSIX; it keeps filling an unlinked inode and then fails at the rename, so the
    cost of that mistake is another client's whole download.

    ``owned_hashes``, or ``owns_all_blobs`` for a job that owns its whole repo dir (one with no
    variant, which claim() will not let a sibling share), are blobs whose only Unsloth-side
    writer has just been reaped. Those do not wait out the full grace -- the corpse would
    outlive the retry that follows a cancel, which is the frozen bar this whole change is
    about -- but they are not simply trusted either: registry ownership proves OUR writer is
    gone, never that no independent process shares the cache. They go through a stillness
    probe instead, which is the one liveness test that survives a filesystem where flock is
    granted to every caller and the lock therefore reads free while somebody writes.
    """
    now = time.time()
    blobs_dir = entry / "blobs"
    if not blobs_dir.is_dir():
        return _PurgeOutcome(0, 0)
    removed = 0
    failed = 0
    watched: list[tuple[Path, str, int, float]] = []
    try:
        candidates = list(blobs_dir.iterdir())
    except OSError:
        # Nothing in an unreadable directory can be certified as safe.
        return _PurgeOutcome(0, 1)
    for blob in candidates:
        try:
            if not blob.is_file():
                continue
            blob_hash = incomplete_blob_hash(blob.name)
            if blob_hash is None:
                continue
            if protected_hashes and blob_hash in protected_hashes:
                continue
            if only_hashes is not None and blob_hash not in only_hashes:
                continue
            if unresumable_only:
                if partial_is_resumable(blob.name, entry.parent):
                    continue
                # Neither signal is sufficient alone: the lock is precise but upstream calls it best-effort and
                # some filesystems grant it to everyone, while mtime cannot tell a dead writer from a stalled one.
                if blob_download_lock_held(entry, blob_hash):
                    continue
                owned = owns_all_blobs or bool(owned_hashes and blob_hash in owned_hashes)
                if not owned:
                    if now - blob.stat().st_mtime < ABANDONED_PARTIAL_SECONDS:
                        continue
                elif now - blob.stat().st_mtime < ABANDONED_PARTIAL_SECONDS:
                    stat = blob.stat()
                    watched.append((blob, blob_hash, stat.st_size, stat.st_mtime))
                    continue
            blob.unlink()
            removed += 1
        except FileNotFoundError:
            # A peer finalized or removed it after enumeration, so the requested end state was reached and this
            # is not a failed purge.
            continue
        except OSError:
            failed += 1
            continue
    watched_outcome = _purge_still_partials(watched)
    return _PurgeOutcome(removed + watched_outcome.removed, failed + watched_outcome.failed)


# One shared pause is enough to tell a frozen corpse from a writer mid-transfer: hf writes a partial continuously.
_STILLNESS_PROBE_SECONDS = 2.0


def _purge_still_partials(watched: "Sequence[tuple[Path, str, int, float]]") -> _PurgeOutcome:
    """Delete the owned partials that do not move while we watch them.

    Sampled once before, once after a single shared sleep, so the cost is one pause per sweep
    rather than one per file. Anything that grew or was touched in between has a live writer,
    whatever the advisory lock had to say about it.
    """
    if not watched:
        return _PurgeOutcome(0, 0)
    time.sleep(_STILLNESS_PROBE_SECONDS)
    removed = 0
    failed = 0
    for blob, _blob_hash, size, mtime in watched:
        try:
            stat = blob.stat()
            if stat.st_size != size or stat.st_mtime != mtime:
                continue
            blob.unlink()
            removed += 1
        except FileNotFoundError:
            continue
        except OSError:
            failed += 1
    return _PurgeOutcome(removed, failed)


def _iter_active_snapshot_dirs(
    repo_type: str,
    repo_id: str,
    *,
    root: Optional[Path] = None,
) -> Iterator[Path]:
    for entry in iter_active_repo_cache_dirs(repo_type, repo_id, root = root):
        snapshots_dir = entry / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        try:
            snapshots = list(snapshots_dir.iterdir())
        except OSError:
            continue
        for snapshot in snapshots:
            if snapshot.is_dir():
                yield snapshot


def _manifest_verifies_against_active_cache(
    repo_type: str,
    repo_id: str,
    manifest,
    *,
    root: Optional[Path] = None,
) -> bool:
    from hub.utils import download_manifest
    for snapshot_dir in _iter_active_snapshot_dirs(repo_type, repo_id, root = root):
        if download_manifest.verify_against_disk(manifest, snapshot_dir).ok:
            return True
    return False


def _manifest_has_active_incomplete_blobs(
    repo_type: str,
    repo_id: str,
    manifest,
    *,
    root: Optional[Path] = None,
) -> bool:
    if not getattr(manifest, "variant", None):
        return has_active_incomplete_blobs(repo_type, repo_id, root = root)
    expected_hashes = frozenset(
        expected.sha256 for expected in manifest.expected_files if expected.sha256
    )
    if not expected_hashes:
        return has_active_incomplete_blobs(repo_type, repo_id, root = root)
    return bool(
        incomplete_blob_hashes(
            repo_type,
            repo_id,
            active_only = True,
            root = root,
        ).intersection(expected_hashes)
    )


def _marker_path(entry: Path, variant: Optional[str] = None) -> Path:
    if not variant:
        return entry / TRANSPORT_MARKER_NAME
    digest = hashlib.sha256(variant.strip().lower().encode("utf-8")).hexdigest()[:24]
    return entry / f"{TRANSPORT_MARKER_NAME}.gguf-{digest}"


def _is_transport_marker_file(path: Path) -> bool:
    # Matches ".transport", its tmps and variant-scoped ".transport.gguf-*"; real HF cache entries
    # (blobs/refs/snapshots/.no_exist) never start with ".transport.".
    return path.name == TRANSPORT_MARKER_NAME or path.name.startswith(f"{TRANSPORT_MARKER_NAME}.")


def _companion_marker_path(entry: Path) -> Path:
    return entry / f"{TRANSPORT_MARKER_NAME}.companion"


def _read_marker_value(marker: Path) -> Optional[str]:
    try:
        if not marker.exists():
            return None
        value = marker.read_text(encoding = "utf-8").strip()
    except (OSError, UnicodeDecodeError):
        # UnicodeDecodeError is a ValueError, so it would escape and abort prepare_cache_for_transport; an
        # unknown value just purges and restarts.
        return None
    return value if value in VALID_TRANSPORTS else None


def _write_marker_value(marker: Path, mode: str) -> None:
    try:
        # tmp + rename so a SIGKILL mid-write cannot leave a half-written marker, with a per-process tmp
        # name so concurrent writers do not clobber tmps.
        tmp = marker.with_name(f"{marker.name}.tmp-{os.getpid()}")
        tmp.write_text(mode, encoding = "utf-8")
        os.replace(tmp, marker)
    except OSError:
        # Best-effort: a missing marker next run purges the partial defensively, which is the safe failure mode.
        pass


def _read_marker(entry: Path, variant: Optional[str] = None) -> Optional[str]:
    return _read_marker_value(_marker_path(entry, variant))


def _write_marker(
    entry: Path,
    mode: str,
    variant: Optional[str] = None,
) -> None:
    _write_marker_value(_marker_path(entry, variant), mode)


def _read_companion_marker(entry: Path) -> Optional[str]:
    return _read_marker_value(_companion_marker_path(entry))


def _write_companion_marker(entry: Path, mode: str) -> None:
    _write_marker_value(_companion_marker_path(entry), mode)


def prepare_cache_for_transport(
    repo_type: str,
    repo_id: str,
    mode: str,
    variant: Optional[str] = None,
    only_blob_hashes: Optional[frozenset[str]] = None,
    companion_blob_hashes: Optional[frozenset[str]] = None,
    protected_blob_hashes: Optional[frozenset[str]] = None,
    root: Optional[Path] = None,
) -> int:
    """Guarantee any pre-existing ``.incomplete`` blobs are SAFE to resume under
    *mode*. Returns the number of partial blobs purged for untrusted provenance.

    Two marker scopes govern GGUF downloads. ``only_blob_hashes`` are the
    variant's own (main quant) blobs, judged by the ``variant``-scoped marker;
    ``None`` widens the scope to every partial for full-repo snapshots/datasets.
    ``companion_blob_hashes`` are blobs shared across sibling variants (a vision
    mmproj), judged by a separate repo-scoped companion marker — so a companion
    partial is trusted against the transport that wrote it, not against
    whichever sibling variant resumes next.

    The contract:
    - HTTP mode: a partial is trusted ONLY when its governing marker equals
      ``"http"``. Any other case (missing/unreadable/mismatched marker) purges,
      since the HTTP resumer would otherwise append to a sparse
      XET/parallel-Range partial and silently produce a corrupt blob. On
      huggingface_hub >= 1.18 there is no resumer left to trust a partial for
      (see ``hf_partials_are_resumable``), so the marker is bypassed and every
      selected partial purges.
    - XET mode: incomplete blobs are purged (``hf_xet.download_files`` rewrites
      from scratch, so this only fixes UI accounting — bytes already in CAS are
      reused via the chunk-cache). Scoped to ``only_blob_hashes``: companion
      blobs fall outside that set and survive (shared, and XET overwrites them).

    ``protected_blob_hashes`` are blobs a concurrent same-repo peer is writing;
    they are excluded from every purge so a shared companion is never deleted
    mid-write.

    Scope: ``root`` selects the cache captured by the caller. It defaults to the
    active ``HF_HUB_CACHE`` root for workers that inherit their cache through
    the environment. Markers are written for the new mode before returning,
    except when an HTTP purge cannot remove every selected partial. Withholding
    the marker keeps the surviving partial untrusted.
    """
    if mode not in VALID_TRANSPORTS:
        if mode == TRANSPORT_AUTO:
            # "auto" is a request preference, not a cache writer: naming it turns "invalid transport" into the
            # actual bug.
            raise ValueError(
                f"{TRANSPORT_AUTO!r} must be resolved to a concrete transport before preparing the "
                f"cache; expected one of {sorted(VALID_TRANSPORTS)}"
            )
        raise ValueError(
            f"Invalid transport mode: {mode!r} (transports: {sorted(VALID_TRANSPORTS)}, "
            f"request modes: {sorted(VALID_TRANSPORT_MODES)})"
        )
    root = hf_cache_root(create = True) if root is None else hf_cache_root(create = True, root = root)
    if root is None:
        return 0
    target = target_dir_name(repo_type, repo_id)
    try:
        entries = [e for e in root.iterdir() if e.name.lower() == target]
    except OSError:
        return 0
    if not entries:
        # Pre-create the repo dir so the marker lands before the worker writes any bytes; otherwise a
        # SIGKILL mid-download leaves a partial with no marker that the resume then purges.
        canonical = repo_cache_dir_name(repo_type, repo_id)
        new_entry = root / canonical
        try:
            new_entry.mkdir(exist_ok = True)
        except OSError:
            return 0
        entries = [new_entry]
    protected = protected_blob_hashes or frozenset()
    has_companion = bool(companion_blob_hashes)
    total_purged = 0
    for entry in entries:
        main_purge = _PurgeOutcome(0, 0)
        companion_purge = _PurgeOutcome(0, 0)
        if mode == TRANSPORT_XET:
            main_purge = _purge_incomplete_blobs(entry, only_blob_hashes, protected)
        else:
            # A matching marker only matters while something can still append to the partial it vouches for.
            if _read_marker(entry, variant) != mode:
                main_purge = _purge_incomplete_blobs(entry, only_blob_hashes, protected)
            else:
                main_purge = _purge_incomplete_blobs(
                    entry,
                    only_blob_hashes,
                    protected,
                    unresumable_only = True,
                )
            if companion_blob_hashes:
                if _read_companion_marker(entry) != mode:
                    companion_purge = _purge_incomplete_blobs(
                        entry,
                        companion_blob_hashes,
                        protected,
                    )
                else:
                    companion_purge = _purge_incomplete_blobs(
                        entry,
                        companion_blob_hashes,
                        protected,
                        unresumable_only = True,
                    )
        total_purged += main_purge.removed + companion_purge.removed
        record_unconditionally = mode == TRANSPORT_XET
        if record_unconditionally or not main_purge.failed:
            _write_marker(entry, mode, variant)
        if has_companion and (record_unconditionally or not companion_purge.failed):
            _write_companion_marker(entry, mode)
    return total_purged


_HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")
_BEARER_RE = re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+")


def scrub_secrets(text: str, *, hf_token: Optional[str] = None) -> str:
    if not text:
        return text
    cleaned = text
    if hf_token:
        cleaned = cleaned.replace(hf_token, "***")
    cleaned = _BEARER_RE.sub("Bearer ***", cleaned)
    cleaned = _HF_TOKEN_RE.sub("hf_***", cleaned)
    return cleaned


def purge_empty_marker_dir(
    repo_type: str,
    repo_id: str,
    variant: Optional[str] = None,
) -> bool:
    """Remove the failed download's own transport marker from a marker-only dir.

    ``prepare_cache_for_transport`` pre-creates the dir + marker before any
    download; a failure during validation/auth/network setup leaves the dir as
    marker-only litter. Only the failed download's OWN marker is removed (the
    repo-scope ``.transport`` or the variant-scoped ``.transport.gguf-*`` plus
    its ``.tmp-*`` siblings); a sibling variant's marker and the shared
    ``.transport.companion`` are left intact, so cancelling one quant never
    strips a peer's provenance. A dir holding ``blobs/``/``snapshots/``/``refs/``
    won't match and is left untouched, so a resumable partial isn't blown away.
    """
    cleaned = False
    for entry in iter_repo_cache_dirs(repo_type, repo_id):
        try:
            contents = list(entry.iterdir())
        except OSError:
            continue
        if not contents or not all(_is_transport_marker_file(item) for item in contents):
            continue
        own_name = _marker_path(entry, variant).name
        own_markers = [
            item
            for item in contents
            if item.name == own_name or item.name.startswith(f"{own_name}.tmp")
        ]
        if not own_markers:
            continue
        try:
            for marker in own_markers:
                marker.unlink()
        except OSError:
            continue
        cleaned = True
        try:
            entry.rmdir()
        except OSError:
            continue
    return cleaned


def read_active_transport_marker(
    repo_type: str,
    repo_id: str,
    variant: Optional[str] = None,
    *,
    root: Optional[Path] = None,
) -> Optional[str]:
    for entry in iter_active_repo_cache_dirs(repo_type, repo_id, root = root):
        value = _read_marker(entry, variant)
        if value is not None:
            return value
    return None


def sweep_abandoned_partials(
    repo_type: str,
    repo_id: str,
    *,
    only_blob_hashes: Optional[frozenset[str]] = None,
    protected_blob_hashes: Optional[frozenset[str]] = None,
    owned_blob_hashes: Optional[frozenset[str]] = None,
    owns_all_blobs: bool = False,
    root: Optional[str | Path] = None,
) -> int:
    """Remove partials nothing can resume and nothing has touched. Returns how many went.

    ``prepare_cache_for_transport`` runs once, before a download, and skips anything still
    inside the abandonment grace. That skip lands on the common case: the orphan is the file a
    hard kill left behind, and the user restarts within seconds of the kill that made it. Run
    this when a download reaches a terminal state and every file skipped then gets a second
    look, by which point the grace has long since elapsed.
    """
    # DownloadMetadata.hub_cache is a str and every caller hands its captured root straight through, so
    # normalize here rather than trusting each one.
    if isinstance(root, str):
        root = Path(root) if root else None
    removed = 0
    # The destructive iterator, not the active one: on a case-insensitive collision the active iterator
    # yields every spelling while this one resolves to the exact directory or refuses.
    for entry in iter_destructive_repo_cache_dirs(repo_type, repo_id, root = root):
        outcome = _purge_incomplete_blobs(
            entry,
            only_blob_hashes,
            protected_blob_hashes,
            unresumable_only = True,
            owned_hashes = owned_blob_hashes,
            owns_all_blobs = owns_all_blobs,
        )
        removed += outcome.removed
    return removed


def sweep_abandoned_partials_in_all_caches() -> int:
    """Boot-time sweep across every known HF cache root. Returns how many partials went.

    Not driven off worker breadcrumbs, because ``drop_process`` removes a breadcrumb during
    ``finalize_worker_exit`` -- before the terminal-state sweep runs -- so a partial that sweep
    skips for being freshly written has no breadcrumb left to be found by. Walking the caches
    instead needs no record to survive, and every deletion still has to clear the same
    unresumable, unlocked and abandoned gates.
    """
    removed = 0
    for root in hf_cache_roots():
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if "--" not in entry.name or not (entry / "blobs").is_dir():
                continue
            removed += _purge_incomplete_blobs(entry, None, None, unresumable_only = True).removed
    return removed


def is_resumable_partial(
    repo_type: str,
    repo_id: str,
    variant: Optional[str] = None,
    *,
    root: Optional[Path] = None,
) -> bool:
    """True only when a partial exists AND something can still resume from it.

    Two ways to fail that. XET partials exist on disk but ``hf_xet`` rewrites the destination
    from scratch, so the marker has to say HTTP. And an HTTP partial is only resumable while a
    writer that reopens it is installed; the UI turns this flag into "Resume with HTTP to keep
    the progress you already have", which must not be promised for bytes about to be swept.

    Decided per cache entry, the way :func:`prepare_cache_for_transport` decides what to purge,
    because one repo can own several active directories at once (a case-sensitive filesystem
    holds ``models--Org--Model`` beside ``models--org--model``). A marker only vouches for
    partials sitting beside it.

    Within an entry the split matters too: main blobs answer to the variant marker, while a
    shared companion (mmproj, MTP drafter) answers to ``.transport.companion``. With a
    ``variant`` the manifest says which hashes are which; a blob in neither set, and a variant
    with no manifest, back nothing rather than an unscoped yes.

    ``root`` is the hub cache the row being judged was found in. A row can come from a
    remembered, legacy or custom cache, and that root holds both its own partials and its own
    manifest scope (state is keyed by a per-cache digest), so leaving it out asked the ACTIVE
    root about a directory it does not contain. ``None`` keeps the active root, for callers
    that have no particular row in hand.
    """
    main, companion = (
        _manifest_hash_split(repo_type, repo_id, variant, root = root) if variant else (set(), set())
    )
    for entry in iter_active_repo_cache_dirs(repo_type, repo_id, root = root):
        resumable = _resumable_blob_hashes(entry)
        if not resumable:
            continue
        if _read_marker(entry, variant) == TRANSPORT_HTTP and (variant is None or resumable & main):
            return True
        if (
            variant is not None
            and resumable & companion
            and _read_companion_marker(entry) == TRANSPORT_HTTP
        ):
            return True
    return False


def _resumable_blob_hashes(entry: Path) -> set[str]:
    """Blob hashes whose partial sits in THIS entry and can still be appended to."""
    out: set[str] = set()
    try:
        for blob in (entry / "blobs").iterdir():
            if not blob.is_file():
                continue
            blob_hash = incomplete_blob_hash(blob.name)
            if blob_hash is not None and partial_is_resumable(blob.name, entry.parent):
                out.add(blob_hash)
    except OSError:
        return out
    return out


def _manifest_hash_split(
    repo_type: str,
    repo_id: str,
    variant: Optional[str],
    *,
    root: Optional[Path] = None,
) -> tuple[set[str], set[str]]:
    """``(main, companion)`` blob hashes from the variant's manifest, split the way the worker
    splits them when it asks for a purge. Empty pair when either step cannot answer, which
    reads as "no resume to promise"."""
    from hub.utils import download_manifest

    manifest = download_manifest.read_manifest(repo_type, repo_id, variant, hub_cache = root)
    if manifest is None or not manifest.expected_files or not variant:
        return set(), set()
    try:
        from hub.utils.gguf_plan import plan_from_expected_files
        plan = plan_from_expected_files(variant, manifest.expected_files)
    except Exception as exc:  # noqa: BLE001 - an unsplittable manifest promises nothing
        logger.debug("Could not split manifest hashes for %s [%s]: %s", repo_id, variant, exc)
        return set(), set()
    return set(plan.main_hashes), set(plan.companion_hashes)


def incomplete_blob_hashes(
    repo_type: str,
    repo_id: str,
    *,
    active_only: bool = False,
    resumable_only: bool = False,
    root: Optional[Path] = None,
) -> set[str]:
    """Logical blob hashes with a partial on disk.

    ``resumable_only`` keeps just the ones a later attempt could actually append to, which is
    what a "resume and keep your progress" claim has to be built on.
    """
    out: set[str] = set()
    entries = (
        iter_active_repo_cache_dirs(repo_type, repo_id, root = root)
        if active_only
        else iter_repo_cache_dirs(repo_type, repo_id)
    )
    for entry in entries:
        blobs_dir = entry / "blobs"
        if not blobs_dir.is_dir():
            continue
        try:
            for blob in blobs_dir.iterdir():
                if not blob.is_file():
                    continue
                blob_hash = incomplete_blob_hash(blob.name)
                if blob_hash is None:
                    continue
                if resumable_only and not partial_is_resumable(blob.name, entry.parent):
                    continue
                out.add(blob_hash)
        except OSError:
            continue
    return out


def completed_blob_bytes(
    repo_type: str,
    repo_id: str,
    blob_hashes: frozenset[str],
    *,
    root: Optional[Path] = None,
) -> int:
    """Sum finalized blob bytes for *blob_hashes* in a single HF cache root.

    A worker only writes to its captured ``HF_HUB_CACHE`` root, so a baseline
    must be scoped to that root (``root``), not re-resolved to whatever cache is
    active now; otherwise a runtime cache switch makes the retry baseline count
    bytes from the wrong disk.
    """
    if not blob_hashes:
        return 0
    total = 0
    for entry in iter_active_repo_cache_dirs(repo_type, repo_id, root = root):
        blobs_dir = entry / "blobs"
        if not blobs_dir.is_dir():
            continue
        for blob_hash in blob_hashes:
            blob = blobs_dir / blob_hash
            try:
                if blob.is_file():
                    total += max(0, int(blob.stat().st_size))
            except OSError:
                continue
    return total


def existing_blob_bytes(
    repo_type: str,
    repo_id: str,
    blob_hashes: frozenset[str],
    *,
    root: Optional[Path] = None,
) -> int:
    """Bytes a download will NOT have to fetch again for *blob_hashes*, in *root* or, when it is
    None, the active HF cache root: finalized blobs, plus partials something can still resume
    from. A row pinned to another root must pass it, since a resume writes into the root the row
    names and blobs in the active one are not bytes it can reuse. A blob is in exactly one state,
    so summing both candidate names never double-counts. Used to size what a (possibly resumed)
    download still needs to write before the run starts."""
    if not blob_hashes:
        return 0
    # One tally for ALL the repo dirs the root holds: the Hub resolves repo ids case-insensitively
    # while huggingface_hub keeps the caller's casing, so a case-sensitive filesystem holds two copies
    # of one blob and summing the dirs counted that shard twice.
    present = {blob_hash: 0 for blob_hash in blob_hashes}
    for entry in iter_active_repo_cache_dirs(repo_type, repo_id, root = root):
        blobs_dir = entry / "blobs"
        if not blobs_dir.is_dir():
            continue
        try:
            entries = list(blobs_dir.iterdir())
        except OSError:
            continue
        for blob in entries:
            try:
                if not blob.is_file():
                    continue
                partial_hash = incomplete_blob_hash(blob.name)
                blob_hash = partial_hash if partial_hash is not None else blob.name
                if blob_hash not in present:
                    continue
                if (
                    partial_hash is not None
                    and not partial_is_resumable(blob.name, entry.parent)
                    and not blob_download_lock_held(entry, blob_hash)
                ):
                    # An unresumable partial is refetched in full into a new path, so counting it would clear a
                    # download for a disk that cannot hold it. A LOCKED one is different: a live peer is finishing it
                    # and snapshot_download blocks on that lock and reuses the result, so those bytes are not ours to
                    # find room for.
                    continue
                # Measured by the bytes actually ON DISK: hf_transfer's parallel Range writer leaves a sparse file
                # whose st_size runs ahead of what was written, observed at 1.2 GB reported against 112 MB. A
                # finalized blob is whole by construction, so it keeps st_size: st_blocks is smaller than the file
                # on a compressing filesystem.
                bytes_here = (
                    blob_bytes_present(blob)
                    if partial_hash is not None
                    else max(0, int(blob.stat().st_size))
                )
                # Broken advisory locks can leave several process-unique writers for one etag: duplicate attempts,
                # not additive completion, so keep the largest.
                present[blob_hash] = max(present[blob_hash], max(0, int(bytes_here)))
            except OSError:
                continue
    return sum(present.values())


JobState = Literal["idle", "running", "cancelling", "cancelled", "complete", "error"]

TERMINAL_STATES = frozenset({"complete", "cancelled", "error"})
_ACTIVE_STATES = frozenset({"running", "cancelling"})


@dataclass(frozen = True)
class DownloadState:
    state: JobState
    error: Optional[str] = None


@dataclass(frozen = True)
class DownloadMetadata:
    repo_type: RepoType
    repo_id: str
    variant: Optional[str]
    transport: Optional[str]
    cancel_marker_transport: Optional[str] = None
    # GGUF variant main/writable hashes, identifying the variant-specific shards for concurrency decisions.
    blob_hashes: frozenset[str] = field(default_factory = frozenset)
    # Includes the shared mmproj companion for vision GGUF repos.
    progress_blob_hashes: frozenset[str] = field(default_factory = frozenset)
    # Bytes already complete before this job started; not counted as this run's
    completed_baseline_bytes: int = 0
    hub_cache: Optional[str] = None
    xet_cache: Optional[str] = None
    # Scoped jobs only: the exact files to fetch, kept so the XET -> HTTP retry respawns the same scoped download.
    scoped_files: tuple[str, ...] = ()


@dataclass(frozen = True)
class ActiveDownloadRef:
    key: str
    state: str
    metadata: Optional[DownloadMetadata]
    generation: int


def normalize_repo_key(repo_id: str) -> str:
    return repo_id.strip().lower()


def normalize_job_key(key: str) -> str:
    repo, sep, variant = key.partition("::")
    repo_key = normalize_repo_key(repo)
    return f"{repo_key}{sep}{variant.strip().lower()}" if sep else repo_key


def _repo_of_key(key: str) -> str:
    return normalize_repo_key(key.split("::", 1)[0])


def variant_from_key(key: str) -> Optional[str]:
    """Parse the variant suffix from a 'repo_id::variant' key. Empty
    variant returns None — matches the manifest/marker calling
    convention for full-snapshot models and datasets."""
    if "::" not in key:
        return None
    _, _, variant = key.partition("::")
    return variant or None


def persist_cancel_marker(
    repo_type: Optional[RepoType],
    repo_id: Optional[str],
    variant: Optional[str],
    transport: Optional[str],
    *,
    hub_cache: Optional[str] = None,
    logger = logger,
) -> None:
    if not repo_type or not repo_id:
        return
    try:
        from hub.utils.download_manifest import write_cancel_marker
        if not write_cancel_marker(
            repo_type,
            repo_id,
            variant,
            transport = transport,
            hub_cache = hub_cache,
        ):
            logger.debug("write_cancel_marker returned False for %s", repo_id)
    except Exception as exc:
        logger.debug("write_cancel_marker failed for %s: %s", repo_id, exc)


_REGISTRIES: "weakref.WeakSet[DownloadRegistry]" = weakref.WeakSet()
_NAMED_REGISTRIES: dict[str, "DownloadRegistry"] = {}
_NAMED_REGISTRIES_LOCK = threading.Lock()


def terminate_active_downloads() -> None:
    """Best-effort shutdown hook called from the FastAPI lifespan.

    Walks every live DownloadRegistry instance and SIGKILLs any in-flight
    workers so the parent exit path doesn't leak zombies. The WeakSet drops
    ad-hoc registries (e.g. test fixtures) automatically once their last
    strong reference is gone; the long-lived named registries stay reachable
    via ``_NAMED_REGISTRIES``. Quiet on its own failures: shutdown must not
    raise.
    """
    for registry in list(_REGISTRIES):
        try:
            registry.terminate_all("download")
        except Exception as exc:
            logger.warning("terminate_active_downloads: %s", exc)


class DownloadRegistry:
    """Thread-safe state machine for background HF download jobs.

    One instance backs model downloads (keys ``repo_id::variant``) and another
    backs dataset downloads (keys ``repo_id``). Repo-scoped tracking serializes
    full snapshots, datasets, cross-transport work, and deletes; same-transport
    GGUF variants may run concurrently.
    """

    def __init__(self, max_terminal: int = 64) -> None:
        self._jobs: dict[str, DownloadState] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._repo_active: dict[str, set[str]] = {}
        self._metadata: dict[str, DownloadMetadata] = {}
        self._cancel_marker_transports: dict[str, str] = {}
        self._pending_cancel: dict[str, Optional[int]] = {}
        self._generations: dict[str, int] = {}
        # Monotonic across keys so an evicted then re-claimed key never reuses a prior generation, which
        # would let a stale cancel match a new run.
        self._generation_seq = 0
        self._deleting: dict[str, set[Optional[str]]] = {}
        # Publish external cache owners under the same lock as Model Hub jobs.
        self._repository_owners: dict[str, object] = {}
        self._lock = threading.Lock()
        _REGISTRIES.add(self)
        self._max_terminal = max_terminal

    def _put_terminal_job_locked(
        self,
        key: str,
        state: JobState,
        error: Optional[str] = None,
    ) -> None:
        self._jobs.pop(key, None)
        self._jobs[key] = DownloadState(state, error)
        if len(self._jobs) > self._max_terminal:
            for stale_key, stale in list(self._jobs.items()):
                if stale.state in TERMINAL_STATES and stale_key != key:
                    self._jobs.pop(stale_key, None)
                    self._metadata.pop(stale_key, None)
                    self._generations.pop(stale_key, None)
                    if len(self._jobs) <= self._max_terminal:
                        break

    def set_job(
        self,
        key: str,
        state: JobState,
        error: Optional[str] = None,
    ) -> None:
        key = normalize_job_key(key)
        with self._lock:
            if state in TERMINAL_STATES:
                self._put_terminal_job_locked(key, state, error)
                self._pending_cancel.pop(key, None)
                self._cancel_marker_transports.pop(key, None)
                repo = _repo_of_key(key)
                active = self._repo_active.get(repo)
                if active is not None:
                    active.discard(key)
                    if not active:
                        self._repo_active.pop(repo, None)
            else:
                self._jobs[key] = DownloadState(state, error)

    def set_error_unless_cancelled(
        self, key: str, error: str
    ) -> tuple[JobState, Optional[DownloadMetadata]]:
        key = normalize_job_key(key)
        with self._lock:
            current = self._jobs.get(key, DownloadState("idle")).state
            has_pending_cancel = key in self._pending_cancel
            pending_generation = self._pending_cancel.get(key)
            metadata = self._metadata.get(key)
            should_cancel = current == "cancelling" or (
                has_pending_cancel and self._generation_matches_locked(key, pending_generation)
            )
            terminal_state: JobState = "cancelled" if should_cancel else "error"
            marker_transport = self._cancel_marker_transports.pop(key, None)
            if marker_transport is None and metadata is not None:
                marker_transport = metadata.cancel_marker_transport
            self._put_terminal_job_locked(
                key,
                terminal_state,
                None if should_cancel else error,
            )
            self._pending_cancel.pop(key, None)
            repo = _repo_of_key(key)
            active = self._repo_active.get(repo)
            if active is not None:
                active.discard(key)
                if not active:
                    self._repo_active.pop(repo, None)
            if should_cancel and metadata is not None and marker_transport is not None:
                metadata = replace(metadata, transport = marker_transport)
            return terminal_state, metadata

    def job_transport(self, key: str) -> Optional[str]:
        """The transport a live job is running on. None when it has no metadata."""
        key = normalize_job_key(key)
        with self._lock:
            metadata = self._metadata.get(key)
            return metadata.transport if metadata is not None else None

    def job_cancel_transport(self, key: str) -> Optional[str]:
        """A live job's cancel marker, when a fallback left one. See metadata."""
        key = normalize_job_key(key)
        with self._lock:
            metadata = self._metadata.get(key)
            return metadata.cancel_marker_transport if metadata is not None else None

    def update_job_transport(self, key: str, transport: str) -> None:
        key = normalize_job_key(key)
        with self._lock:
            metadata = self._metadata.get(key)
            if metadata is None or metadata.transport == transport:
                return
            self._metadata[key] = replace(metadata, transport = transport)

    def release_active_slot(self, key: str) -> None:
        key = normalize_job_key(key)
        repo = _repo_of_key(key)
        with self._lock:
            active = self._repo_active.get(repo)
            if active is None:
                return
            active.discard(key)
            if not active:
                self._repo_active.pop(repo, None)

    def get_job(self, key: str) -> DownloadState:
        key = normalize_job_key(key)
        with self._lock:
            return self._jobs.get(key, DownloadState("idle"))

    def current_generation(self, key: str) -> int:
        key = normalize_job_key(key)
        with self._lock:
            return self._generations.get(key, 0)

    def get_job_metadata(self, key: str) -> Optional[DownloadMetadata]:
        key = normalize_job_key(key)
        with self._lock:
            return self._metadata.get(key)

    def _generation_matches_locked(self, key: str, generation: Optional[int]) -> bool:
        key = normalize_job_key(key)
        return generation is None or self._generations.get(key, 0) == generation

    def register_process(self, key: str, proc: subprocess.Popen) -> bool:
        """Register *proc* for *key*. Returns ``False`` when a cancel was
        requested during the claim→register window (the caller must kill
        *proc* immediately); ``True`` otherwise."""
        key = normalize_job_key(key)
        metadata_to_persist: Optional[DownloadMetadata] = None
        registered = False
        breadcrumb_metadata: Optional[DownloadMetadata] = None
        with self._lock:
            has_pending_cancel = key in self._pending_cancel
            pending_generation = self._pending_cancel.pop(key, None)
            if has_pending_cancel and self._generation_matches_locked(
                key,
                pending_generation,
            ):
                self._put_terminal_job_locked(key, "cancelled")
                metadata_to_persist = self._metadata.pop(key, None)
                marker_transport = self._cancel_marker_transports.pop(key, None)
                if marker_transport is None and metadata_to_persist is not None:
                    marker_transport = metadata_to_persist.cancel_marker_transport
                if metadata_to_persist is not None and marker_transport is not None:
                    metadata_to_persist = replace(
                        metadata_to_persist,
                        transport = marker_transport,
                    )
                repo = _repo_of_key(key)
                active = self._repo_active.get(repo)
                if active is not None:
                    active.discard(key)
                    if not active:
                        self._repo_active.pop(repo, None)
            else:
                self._processes[key] = proc
                breadcrumb_metadata = self._metadata.get(key)
                registered = True
        if registered:
            try:
                write_worker_breadcrumb(key, proc.pid, breadcrumb_metadata)
            except Exception as exc:
                logger.debug("Could not record worker breadcrumb: %s", exc)
            return True
        if metadata_to_persist is not None:
            persist_cancel_marker(
                metadata_to_persist.repo_type,
                metadata_to_persist.repo_id,
                metadata_to_persist.variant,
                metadata_to_persist.transport,
                hub_cache = metadata_to_persist.hub_cache,
            )
        return False

    def mark_pending_cancel(
        self,
        key: str,
        generation: Optional[int] = None,
    ) -> bool:
        """Record a cancel for an active job whose worker process hasn't
        registered yet. Returns ``True`` when the pending cancel was armed,
        so :func:`register_process` will kill the process on arrival."""
        key = normalize_job_key(key)
        with self._lock:
            if self._jobs.get(key, DownloadState("idle")).state not in _ACTIVE_STATES:
                return False
            if not self._generation_matches_locked(key, generation):
                return False
            self._pending_cancel[key] = generation
            self._jobs[key] = DownloadState("cancelling")
            return True

    def cancel_requested(self, key: str) -> bool:
        """True when *we* initiated a stop for *key* (a pending cancel armed
        before the worker registered, or the job already moved to
        ``cancelling``). Lets exit classification tell an intentional kill
        apart from an OOM/external SIGKILL."""
        key = normalize_job_key(key)
        with self._lock:
            if key in self._pending_cancel:
                return True
            return self._jobs.get(key, DownloadState("idle")).state == "cancelling"

    def get_process(self, key: str) -> Optional[subprocess.Popen]:
        key = normalize_job_key(key)
        with self._lock:
            return self._processes.get(key)

    def drop_process(self, key: str, proc: subprocess.Popen) -> bool:
        key = normalize_job_key(key)
        with self._lock:
            if self._processes.get(key) is not proc:
                return False
            self._processes.pop(key, None)
        remove_worker_breadcrumb(key)
        return True

    def claim(
        self,
        key: str,
        transport: str,
        *,
        repo_type: Optional[RepoType] = None,
        repo_id: Optional[str] = None,
        variant: Optional[str] = None,
        blob_hashes: Optional[frozenset[str]] = None,
        progress_blob_hashes: Optional[frozenset[str]] = None,
        completed_baseline_bytes: int = 0,
        admission_check: Optional[Callable[[], bool]] = None,
        generation: Optional[int] = None,
        replace_active: bool = False,
        metadata_transport: Optional[str] = None,
        cancel_marker_transport: Optional[str] = None,
        hub_cache: Optional[str] = None,
        xet_cache: Optional[str] = None,
        scoped_files: Optional[Sequence[str]] = None,
    ) -> tuple[bool, str]:
        key = normalize_job_key(key)
        repo = _repo_of_key(key)
        requested_hashes = blob_hashes or frozenset()
        requested_progress_hashes = progress_blob_hashes or frozenset()
        with self._lock:
            if repo in self._repository_owners:
                return False, "repository_owned"
            # Run the final admission check under the registry lock: the GGUF load path establishes its marker
            # before its active-job probe, so either this claim sees that marker or the load sees this claim.
            if admission_check is not None and not admission_check():
                return False, "admission_blocked"
            deleting_scopes = self._deleting.get(repo)
            if deleting_scopes is not None and (
                None in deleting_scopes or variant_from_key(key) in deleting_scopes
            ):
                return False, "deleting"
            active = self._repo_active.get(repo, set())
            stale_keys: list[str] = []
            conflict_state: Optional[str] = None
            for other_key in active:
                if other_key == key:
                    continue
                other_status = self._jobs.get(other_key)
                if other_status is None or other_status.state not in _ACTIVE_STATES:
                    stale_keys.append(other_key)
                    continue
                other_metadata = self._metadata.get(other_key)
                # Same-transport variants of one model run concurrently, since each worker purges only its own
                # re-resolved main blobs and the shared companion is guarded by its marker; cross-transport stays
                # serialized so an HTTP resume and an XET rewrite never write one blob at once.
                concurrent_gguf_variants = (
                    repo_type == "model"
                    and bool(variant)
                    and other_metadata is not None
                    and other_metadata.repo_type == "model"
                    and bool(other_metadata.variant)
                    and other_metadata.transport == transport
                )
                if concurrent_gguf_variants:
                    continue
                conflict_state = other_status.state
                break
            for stale_key in stale_keys:
                active.discard(stale_key)
            if conflict_state is not None:
                return False, conflict_state
            current = self._jobs.get(key, DownloadState("idle")).state
            if current in _ACTIVE_STATES and not replace_active:
                # A scope slot is shared by every file set that rides it, so adopting the live job would let the
                # caller wait on files it never asked for.
                live = self._metadata.get(key)
                if (
                    scoped_files is not None
                    and live is not None
                    and sorted(set(live.scoped_files)) != sorted(set(scoped_files))
                ):
                    return False, "scope_file_mismatch"
                return False, current
            if generation is None:
                self._generation_seq += 1
                self._generations[key] = self._generation_seq
            else:
                self._generations[key] = generation
            self._jobs[key] = DownloadState("running")
            self._repo_active.setdefault(repo, active).add(key)
            if repo_type and repo_id:
                self._metadata[key] = DownloadMetadata(
                    repo_type = repo_type,
                    repo_id = repo_id,
                    variant = variant,
                    transport = metadata_transport if metadata_transport is not None else transport,
                    cancel_marker_transport = cancel_marker_transport,
                    blob_hashes = requested_hashes,
                    progress_blob_hashes = requested_progress_hashes,
                    completed_baseline_bytes = max(
                        0,
                        int(completed_baseline_bytes or 0),
                    ),
                    hub_cache = hub_cache,
                    xet_cache = xet_cache,
                    scoped_files = tuple(scoped_files or ()),
                )
                if cancel_marker_transport is not None:
                    self._cancel_marker_transports[key] = cancel_marker_transport
                else:
                    self._cancel_marker_transports.pop(key, None)
            else:
                self._metadata.pop(key, None)
                self._cancel_marker_transports.pop(key, None)
            return True, "running"

    def claim_repository_owner(self, repo_id: str, owner: object) -> tuple[bool, str]:
        """Atomically reserve all cache writes for one repository.

        This covers snapshots, GGUF variants, and deletion. The opaque owner
        prevents a stale run from releasing a newer claim.
        """
        repo = normalize_repo_key(repo_id)
        with self._lock:
            if repo in self._repository_owners:
                return False, "repository_owned"
            if repo in self._deleting:
                return False, "deleting"
            for key, job in self._jobs.items():
                if _repo_of_key(key) != repo or job.state not in _ACTIVE_STATES:
                    continue
                # Retry handoffs can temporarily disappear from _repo_active.
                return False, job.state
            self._repository_owners[repo] = owner
            return True, "owned"

    def release_repository_owner(self, repo_id: str, owner: object) -> bool:
        """Release *repo_id* only when *owner* still holds its reservation."""
        repo = normalize_repo_key(repo_id)
        with self._lock:
            if self._repository_owners.get(repo) is not owner:
                return False
            self._repository_owners.pop(repo, None)
            return True

    def adoptable(self, key: str) -> bool:
        """True when *key* itself has a live job a client can attach to.

        Lets a rejected claim distinguish a collision with this key's own
        in-flight job (pollable) from one blocked by a different repo job
        or an in-progress delete, where no job exists for this key."""
        key = normalize_job_key(key)
        with self._lock:
            return self._jobs.get(key, DownloadState("idle")).state in _ACTIVE_STATES

    def _active_job_variant_locked(self, key: str) -> Optional[str]:
        metadata = self._metadata.get(key)
        if metadata is not None:
            return (metadata.variant or "").strip().lower() or None
        return variant_from_key(key)

    def _delete_blocked_by_active_locked(self, repo_id: str, variant: Optional[str]) -> bool:
        """Whether an active download conflicts with deleting *repo_id*/*variant*.

        A whole-repo delete (``variant is None``) conflicts with any active
        download. A variant delete conflicts only with that same variant or a
        whole-repo download writing the shared snapshot; other quantizations
        download concurrently and never block it."""
        active_keys = self._repo_active.get(repo_id, set())
        for key in active_keys:
            job = self._jobs.get(key)
            if job is None or job.state not in _ACTIVE_STATES:
                continue
            if variant is None:
                return True
            other_variant = self._active_job_variant_locked(key)
            if other_variant is None or other_variant == variant:
                return True
        for key, job in self._jobs.items():
            if key in active_keys or _repo_of_key(key) != repo_id:
                continue
            if job.state not in _ACTIVE_STATES:
                continue
            if variant is None:
                return True
            other_variant = self._active_job_variant_locked(key)
            if other_variant is None or other_variant == variant:
                return True
        return False

    def peer_blob_hashes(self, key: str) -> frozenset[str]:
        """Union of the writable blob hashes of every OTHER active download for
        this key's repo. A worker excludes these from its purge so it never
        deletes an ``.incomplete`` a concurrent same-repo variant is writing
        (e.g. a shared mmproj bundled with two GGUF quants)."""
        key = normalize_job_key(key)
        repo = _repo_of_key(key)
        out: set[str] = set()
        with self._lock:
            for other_key in self._repo_active.get(repo, set()):
                if other_key == key:
                    continue
                job = self._jobs.get(other_key)
                if job is None or job.state not in _ACTIVE_STATES:
                    continue
                metadata = self._metadata.get(other_key)
                if metadata is not None:
                    out |= set(metadata.progress_blob_hashes or metadata.blob_hashes)
        return frozenset(out)

    def active_jobs(self, repo_id: str) -> dict[str, str]:
        """Map of every active job key for *repo_id* to its state."""
        repo_id = normalize_repo_key(repo_id)
        with self._lock:
            result: dict[str, str] = {}
            for key in self._repo_active.get(repo_id, set()):
                job = self._jobs.get(key)
                if job is not None and job.state in _ACTIVE_STATES:
                    metadata = self._metadata.get(key)
                    display_key = (
                        f"{_repo_of_key(key)}::{metadata.variant}"
                        if metadata is not None and metadata.variant
                        else key
                    )
                    result[display_key] = job.state
            return result

    def active_job_refs(self, repo_id: Optional[str] = None) -> list[ActiveDownloadRef]:
        repo_key = normalize_repo_key(repo_id) if repo_id else None
        with self._lock:
            if repo_key:
                candidate_keys = list(self._repo_active.get(repo_key, set()))
            else:
                candidate_keys = [key for active in self._repo_active.values() for key in active]
            # An XET->HTTP retry handoff briefly drops its key from _repo_active while its job stays active, so
            # include those released-but-active jobs.
            seen = set(candidate_keys)
            for key, job in self._jobs.items():
                if key in seen or job.state not in _ACTIVE_STATES:
                    continue
                if repo_key is not None and _repo_of_key(key) != repo_key:
                    continue
                candidate_keys.append(key)
            refs: list[ActiveDownloadRef] = []
            for key in candidate_keys:
                job = self._jobs.get(key)
                if job is None or job.state not in _ACTIVE_STATES:
                    continue
                refs.append(
                    ActiveDownloadRef(
                        key = key,
                        state = job.state,
                        metadata = self._metadata.get(key),
                        generation = self._generations.get(key, 0),
                    )
                )
            return refs

    def has_active_variant(self, repo_id: str, variant: Optional[str]) -> bool:
        """Whether an active model job targets this exact GGUF variant.

        Scans the job table rather than only ``_repo_active`` so an XET-to-HTTP
        retry handoff remains visible while it has temporarily released its
        active slot.
        """
        repo_key = normalize_repo_key(repo_id)
        target = (variant or "").strip().lower() or None
        with self._lock:
            for key, job in self._jobs.items():
                if _repo_of_key(key) != repo_key or job.state not in _ACTIVE_STATES:
                    continue
                if self._active_job_variant_locked(key) == target:
                    return True
        return False

    def begin_delete(
        self,
        repo_id: str,
        variant: Optional[str] = None,
    ) -> bool:
        """Reserve *repo_id* (or one GGUF *variant* of it) for deletion. Returns
        ``False`` when a conflicting download is active (a whole-repo delete vs
        any download, a variant delete vs that same variant or a whole-repo
        download), so sibling quantizations keep downloading. On success the
        scope is marked so :func:`claim` rejects overlapping downloads until
        :func:`end_delete` runs, closing the check-then-delete race against a
        concurrently spawned worker."""
        repo_id = normalize_repo_key(repo_id)
        variant_key = (variant or "").strip().lower() or None
        with self._lock:
            if repo_id in self._repository_owners:
                return False
            if self._delete_blocked_by_active_locked(repo_id, variant_key):
                return False
            self._deleting.setdefault(repo_id, set()).add(variant_key)
            return True

    def end_delete(
        self,
        repo_id: str,
        variant: Optional[str] = None,
    ) -> None:
        repo_id = normalize_repo_key(repo_id)
        variant_key = (variant or "").strip().lower() or None
        with self._lock:
            scopes = self._deleting.get(repo_id)
            if scopes is None:
                return
            scopes.discard(variant_key)
            if not scopes:
                self._deleting.pop(repo_id, None)

    def has_active_peer_variant(self, repo_id: str, variant: Optional[str]) -> bool:
        """Whether a DIFFERENT quantization of *repo_id* is downloading while
        *variant* is being deleted. When one is, the delete reclaims only this
        variant's files and leaves the shared companion (mmproj) for the live
        sibling. Point-in-time (a sibling may claim just after it returns), but
        safe: the finalized companion is held by deletion's reference-count
        walk and a sibling starting mid-delete re-fetches it, so protection
        never depends on the sibling having resolved its blob hashes."""
        repo_id = normalize_repo_key(repo_id)
        target = (variant or "").strip().lower() or None
        with self._lock:
            active_keys = self._repo_active.get(repo_id, set())
            for key in active_keys:
                job = self._jobs.get(key)
                if job is None or job.state not in _ACTIVE_STATES:
                    continue
                if self._active_job_variant_locked(key) != target:
                    return True
            # A retry peer between release_active_slot() and its reclaim is briefly absent from _repo_active
            # while it still owns the shared companion, so mirror the released-but-active scan.
            for key, job in self._jobs.items():
                if key in active_keys or _repo_of_key(key) != repo_id:
                    continue
                if job.state not in _ACTIVE_STATES:
                    continue
                if self._active_job_variant_locked(key) != target:
                    return True
        return False

    def request_cancel(
        self,
        key: str,
        proc: subprocess.Popen,
        generation: Optional[int] = None,
    ) -> bool:
        """Authorize a SIGKILL for the registered *proc*. Idempotent across an
        active job's lifetime: a repeated cancel while already ``cancelling``
        still returns ``True`` so a kill that raced and lost can be re-sent."""
        key = normalize_job_key(key)
        with self._lock:
            if self._processes.get(key) is not proc:
                return False
            if not self._generation_matches_locked(key, generation):
                return False
            if self._jobs.get(key, DownloadState("idle")).state not in _ACTIVE_STATES:
                return False
            self._jobs[key] = DownloadState("cancelling")
            return True

    def terminate_all(self, kind: str = "download") -> None:
        settled_no_proc: list[Optional[DownloadMetadata]] = []
        with self._lock:
            live = [
                (key, proc, self._metadata.get(key))
                for key, proc in self._processes.items()
                if proc.poll() is None
            ]
            live_keys = {key for key, _proc, _metadata in live}
            # Flag as an intentional stop so the watcher's exit classification
            for key, _proc, _metadata in live:
                if self._jobs.get(key, DownloadState("idle")).state == "running":
                    self._jobs[key] = DownloadState("cancelling")
            # Settle active jobs without a live worker: a retry parked in the reclaim wait has dropped its
            # worker, and a registered worker that errored before its watcher ran would stay running and spawn
            # an HTTP retry. Skip one that exited cleanly, which would strand a stale marker.
            for key, job in list(self._jobs.items()):
                if job.state not in _ACTIVE_STATES or key in live_keys:
                    continue
                proc = self._processes.get(key)
                if proc is not None:
                    if proc.poll() == 0:
                        continue
                    # A registered worker that exited nonzero over HTTP is a genuine terminal failure, not a shutdown
                    # cancel: only an exited XET worker could still spawn a post-shutdown HTTP retry.
                    metadata = self._metadata.get(key)
                    if metadata is not None and metadata.transport == TRANSPORT_HTTP:
                        continue
                self._pending_cancel[key] = self._generations.get(key)
                self._jobs[key] = DownloadState("cancelling")
                settled_no_proc.append(self._metadata.get(key))
        # Persist the cancel marker outside the lock so shutdown records resumable state even if it returns
        # before the daemon watcher wakes.
        for metadata in settled_no_proc:
            if metadata is not None:
                persist_cancel_marker(
                    metadata.repo_type,
                    metadata.repo_id,
                    metadata.variant,
                    metadata.cancel_marker_transport or metadata.transport,
                    hub_cache = metadata.hub_cache,
                )
        reaped: list[tuple[str, subprocess.Popen, Optional[DownloadMetadata]]] = []
        for key, proc, metadata in live:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            except Exception as e:
                logger.warning(f"shutdown: failed to kill {kind} worker for {key}: {e}")
                if metadata is not None:
                    persist_cancel_marker(
                        metadata.repo_type,
                        metadata.repo_id,
                        metadata.variant,
                        metadata.cancel_marker_transport or metadata.transport,
                        hub_cache = metadata.hub_cache,
                    )
                continue
            reaped.append((key, proc, metadata))
        deadline = time.monotonic() + 10.0
        for key, proc, metadata in reaped:
            try:
                proc.wait(timeout = max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                logger.warning(f"shutdown: {kind} worker for {key} did not exit after kill")
            except Exception:
                pass
            # Mark only genuinely interrupted workers: persisting before the exit is known would strand a stale
            # marker on a worker that completed cleanly during shutdown.
            if metadata is not None and proc.poll() != 0:
                persist_cancel_marker(
                    metadata.repo_type,
                    metadata.repo_id,
                    metadata.variant,
                    metadata.cancel_marker_transport or metadata.transport,
                    hub_cache = metadata.hub_cache,
                )


def _named_registry(name: str) -> DownloadRegistry:
    with _NAMED_REGISTRIES_LOCK:
        registry = _NAMED_REGISTRIES.get(name)
        if registry is None:
            registry = DownloadRegistry()
            _NAMED_REGISTRIES[name] = registry
        return registry


def get_models_registry() -> DownloadRegistry:
    return _named_registry("models")


def get_datasets_registry() -> DownloadRegistry:
    return _named_registry("datasets")
