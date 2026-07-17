# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HF cache inspection, download registry state, and orphan-worker reaping.

Worker spawning and exit handling live in
:mod:`hub.services.download_lifecycle`; this module owns the registry state
machine plus the cache/marker inspection those workers depend on.

Resume model
------------
Only the HTTP transport supports true partial-file resume:
huggingface_hub's HTTP resumer opens ``<etag>.incomplete`` in append mode
and sends ``Range: bytes={resume_size}-`` to continue from disk.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Literal, Optional

from loggers import get_logger

from hub.utils import state_dir
from hub.utils.state_dir import RepoType

logger = get_logger(__name__)

from hub.utils.hf_cache_state import (
    INCOMPLETE_SUFFIX,
    TRANSPORT_HTTP,
    TRANSPORT_XET,
    TRANSPORT_MARKER_NAME,
    VALID_TRANSPORTS,
    has_active_incomplete_blobs,
    iter_repo_cache_dirs,
    iter_active_repo_cache_dirs,
    repo_cache_dir_name,
    target_dir_name,
    hf_cache_root,
)


@dataclass(frozen = True)
class DownloadTransportCapability:
    available: bool
    reason: Optional[str] = None


@dataclass(frozen = True)
class DownloadTransportCapabilities:
    http: DownloadTransportCapability
    xet: DownloadTransportCapability


def get_download_transport_capabilities() -> DownloadTransportCapabilities:
    xet_available = importlib.util.find_spec("hf_xet") is not None
    return DownloadTransportCapabilities(
        http = DownloadTransportCapability(available = True),
        xet = DownloadTransportCapability(
            available = xet_available,
            reason = None
            if xet_available
            else "Xet transport is unavailable because hf_xet is not installed.",
        ),
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
    # Exact --repo-id match: a substring match would let a stale breadcrumb for
    # Org/Model reap a live worker for Org/Model-v2.
    if isinstance(repo_id, str) and repo_id:
        return _cmdline_repo_id(cmdline) == repo_id
    return True


def _kill_orphan(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL)
    except OSError:
        pass


def _settle_orphaned_download(
    repo_type: Optional[str],
    repo_id: Optional[str],
    variant: Optional[str],
    transport: Optional[str],
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

    manifest = download_manifest.read_manifest(repo_type, repo_id, variant)
    if repo_type == "model" and variant and manifest is None:
        return
    if manifest is None:
        if not has_active_incomplete_blobs(repo_type, repo_id):
            return
    else:
        if _manifest_verifies_against_active_cache(repo_type, repo_id, manifest):
            return
        if not _manifest_has_active_incomplete_blobs(repo_type, repo_id, manifest):
            return
    persist_cancel_marker(repo_type, repo_id, variant, transport, logger = logger)


def reap_orphan_workers() -> None:
    """Kill download workers left running by a previous backend instance.

    Verifies each breadcrumb's PID is alive AND its command line is one of our
    workers before terminating, so a recycled PID can't take down an unrelated
    process. Partial blobs are never touched, so a reaped download stays
    resumable; an interrupted one with bytes on disk is settled to a cancelled
    marker (see :func:`_settle_orphaned_download`) so its resume affordance
    survives a hard crash like a graceful shutdown's does. Runs once at startup
    and never raises."""
    parent = state_dir.workers_dir()
    if parent is None:
        return
    try:
        entries = list(parent.iterdir())
    except OSError:
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
        if not isinstance(pid, int) or pid <= 0:
            _safe_unlink(entry)
            continue
        try:
            if _process_alive(pid) and _is_our_worker(pid, repo_id):
                _kill_orphan(pid)
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
                data.get("transport"),
            )
        except Exception as exc:
            logger.debug("Reaper failed for breadcrumb %s: %s", entry, exc)
        _safe_unlink(entry)


def _purge_incomplete_blobs(
    entry: Path,
    only_hashes: Optional[frozenset[str]] = None,
    protected_hashes: Optional[frozenset[str]] = None,
) -> int:
    """Delete matching ``*.incomplete`` blobs beneath *entry*; return the count
    removed. Per-file failures are swallowed.

    ``only_hashes`` whitelists which partials may be purged; ``None`` means
    every partial (full-repo snapshot/dataset). ``protected_hashes`` is honoured
    unconditionally, even when ``only_hashes`` is ``None``, so a blob a
    concurrent same-repo peer is writing is never purged from under it."""
    blobs_dir = entry / "blobs"
    if not blobs_dir.is_dir():
        return 0
    removed = 0
    try:
        candidates = list(blobs_dir.iterdir())
    except OSError:
        return 0
    for blob in candidates:
        try:
            if not blob.is_file():
                continue
            if not blob.name.endswith(INCOMPLETE_SUFFIX):
                continue
            blob_hash = blob.name[: -len(INCOMPLETE_SUFFIX)]
            if protected_hashes and blob_hash in protected_hashes:
                continue
            if only_hashes is not None and blob_hash not in only_hashes:
                continue
            blob.unlink()
            removed += 1
        except OSError:
            # Swallow; downstream snapshot_download surfaces a precise error if
            # it actually can't proceed.
            continue
    return removed


def _iter_active_snapshot_dirs(repo_type: str, repo_id: str) -> Iterator[Path]:
    for entry in iter_active_repo_cache_dirs(repo_type, repo_id):
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


def _manifest_verifies_against_active_cache(repo_type: str, repo_id: str, manifest) -> bool:
    from hub.utils import download_manifest
    for snapshot_dir in _iter_active_snapshot_dirs(repo_type, repo_id):
        if download_manifest.verify_against_disk(manifest, snapshot_dir).ok:
            return True
    return False


def _manifest_has_active_incomplete_blobs(repo_type: str, repo_id: str, manifest) -> bool:
    if not getattr(manifest, "variant", None):
        return has_active_incomplete_blobs(repo_type, repo_id)
    expected_hashes = frozenset(
        expected.sha256 for expected in manifest.expected_files if expected.sha256
    )
    if not expected_hashes:
        return has_active_incomplete_blobs(repo_type, repo_id)
    return bool(
        incomplete_blob_hashes(repo_type, repo_id, active_only = True).intersection(expected_hashes)
    )


def _marker_path(entry: Path, variant: Optional[str] = None) -> Path:
    if not variant:
        return entry / TRANSPORT_MARKER_NAME
    digest = hashlib.sha256(variant.strip().lower().encode("utf-8")).hexdigest()[:24]
    return entry / f"{TRANSPORT_MARKER_NAME}.gguf-{digest}"


def _is_transport_marker_file(path: Path) -> bool:
    # Matches ".transport", its tmps, and variant-scoped ".transport.gguf-*".
    # Real HF cache entries (blobs/refs/snapshots/.no_exist) never start with
    # ".transport.".
    return path.name == TRANSPORT_MARKER_NAME or path.name.startswith(f"{TRANSPORT_MARKER_NAME}.")


def _companion_marker_path(entry: Path) -> Path:
    return entry / f"{TRANSPORT_MARKER_NAME}.companion"


def _read_marker_value(marker: Path) -> Optional[str]:
    try:
        if not marker.exists():
            return None
        value = marker.read_text().strip()
    except OSError:
        return None
    return value if value in VALID_TRANSPORTS else None


def _write_marker_value(marker: Path, mode: str) -> None:
    try:
        # tmp + rename so a SIGKILL mid-write can't leave a half-written marker.
        # The tmp name is per-process so concurrent writers don't clobber tmps.
        tmp = marker.with_name(f"{marker.name}.tmp-{os.getpid()}")
        tmp.write_text(mode)
        os.replace(tmp, marker)
    except OSError:
        # Best-effort: a missing marker next run purges the partial defensively,
        # the safe failure mode.
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
      XET/parallel-Range partial and silently produce a corrupt blob.
    - XET mode: incomplete blobs are purged (``hf_xet.download_files`` rewrites
      from scratch, so this only fixes UI accounting — bytes already in CAS are
      reused via the chunk-cache). Scoped to ``only_blob_hashes``: companion
      blobs fall outside that set and survive (shared, and XET overwrites them).

    ``protected_blob_hashes`` are blobs a concurrent same-repo peer is writing;
    they are excluded from every purge so a shared companion is never deleted
    mid-write.

    Scope: only the active ``HF_HUB_CACHE`` root is inspected. That suffices for
    resume safety because ``snapshot_download`` runs without a ``cache_dir``
    override and so can only read or resume a ``.incomplete`` under this same
    active root. Markers are written for the new mode before returning.
    """
    if mode not in VALID_TRANSPORTS:
        raise ValueError(f"Invalid transport mode: {mode!r}")
    root = hf_cache_root(create = True)
    if root is None:
        return 0
    target = target_dir_name(repo_type, repo_id)
    try:
        entries = [e for e in root.iterdir() if e.name.lower() == target]
    except OSError:
        return 0
    if not entries:
        # First download: pre-create the repo dir so the marker lands before the
        # worker writes any bytes. Otherwise a SIGKILL mid-download leaves a
        # partial with no marker that the resume then purges.
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
        if mode == TRANSPORT_XET:
            total_purged += _purge_incomplete_blobs(entry, only_blob_hashes, protected)
        else:
            if _read_marker(entry, variant) != mode:
                total_purged += _purge_incomplete_blobs(entry, only_blob_hashes, protected)
            if companion_blob_hashes and _read_companion_marker(entry) != mode:
                total_purged += _purge_incomplete_blobs(entry, companion_blob_hashes, protected)
        _write_marker(entry, mode, variant)
        if has_companion:
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
) -> Optional[str]:
    for entry in iter_active_repo_cache_dirs(repo_type, repo_id):
        value = _read_marker(entry, variant)
        if value is not None:
            return value
    return None


def is_resumable_partial(
    repo_type: str,
    repo_id: str,
    variant: Optional[str] = None,
) -> bool:
    """True only when a partial exists AND was produced by a byte-resumable
    writer (the HTTP transport). XET partials exist on disk but are discarded on
    the next download attempt."""
    if not has_active_incomplete_blobs(repo_type, repo_id):
        return False
    return read_active_transport_marker(repo_type, repo_id, variant) == TRANSPORT_HTTP


def incomplete_blob_hashes(
    repo_type: str,
    repo_id: str,
    *,
    active_only: bool = False,
) -> set[str]:
    out: set[str] = set()
    entries = (
        iter_active_repo_cache_dirs(repo_type, repo_id)
        if active_only
        else iter_repo_cache_dirs(repo_type, repo_id)
    )
    for entry in entries:
        blobs_dir = entry / "blobs"
        if not blobs_dir.is_dir():
            continue
        try:
            for blob in blobs_dir.iterdir():
                if blob.is_file() and blob.name.endswith(INCOMPLETE_SUFFIX):
                    out.add(blob.name[: -len(INCOMPLETE_SUFFIX)])
        except OSError:
            continue
    return out


def completed_blob_bytes(repo_type: str, repo_id: str, blob_hashes: frozenset[str]) -> int:
    """Sum finalized blob bytes for *blob_hashes* in the active HF cache root.

    A worker only writes to the active ``HF_HUB_CACHE`` root, so a baseline must
    ignore legacy/default roots that ``snapshot_download`` won't reuse this run.
    """
    if not blob_hashes:
        return 0
    total = 0
    for entry in iter_active_repo_cache_dirs(repo_type, repo_id):
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


def existing_blob_bytes(repo_type: str, repo_id: str, blob_hashes: frozenset[str]) -> int:
    """Bytes already on disk (finalized + ``.incomplete``) for *blob_hashes* in
    the active HF cache root. A blob is in exactly one state, so summing both
    candidate names never double-counts. Used to size what a (possibly resumed)
    download still needs to write before the run starts."""
    if not blob_hashes:
        return 0
    total = 0
    for entry in iter_active_repo_cache_dirs(repo_type, repo_id):
        blobs_dir = entry / "blobs"
        if not blobs_dir.is_dir():
            continue
        for blob_hash in blob_hashes:
            for name in (blob_hash, f"{blob_hash}{INCOMPLETE_SUFFIX}"):
                blob = blobs_dir / name
                try:
                    if blob.is_file():
                        total += max(0, int(blob.stat().st_size))
                except OSError:
                    continue
    return total


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
    # GGUF variant main/writable hashes, identifying the variant-specific shards
    # for concurrency decisions.
    blob_hashes: frozenset[str] = field(default_factory = frozenset)
    # Full required hash set for progress/completion (includes the shared mmproj
    # companion for vision GGUF repos).
    progress_blob_hashes: frozenset[str] = field(default_factory = frozenset)
    # Bytes already complete before this job started; not counted as this run's
    # progress.
    completed_baseline_bytes: int = 0


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
        self._pending_cancel: dict[str, Optional[int]] = {}
        self._generations: dict[str, int] = {}
        # Monotonic across keys so an evicted then re-claimed key never reuses a
        # prior generation (which would let a stale cancel match a new run).
        self._generation_seq = 0
        self._deleting: dict[str, set[Optional[str]]] = {}
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
                repo = _repo_of_key(key)
                active = self._repo_active.get(repo)
                if active is not None:
                    active.discard(key)
                    if not active:
                        self._repo_active.pop(repo, None)
            else:
                self._jobs[key] = DownloadState(state, error)

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
    ) -> tuple[bool, str]:
        key = normalize_job_key(key)
        repo = _repo_of_key(key)
        requested_hashes = blob_hashes or frozenset()
        requested_progress_hashes = progress_blob_hashes or frozenset()
        with self._lock:
            # Run the final external admission check while the registry lock is
            # held, immediately before inspecting and publishing active state.
            # The GGUF load path establishes its marker before calling
            # active_jobs(), so either this claim observes that marker or the
            # load's later active_jobs() observes this claim.
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
                # Same-transport variants of one model run concurrently: each
                # worker purges only its own re-resolved main blobs and the
                # shared companion is guarded by its marker. Cross-transport
                # stays serialized so an HTTP resume and an XET rewrite never
                # write one shared blob at once.
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
            if current in _ACTIVE_STATES:
                return False, current
            self._generation_seq += 1
            self._generations[key] = self._generation_seq
            self._jobs[key] = DownloadState("running")
            self._repo_active.setdefault(repo, active).add(key)
            if repo_type and repo_id:
                self._metadata[key] = DownloadMetadata(
                    repo_type = repo_type,
                    repo_id = repo_id,
                    variant = variant,
                    transport = transport,
                    blob_hashes = requested_hashes,
                    progress_blob_hashes = requested_progress_hashes,
                    completed_baseline_bytes = max(
                        0,
                        int(completed_baseline_bytes or 0),
                    ),
                )
            else:
                self._metadata.pop(key, None)
            return True, "running"

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
        for key in self._repo_active.get(repo_id, set()):
            job = self._jobs.get(key)
            if job is None or job.state not in _ACTIVE_STATES:
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
            for key in self._repo_active.get(repo_id, set()):
                job = self._jobs.get(key)
                if job is None or job.state not in _ACTIVE_STATES:
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
        with self._lock:
            live = [
                (key, proc, self._metadata.get(key))
                for key, proc in self._processes.items()
                if proc.poll() is None
            ]
            # Flag as an intentional stop so the watcher's exit classification
            # reports them cancelled rather than an OOM/crash once SIGKILL lands.
            for key, _proc, _metadata in live:
                if self._jobs.get(key, DownloadState("idle")).state == "running":
                    self._jobs[key] = DownloadState("cancelling")
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
                        metadata.transport,
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
            # Mark only genuinely interrupted workers (rc != 0, or None on wait
            # timeout); persisting before the exit is known would strand a stale
            # marker on a worker that completed cleanly during shutdown.
            if metadata is not None and proc.poll() != 0:
                persist_cancel_marker(
                    metadata.repo_type,
                    metadata.repo_id,
                    metadata.variant,
                    metadata.transport,
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
