# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared HF cache inspection + download-worker spawning.

Resume model
------------
Only the HTTP transport supports true partial-file resume.
huggingface_hub's HTTP resumer opens ``<etag>.incomplete`` in append
mode, reads ``resume_size = f.tell()``, and sends ``Range:
bytes={resume_size}-`` to continue from disk.

The XET transport CANNOT resume from a ``.incomplete`` partial:
``hf_xet.download_files`` takes only ``(destination_path, hash,
file_size)`` and rewrites the destination from scratch. Network-level
deduplication does still happen, but through the chunk cache at
``~/.cache/huggingface/xet/chunk-cache`` (separate from the repo cache),
which is never touched by these helpers.

Cross-transport corruption: a partial written by XET (or by
``hf_transfer``'s parallel-Range writer) can be sparse — high reported
size, zero-filled gaps below. Feeding such a file to the HTTP resumer
would produce a final blob of correct size whose internal bytes are
silently wrong. To prevent that, we maintain a per-repo ``.transport``
marker and refuse to inherit any HTTP partial unless the marker proves
the previous writer was the same single-stream sequential writer.

Marker writes go through tmp+rename in :func:`prepare_cache_for_transport`
before the worker hands off to ``snapshot_download``, so the next process
always sees a consistent provenance signal.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import signal
import stat as stat_module
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from loggers import get_logger

logger = get_logger(__name__)


EXIT_CANCELLED = 130

TRANSPORT_HTTP = "http"
TRANSPORT_XET = "xet"
_VALID_TRANSPORTS = frozenset({TRANSPORT_HTTP, TRANSPORT_XET})
_MARKER_NAME = ".transport"
_INCOMPLETE_SUFFIX = ".incomplete"
CacheProgressReading = tuple[int, int, Optional[str]]


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
            reason = None if xet_available else "Xet transport is unavailable because hf_xet is not installed.",
        ),
    )


def download_transport_unavailable_reason(transport: str) -> Optional[str]:
    if transport == TRANSPORT_HTTP:
        return None
    if transport == TRANSPORT_XET:
        caps = get_download_transport_capabilities().xet
        return None if caps.available else caps.reason
    return f"Unsupported download transport: {transport}"


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


_DOWNLOAD_STALL_TIMEOUT_SECONDS = _float_env(
    "UNSLOTH_HF_DOWNLOAD_STALL_TIMEOUT_SECONDS", 300.0
)
_DOWNLOAD_STALL_POLL_SECONDS = _float_env(
    "UNSLOTH_HF_DOWNLOAD_STALL_POLL_SECONDS", 15.0
)
_DOWNLOAD_STALL_STARTUP_GRACE_SECONDS = _float_env(
    "UNSLOTH_HF_DOWNLOAD_STALL_STARTUP_GRACE_SECONDS", 30.0
)
_HF_CACHE_SCANS_TTL_SECONDS = 3.0
_hf_cache_scans_lock = threading.Lock()


@dataclass
class _HfCacheScanFlight:
    event: threading.Event
    result: Optional[list] = None
    error: Optional[BaseException] = None


_hf_cache_scans_flight: Optional[_HfCacheScanFlight] = None
_hf_cache_scans_result: Optional[list] = None
_hf_cache_scans_cached_at: float = 0.0


def invalidate_hf_cache_scans() -> None:
    global _hf_cache_scans_result, _hf_cache_scans_cached_at
    with _hf_cache_scans_lock:
        _hf_cache_scans_result = None
        _hf_cache_scans_cached_at = 0.0


def all_hf_cache_scans() -> list:
    global _hf_cache_scans_flight, _hf_cache_scans_result, _hf_cache_scans_cached_at

    now = time.monotonic()
    with _hf_cache_scans_lock:
        if (
            _hf_cache_scans_result is not None
            and (now - _hf_cache_scans_cached_at) < _HF_CACHE_SCANS_TTL_SECONDS
        ):
            return list(_hf_cache_scans_result)
        flight = _hf_cache_scans_flight
        if flight is None:
            flight = _HfCacheScanFlight(event = threading.Event())
            _hf_cache_scans_flight = flight
            owner = True
        else:
            owner = False

    if not owner:
        flight.event.wait()
        if flight.error is not None:
            raise flight.error
        return list(flight.result or [])

    try:
        scans = _compute_all_hf_cache_scans()
        with _hf_cache_scans_lock:
            _hf_cache_scans_result = scans
            _hf_cache_scans_cached_at = time.monotonic()
            flight.result = scans
        return scans
    except Exception as exc:
        with _hf_cache_scans_lock:
            _hf_cache_scans_result = None
            _hf_cache_scans_cached_at = 0.0
            flight.error = exc
        raise
    finally:
        with _hf_cache_scans_lock:
            if _hf_cache_scans_flight is flight:
                _hf_cache_scans_flight = None
            flight.event.set()


def _compute_all_hf_cache_scans() -> list:
    from huggingface_hub import scan_cache_dir
    from utils.paths import legacy_hf_cache_dir, hf_default_cache_dir

    scans: list = []
    seen: set[str] = set()
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        active = Path(HF_HUB_CACHE).resolve()
        seen.add(str(active))
        if active.is_dir():
            scans.append(scan_cache_dir())
    except Exception as exc:
        logger.warning("Could not scan active HF cache: %s", exc)

    for extra_fn in (legacy_hf_cache_dir, hf_default_cache_dir):
        extra = extra_fn()
        if extra.is_dir() and str(extra.resolve()) not in seen:
            seen.add(str(extra.resolve()))
            try:
                scans.append(scan_cache_dir(cache_dir = str(extra)))
            except Exception as exc:
                logger.warning("Could not scan HF cache %s: %s", extra, exc)
    return scans


def token_fingerprint(hf_token: Optional[str]) -> str:
    """16-char SHA256 prefix used as a cache-key qualifier for gated repos.

    Lets per-token size/snapshot caches refuse to serve a previously
    fetched value back to a different token (a private/gated repo's
    metadata is only valid for the credential that fetched it).
    """
    if not hf_token:
        return ""
    return hashlib.sha256(hf_token.encode()).hexdigest()[:16]


def resolve_hf_cache_realpath(repo_dir: Path) -> Optional[str]:
    """Pick the most useful on-disk path for a HF cache repo dir.

    Prefers the most-recent snapshot dir (what ``from_pretrained``
    actually points at). Falls back to the cache repo root. Returns the
    resolved realpath so symlinks under ``snapshots/`` are followed back
    to ``blobs/``.
    """
    try:
        snapshots_dir = repo_dir / "snapshots"
        if snapshots_dir.is_dir():
            snaps = [s for s in snapshots_dir.iterdir() if s.is_dir()]
            if snaps:
                latest = max(snaps, key = lambda s: s.stat().st_mtime)
                return str(latest.resolve())
        return str(repo_dir.resolve())
    except Exception:
        return None


def latest_snapshot_from_cache_path(
    local_path: Optional[str],
    repo_type: str,
    repo_id: str,
    metadata_filenames: tuple[str, ...] = (),
) -> Optional[str]:
    if not local_path or not repo_id:
        return None
    try:
        root = Path(local_path).expanduser()
        if not root.exists():
            return None
        expected_repo_dir = _target_dir_name(repo_type, repo_id)
        if expected_repo_dir not in {part.lower() for part in root.parts}:
            return None

        def has_metadata(path: Path) -> bool:
            if not metadata_filenames:
                return True
            return any((path / name).is_file() for name in metadata_filenames)

        candidates: list[Path] = []
        if root.is_dir() and has_metadata(root):
            candidates.append(root)
        snapshots = root / "snapshots" if root.is_dir() else None
        if snapshots is not None and snapshots.is_dir():
            candidates.extend(
                p for p in snapshots.iterdir() if p.is_dir() and has_metadata(p)
            )
        if not candidates:
            return None
        candidates.sort(
            key = lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse = True,
        )
        return str(candidates[0].resolve())
    except Exception:
        return None


def _hf_cache_root(*, create: bool = False) -> Optional[Path]:
    try:
        from huggingface_hub import constants as hf_constants
    except ImportError:
        return None
    root = Path(hf_constants.HF_HUB_CACHE)
    if create:
        try:
            root.mkdir(parents = True, exist_ok = True)
        except OSError:
            return None
        return root
    return root if root.is_dir() else None


def _hf_cache_roots() -> list[Path]:
    """Existing HF hub caches used by read-only inventory scans."""
    from utils.paths import hf_default_cache_dir, legacy_hf_cache_dir

    roots: list[Path] = []
    seen: set[str] = set()

    def _add(path: Optional[Path]) -> None:
        if path is None or not path.is_dir():
            return
        try:
            key = str(path.resolve())
        except OSError:
            return
        if key in seen:
            return
        seen.add(key)
        roots.append(path)

    _add(_hf_cache_root())
    _add(legacy_hf_cache_dir())
    _add(hf_default_cache_dir())
    return roots


def _target_dir_name(repo_type: str, repo_id: str) -> str:
    return f"{repo_type}s--{repo_id.replace('/', '--')}".lower()


def _blob_dir_is_partial(blobs_dir: Path) -> bool:
    try:
        for blob in blobs_dir.iterdir():
            if blob.is_file() and blob.name.endswith(_INCOMPLETE_SUFFIX):
                return True
    except OSError:
        return False
    return False


def _repo_dir_has_broken_snapshot_symlinks(repo_dir: Path) -> bool:
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return False
    try:
        snapshots = [entry for entry in snapshots_dir.iterdir() if entry.is_dir()]
        if not snapshots:
            return False
        latest = max(snapshots, key = lambda entry: entry.stat().st_mtime)
        for entry in latest.rglob("*"):
            if entry.is_symlink() and not entry.exists():
                return True
    except OSError:
        return False
    return False


def iter_repo_cache_dirs(repo_type: str, repo_id: str) -> Iterator[Path]:
    target = _target_dir_name(repo_type, repo_id)
    for root in _hf_cache_roots():
        try:
            for entry in root.iterdir():
                if entry.name.lower() == target:
                    yield entry
        except OSError:
            continue


def iter_active_repo_cache_dirs(repo_type: str, repo_id: str) -> Iterator[Path]:
    root = _hf_cache_root()
    if root is None:
        return
    target = _target_dir_name(repo_type, repo_id)
    try:
        for entry in root.iterdir():
            if entry.name.lower() == target:
                yield entry
    except OSError:
        return


def preferred_repo_cache_dirs(
    repo_type: str,
    repo_id: str,
    *,
    force_active: bool = False,
) -> list[Path]:
    active_entries = list(iter_active_repo_cache_dirs(repo_type, repo_id))
    if active_entries:
        return active_entries
    if force_active:
        root = _hf_cache_root()
        if root is not None:
            canonical = f"{repo_type}s--{repo_id.replace('/', '--')}"
            return [root / canonical]
    return list(iter_repo_cache_dirs(repo_type, repo_id))


def select_best_cache_progress(
    readings: Iterable[CacheProgressReading],
) -> Optional[CacheProgressReading]:
    return max(
        readings,
        key = lambda item: (item[0] + item[1], item[0]),
        default = None,
    )


def _repo_incomplete_blob_progress_marker(
    repo_type: str, repo_id: str
) -> Optional[tuple[int, int, int]]:
    count = 0
    total = 0
    latest_mtime_ns = 0
    for entry in preferred_repo_cache_dirs(repo_type, repo_id, force_active = True):
        blobs_dir = entry / "blobs"
        if not blobs_dir.is_dir():
            continue
        try:
            blob_entries = list(blobs_dir.iterdir())
        except OSError:
            continue
        for blob in blob_entries:
            if not blob.name.endswith(_INCOMPLETE_SUFFIX):
                continue
            try:
                blob_stat = blob.stat()
            except OSError:
                continue
            if not stat_module.S_ISREG(blob_stat.st_mode):
                continue
            count += 1
            total += blob_stat.st_size
            latest_mtime_ns = max(latest_mtime_ns, blob_stat.st_mtime_ns)
    if count == 0:
        return None
    return count, total, latest_mtime_ns


def has_incomplete_blobs(repo_type: str, repo_id: str) -> bool:
    for entry in iter_repo_cache_dirs(repo_type, repo_id):
        if repo_cache_dir_has_incomplete_blobs(entry):
            return True
    return False


def has_active_incomplete_blobs(repo_type: str, repo_id: str) -> bool:
    for entry in iter_active_repo_cache_dirs(repo_type, repo_id):
        if repo_cache_dir_has_incomplete_blobs(entry):
            return True
    return False


def repo_cache_dir_has_incomplete_blobs(repo_dir: Path) -> bool:
    blobs_dir = repo_dir / "blobs"
    return (blobs_dir.is_dir() and _blob_dir_is_partial(blobs_dir)) or (
        _repo_dir_has_broken_snapshot_symlinks(repo_dir)
    )


def partial_repo_ids(repo_type: str, repo_ids: Iterable[str]) -> set[str]:
    wanted = {_target_dir_name(repo_type, r): r for r in repo_ids}
    if not wanted:
        return set()
    partial: set[str] = set()
    for root in _hf_cache_roots():
        try:
            for entry in root.iterdir():
                repo_id = wanted.get(entry.name.lower())
                if repo_id is None or repo_id in partial:
                    continue
                blobs_dir = entry / "blobs"
                if blobs_dir.is_dir() and _blob_dir_is_partial(blobs_dir):
                    partial.add(repo_id)
        except OSError:
            continue
    return partial


def _is_marker(path: Path) -> bool:
    return path.name in (_MARKER_NAME, f"{_MARKER_NAME}.tmp")


def purge_partial_repo(repo_type: str, repo_id: str) -> bool:
    import shutil

    removed = False
    for entry in iter_repo_cache_dirs(repo_type, repo_id):
        try:
            blobs_dir = entry / "blobs"
            if blobs_dir.is_dir():
                for blob in blobs_dir.iterdir():
                    if blob.is_file() and blob.name.endswith(_INCOMPLETE_SUFFIX):
                        try:
                            blob.unlink()
                            removed = True
                        except OSError:
                            pass
            if not any(
                p.is_file() and not _is_marker(p) for p in entry.rglob("*")
            ):
                shutil.rmtree(entry, ignore_errors = True)
                removed = True
        except OSError:
            pass
    return removed


def purge_repo_cache_dirs(repo_type: str, repo_id: str) -> bool:
    import shutil

    removed = False
    for entry in iter_repo_cache_dirs(repo_type, repo_id):
        try:
            if not entry.is_dir():
                continue
            shutil.rmtree(entry, ignore_errors = True)
            if not entry.exists():
                removed = True
        except OSError:
            pass
    return removed


def spawn_worker(
    args: list[str],
    hf_token: Optional[str],
    cwd: Path,
    use_xet: bool = False,
) -> subprocess.Popen:
    """Spawn the download worker.

    Both the XET client and the ``hf_transfer`` Rust extension write file
    chunks out of order, which makes their partial blobs unsafe to resume
    under a sequential writer (see module docstring). The HTTP path stays
    on the built-in sequential downloader so SIGKILL → resume produces a
    byte-identical final file.
    """
    mode = TRANSPORT_XET if use_xet else TRANSPORT_HTTP
    env = os.environ.copy()
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["HF_HUB_DISABLE_XET"] = "0" if use_xet else "1"
    # hf_transfer writes parallel HTTP Range chunks. Even within "http"
    # mode it can leave sparse partials. Disable unconditionally so that
    # the writer used by the worker is always single-stream sequential
    # when transport=http.
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    if hf_token:
        env["HF_TOKEN"] = hf_token
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{cwd}{os.pathsep}{existing_path}" if existing_path else str(cwd)
    )
    return subprocess.Popen(
        [sys.executable, "-m", "workers.hf_download", *args, "--transport", mode],
        env = env,
        cwd = str(cwd),
        stdout = subprocess.DEVNULL,
        stderr = subprocess.PIPE,
        start_new_session = sys.platform != "win32",
    )


def _purge_incomplete_blobs(entry: Path) -> int:
    """Delete every ``*.incomplete`` blob beneath *entry*. Returns the
    count removed. Per-file failures are swallowed and counted as zero —
    a stuck partial shouldn't block the new download from starting; the
    downloader will detect the existing file and error out clearly if it
    really is unreadable."""
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
            if not blob.name.endswith(_INCOMPLETE_SUFFIX):
                continue
            blob.unlink()
            removed += 1
        except OSError:
            # Swallow: e.g. permission denied on a stale partial. The
            # downstream snapshot_download call will surface a precise
            # error message if it actually can't proceed.
            continue
    return removed


def _read_marker(entry: Path) -> Optional[str]:
    marker = entry / _MARKER_NAME
    try:
        if not marker.exists():
            return None
        value = marker.read_text().strip()
    except OSError:
        return None
    return value if value in _VALID_TRANSPORTS else None


def _write_marker(entry: Path, mode: str) -> None:
    marker = entry / _MARKER_NAME
    try:
        # tmp + rename so a SIGKILL mid-write can never leave a
        # half-written marker that confuses the next run.
        tmp = entry / f"{_MARKER_NAME}.tmp"
        tmp.write_text(mode)
        os.replace(tmp, marker)
    except OSError:
        # Best-effort: missing marker on the next run will purge the
        # partial defensively, which is the safe failure mode.
        pass


def prepare_cache_for_transport(repo_type: str, repo_id: str, mode: str) -> int:
    """Guarantee any pre-existing ``.incomplete`` blobs are SAFE to resume
    under *mode*. Returns the number of partial blobs that were purged
    because their provenance couldn't be trusted.

    The contract:
    - HTTP mode: a partial is trusted ONLY when its repo directory
      carries a ``.transport`` marker whose value equals ``"http"``. Any
      other case — missing marker, unreadable marker, mismatched marker
      — triggers a purge.
    - XET mode: partials are always purged. ``hf_xet.download_files``
      has no resume-offset API and rewrites the destination from scratch
      on every call, so the prior ``.incomplete`` bytes are never
      appended to. Network resume for XET happens transparently through
      the chunk-cache at ``~/.cache/huggingface/xet/chunk-cache`` (which
      lives outside the repo cache dir and is left untouched), so
      removing the stale partial only fixes UI accounting — the actual
      bytes already in CAS are still reused.

    Scope: only the active ``HF_HUB_CACHE`` root is inspected, unlike the
    all-roots status/listing helpers. That is sufficient for resume
    safety because ``snapshot_download`` runs without a ``cache_dir``
    override and so can only ever read or resume a ``.incomplete`` blob
    under this same active root; a stale partial in a legacy/default root
    is never handed to the resumer. The marker is written for the new
    mode before returning.
    """
    if mode not in _VALID_TRANSPORTS:
        raise ValueError(f"Invalid transport mode: {mode!r}")
    root = _hf_cache_root(create = True)
    if root is None:
        return 0
    target = _target_dir_name(repo_type, repo_id)
    try:
        entries = [e for e in root.iterdir() if e.name.lower() == target]
    except OSError:
        return 0
    if not entries:
        # First download: pre-create the repo dir so the marker is on disk
        # before the worker writes any bytes. Without this, a SIGKILL mid-
        # download leaves a partial with no marker, and the resume purges it.
        canonical = f"{repo_type}s--{repo_id.replace('/', '--')}"
        new_entry = root / canonical
        try:
            new_entry.mkdir(exist_ok = True)
        except OSError:
            return 0
        entries = [new_entry]
    total_purged = 0
    for entry in entries:
        if mode == TRANSPORT_XET:
            total_purged += _purge_incomplete_blobs(entry)
        else:
            last_mode = _read_marker(entry)
            if last_mode != mode:
                total_purged += _purge_incomplete_blobs(entry)
        _write_marker(entry, mode)
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


def finalize_worker_exit(
    registry: DownloadRegistry,
    key: str,
    proc: subprocess.Popen,
    *,
    hf_token: Optional[str],
    label: str,
    log_prefix: str,
    logger,
    repo_type: Optional[str] = None,
    repo_id: Optional[str] = None,
) -> None:
    """Block until *proc* exits, then record the job's terminal state in
    *registry*. Drains stderr first so the pipe never fills, scrubs secrets
    from it, and classifies the exit code. A no-op when the process was
    already dropped (e.g. superseded by a newer job for the same key)."""
    stop_watchdog = threading.Event()
    watchdog_error: list[str] = []

    def _watch_for_stall() -> None:
        if not repo_type or not repo_id:
            return
        timeout = max(1.0, _DOWNLOAD_STALL_TIMEOUT_SECONDS)
        poll = max(1.0, min(_DOWNLOAD_STALL_POLL_SECONDS, timeout))
        grace = max(0.0, _DOWNLOAD_STALL_STARTUP_GRACE_SECONDS)
        if grace > 0 and stop_watchdog.wait(grace):
            return
        if proc.poll() is not None or registry.cancel_requested(key):
            return
        last_marker = _repo_incomplete_blob_progress_marker(repo_type, repo_id)
        last_progress_at = time.monotonic()
        while not stop_watchdog.wait(poll):
            if proc.poll() is not None or registry.cancel_requested(key):
                return
            current_marker = _repo_incomplete_blob_progress_marker(repo_type, repo_id)
            now = time.monotonic()
            if current_marker is None:
                last_marker = None
                last_progress_at = now
                continue
            if current_marker != last_marker:
                last_marker = current_marker
                last_progress_at = now
                continue
            stalled_for = now - last_progress_at
            if stalled_for < timeout:
                continue
            message = (
                f"{log_prefix} stalled for {int(stalled_for)}s with no cache "
                f"progress. Check the network connection and restart the download."
            )
            watchdog_error.append(message)
            logger.warning(f"{message}: {label}")
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            except Exception as exc:
                logger.warning(
                    f"{log_prefix} watchdog failed to kill {label}: {exc}"
                )
            return

    watchdog_thread: Optional[threading.Thread] = None
    if repo_type and repo_id:
        watchdog_thread = threading.Thread(
            target = _watch_for_stall,
            name = f"hf-download-stall-watch-{key}",
            daemon = True,
        )
        watchdog_thread.start()

    stderr_data = drain_stderr_excerpt(proc.stderr)
    rc = proc.wait()
    stop_watchdog.set()
    if watchdog_thread is not None:
        watchdog_thread.join(timeout = 1)
    if not registry.drop_process(key, proc):
        return
    stderr_text = scrub_secrets(
        (stderr_data or b"").decode("utf-8", "replace").strip(),
        hf_token = hf_token,
    )
    state = classify_exit(rc, cancel_requested = registry.cancel_requested(key))
    if state == "complete":
        registry.set_job(key, "complete")
        logger.info(f"{log_prefix} complete: {label}")
    elif state == "cancelled":
        registry.set_job(key, "cancelled")
        logger.info(f"{log_prefix} cancelled by user: {label} (rc={rc})")
    else:
        error_text = watchdog_error[-1] if watchdog_error else stderr_text
        registry.set_job(
            key, "error", error_text or f"worker exited with code {rc}",
        )
        logger.error(
            f"{log_prefix} failed for {label} (rc={rc}): {error_text}",
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


def purge_empty_marker_dir(repo_type: str, repo_id: str) -> bool:
    """Remove a repo cache dir that only contains the ``.transport`` marker.

    ``prepare_cache_for_transport`` pre-creates the dir + marker before the
    worker downloads anything; a failure during validation/auth/network
    setup leaves the dir as marker-only litter that survives until an
    explicit delete. This helper is safe to call after any failed
    download: a dir holding ``blobs/``/``snapshots/``/``refs/`` (i.e. real
    or in-progress content) won't match and is left untouched, so an
    interrupted-but-resumable partial isn't blown away.
    """
    cleaned = False
    for entry in iter_repo_cache_dirs(repo_type, repo_id):
        try:
            contents = list(entry.iterdir())
        except OSError:
            continue
        if len(contents) != 1 or contents[0].name != _MARKER_NAME:
            continue
        try:
            contents[0].unlink()
            entry.rmdir()
            cleaned = True
        except OSError:
            continue
    return cleaned


def read_transport_marker(repo_type: str, repo_id: str) -> Optional[str]:
    for entry in iter_repo_cache_dirs(repo_type, repo_id):
        value = _read_marker(entry)
        if value is not None:
            return value
    return None


def read_active_transport_marker(repo_type: str, repo_id: str) -> Optional[str]:
    for entry in iter_active_repo_cache_dirs(repo_type, repo_id):
        value = _read_marker(entry)
        if value is not None:
            return value
    return None


def is_resumable_partial(repo_type: str, repo_id: str) -> bool:
    """True only when a partial exists AND was produced by a writer that
    supports byte-level resume — i.e. the HTTP transport. XET partials
    exist on disk but are not resumable; they will be discarded on the
    next download attempt."""
    if not has_active_incomplete_blobs(repo_type, repo_id):
        return False
    return read_active_transport_marker(repo_type, repo_id) == TRANSPORT_HTTP


def incomplete_blob_hashes(repo_type: str, repo_id: str) -> set[str]:
    out: set[str] = set()
    for entry in iter_repo_cache_dirs(repo_type, repo_id):
        blobs_dir = entry / "blobs"
        if not blobs_dir.is_dir():
            continue
        try:
            for blob in blobs_dir.iterdir():
                if blob.is_file() and blob.name.endswith(_INCOMPLETE_SUFFIX):
                    out.add(blob.name[: -len(_INCOMPLETE_SUFFIX)])
        except OSError:
            continue
    return out


TERMINAL_STATES = frozenset({"complete", "cancelled", "error"})
_ACTIVE_STATES = frozenset({"running", "cancelling"})


@dataclass(frozen = True)
class DownloadState:
    state: str
    error: Optional[str] = None


def _repo_of_key(key: str) -> str:
    return key.split("::", 1)[0]


class DownloadRegistry:
    """Thread-safe state machine for background HF download jobs.

    One instance backs model downloads (keys ``repo_id::variant``) and
    another backs dataset downloads (keys ``repo_id``). ``_repo_of_key``
    groups keys by repo so only one writer can touch a repo cache at a time,
    and the no-variant dataset case (one key per repo) falls through that
    check unchanged.
    """

    def __init__(self, max_terminal: int = 64) -> None:
        self._jobs: dict[str, DownloadState] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._repo_active: dict[str, dict[str, str]] = {}
        self._pending_cancel: dict[str, Optional[int]] = {}
        self._generations: dict[str, int] = {}
        self._deleting: set[str] = set()
        self._lock = threading.Lock()
        self._max_terminal = max_terminal

    def set_job(self, key: str, state: str, error: Optional[str] = None) -> None:
        with self._lock:
            self._jobs[key] = DownloadState(state, error)
            if state in TERMINAL_STATES:
                self._pending_cancel.pop(key, None)
                repo = _repo_of_key(key)
                active = self._repo_active.get(repo)
                if active is not None:
                    active.pop(key, None)
                    if not active:
                        self._repo_active.pop(repo, None)
                if len(self._jobs) > self._max_terminal:
                    for stale_key, stale in list(self._jobs.items()):
                        if stale.state in TERMINAL_STATES and stale_key != key:
                            self._jobs.pop(stale_key, None)
                            if len(self._jobs) <= self._max_terminal:
                                break

    def get_job(self, key: str) -> DownloadState:
        with self._lock:
            return self._jobs.get(key, DownloadState("idle"))

    def current_generation(self, key: str) -> int:
        with self._lock:
            return self._generations.get(key, 0)

    def _generation_matches_locked(
        self,
        key: str,
        generation: Optional[int],
    ) -> bool:
        return generation is None or self._generations.get(key, 0) == generation

    def register_process(self, key: str, proc: subprocess.Popen) -> bool:
        """Register *proc* for *key*. Returns ``False`` when a cancel was
        requested during the claim→register window (the caller must kill
        *proc* immediately); ``True`` otherwise."""
        with self._lock:
            has_pending_cancel = key in self._pending_cancel
            pending_generation = self._pending_cancel.pop(key, None)
            if has_pending_cancel and self._generation_matches_locked(
                key,
                pending_generation,
            ):
                self._jobs[key] = DownloadState("cancelled")
                repo = _repo_of_key(key)
                active = self._repo_active.get(repo)
                if active is not None:
                    active.pop(key, None)
                    if not active:
                        self._repo_active.pop(repo, None)
                return False
            self._processes[key] = proc
            return True

    def mark_pending_cancel(
        self,
        key: str,
        generation: Optional[int] = None,
    ) -> bool:
        """Record a cancel for an active job whose worker process hasn't
        registered yet. Returns ``True`` when the pending cancel was armed,
        so :func:`register_process` will kill the process on arrival."""
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
        with self._lock:
            if key in self._pending_cancel:
                return True
            return self._jobs.get(key, DownloadState("idle")).state == "cancelling"

    def get_process(self, key: str) -> Optional[subprocess.Popen]:
        with self._lock:
            return self._processes.get(key)

    def drop_process(self, key: str, proc: subprocess.Popen) -> bool:
        with self._lock:
            if self._processes.get(key) is not proc:
                return False
            self._processes.pop(key, None)
            return True

    def claim(self, key: str, transport: str) -> tuple[bool, str]:
        repo = _repo_of_key(key)
        with self._lock:
            if repo in self._deleting:
                return False, "deleting"
            active = self._repo_active.get(repo, {})
            stale_keys: list[str] = []
            conflict_state: Optional[str] = None
            for other_key in active:
                if other_key == key:
                    continue
                other_status = self._jobs.get(other_key)
                if other_status is None or other_status.state not in _ACTIVE_STATES:
                    stale_keys.append(other_key)
                    continue
                conflict_state = other_status.state
                break
            for stale_key in stale_keys:
                active.pop(stale_key, None)
            if conflict_state is not None:
                return False, conflict_state
            current = self._jobs.get(key, DownloadState("idle")).state
            if current in _ACTIVE_STATES:
                return False, current
            self._generations[key] = self._generations.get(key, 0) + 1
            self._jobs[key] = DownloadState("running")
            self._repo_active.setdefault(repo, active)[key] = transport
            return True, "running"

    def adoptable(self, key: str) -> bool:
        """True when *key* itself has a live job a client can attach to.

        Lets a rejected claim distinguish a collision with this key's own
        in-flight job (pollable) from one blocked by a different repo job
        or an in-progress delete, where no job exists for this key."""
        with self._lock:
            return self._jobs.get(key, DownloadState("idle")).state in _ACTIVE_STATES

    def has_active(self, repo_id: str) -> bool:
        with self._lock:
            return self._has_active_locked(repo_id)

    def _has_active_locked(self, repo_id: str) -> bool:
        active = self._repo_active.get(repo_id, {})
        for key in active:
            job = self._jobs.get(key)
            if job is not None and job.state in _ACTIVE_STATES:
                return True
        return False

    def active_jobs(self, repo_id: str) -> dict[str, str]:
        """Map of every active job key for *repo_id* to its state."""
        with self._lock:
            result: dict[str, str] = {}
            for key in self._repo_active.get(repo_id, {}):
                job = self._jobs.get(key)
                if job is not None and job.state in _ACTIVE_STATES:
                    result[key] = job.state
            return result

    def begin_delete(self, repo_id: str) -> bool:
        """Reserve *repo_id* for deletion. Returns ``False`` when a download
        is active for the repo (the caller must refuse the delete). On
        success the repo is marked so :func:`claim` rejects new downloads
        until :func:`end_delete` runs, closing the check-then-delete race
        against a concurrently spawned worker."""
        with self._lock:
            if self._has_active_locked(repo_id):
                return False
            self._deleting.add(repo_id)
            return True

    def end_delete(self, repo_id: str) -> None:
        with self._lock:
            self._deleting.discard(repo_id)

    def request_cancel(
        self,
        key: str,
        proc: subprocess.Popen,
        generation: Optional[int] = None,
    ) -> bool:
        """Authorize a SIGKILL for the registered *proc*. Idempotent across an
        active job's lifetime: a repeated cancel while already ``cancelling``
        still returns ``True`` so a kill that raced and lost can be re-sent."""
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
                (key, proc)
                for key, proc in self._processes.items()
                if proc.poll() is None
            ]
            # Flag these as an intentional stop so the per-worker watcher's
            # exit classification reports them as cancelled rather than as an
            # OOM/crash error once the SIGKILL lands.
            for key, _proc in live:
                if self._jobs.get(key, DownloadState("idle")).state == "running":
                    self._jobs[key] = DownloadState("cancelling")
        for key, proc in live:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            except Exception as e:
                logger.warning(f"shutdown: failed to kill {kind} worker for {key}: {e}")
                continue
            # Reap synchronously: the per-worker watcher that calls wait() is a
            # daemon thread that may not run at interpreter shutdown, so without
            # this the killed child leaks as a zombie holding its stdio FDs.
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"shutdown: {kind} worker for {key} did not exit after kill"
                )
            except Exception:
                pass
