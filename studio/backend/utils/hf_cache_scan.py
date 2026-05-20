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

import os
import re
import signal
import subprocess
import sys
import threading
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


def iter_repo_cache_dirs(repo_type: str, repo_id: str) -> Iterator[Path]:
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


def has_incomplete_blobs(repo_type: str, repo_id: str) -> bool:
    for entry in iter_repo_cache_dirs(repo_type, repo_id):
        blobs_dir = entry / "blobs"
        if blobs_dir.is_dir() and _blob_dir_is_partial(blobs_dir):
            return True
    return False


def partial_repo_ids(repo_type: str, repo_ids: Iterable[str]) -> set[str]:
    wanted = {_target_dir_name(repo_type, r): r for r in repo_ids}
    if not wanted:
        return set()
    root = _hf_cache_root()
    if root is None:
        return set()
    partial: set[str] = set()
    try:
        for entry in root.iterdir():
            repo_id = wanted.get(entry.name.lower())
            if repo_id is None or repo_id in partial:
                continue
            blobs_dir = entry / "blobs"
            if blobs_dir.is_dir() and _blob_dir_is_partial(blobs_dir):
                partial.add(repo_id)
    except OSError:
        pass
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

    The marker is written for the new mode before returning.
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


def classify_exit(rc: int) -> str:
    """Map a worker process exit code to a job state.

    - rc == 0: clean completion.
    - rc == EXIT_CANCELLED (130), or rc killed by a cancellation signal
      (SIGKILL/SIGTERM/SIGINT): cancelled by the user or shutdown.
    - any other non-zero rc, including crash signals (SIGSEGV, SIGABRT):
      worker errored out.
    """
    if rc == 0:
        return "complete"
    if rc == EXIT_CANCELLED or rc in _CANCELLATION_RETURN_CODES:
        return "cancelled"
    return "error"


def drain_stderr_tail(stream, max_bytes: int = 8192) -> bytes:
    """Drain a worker's stderr to EOF, retaining only the trailing bytes.

    Reading incrementally keeps the pipe from filling while bounding memory:
    only the tail is surfaced in error reports, so the rest is discarded."""
    if stream is None:
        return b""
    tail = bytearray()
    for chunk in iter(lambda: stream.read(4096), b""):
        tail.extend(chunk)
        if len(tail) > max_bytes:
            del tail[:-max_bytes]
    return bytes(tail)


def finalize_worker_exit(
    registry: DownloadRegistry,
    key: str,
    proc: subprocess.Popen,
    *,
    hf_token: Optional[str],
    label: str,
    log_prefix: str,
    logger,
) -> None:
    """Block until *proc* exits, then record the job's terminal state in
    *registry*. Drains stderr first so the pipe never fills, scrubs secrets
    from it, and classifies the exit code. A no-op when the process was
    already dropped (e.g. superseded by a newer job for the same key)."""
    stderr_data = drain_stderr_tail(proc.stderr)
    rc = proc.wait()
    if not registry.drop_process(key, proc):
        return
    stderr_text = scrub_secrets(
        (stderr_data or b"").decode("utf-8", "replace").strip(),
        hf_token = hf_token,
    )
    state = classify_exit(rc)
    if state == "complete":
        registry.set_job(key, "complete")
        logger.info(f"{log_prefix} complete: {label}")
    elif state == "cancelled":
        registry.set_job(key, "cancelled")
        logger.info(f"{log_prefix} cancelled by user: {label} (rc={rc})")
    else:
        registry.set_job(
            key, "error", stderr_text[-500:] or f"worker exited with code {rc}",
        )
        logger.error(
            f"{log_prefix} failed for {label} (rc={rc}): {stderr_text[-500:]}",
        )


def read_transport_marker(repo_type: str, repo_id: str) -> Optional[str]:
    for entry in iter_repo_cache_dirs(repo_type, repo_id):
        value = _read_marker(entry)
        if value is not None:
            return value
    return None


def is_resumable_partial(repo_type: str, repo_id: str) -> bool:
    """True only when a partial exists AND was produced by a writer that
    supports byte-level resume — i.e. the HTTP transport. XET partials
    exist on disk but are not resumable; they will be discarded on the
    next download attempt."""
    if not has_incomplete_blobs(repo_type, repo_id):
        return False
    return read_transport_marker(repo_type, repo_id) == TRANSPORT_HTTP


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
    groups keys by repo so concurrent variants of one repo cannot run
    under conflicting transports, and the no-variant dataset case (one key
    per repo) falls through that check unchanged.
    """

    def __init__(self, max_terminal: int = 64) -> None:
        self._jobs: dict[str, DownloadState] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._repo_active: dict[str, dict[str, str]] = {}
        self._pending_cancel: set[str] = set()
        self._deleting: set[str] = set()
        self._lock = threading.Lock()
        self._max_terminal = max_terminal

    def set_job(self, key: str, state: str, error: Optional[str] = None) -> None:
        with self._lock:
            self._jobs[key] = DownloadState(state, error)
            if state in TERMINAL_STATES:
                self._pending_cancel.discard(key)
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

    def register_process(self, key: str, proc: subprocess.Popen) -> bool:
        """Register *proc* for *key*. Returns ``False`` when a cancel was
        requested during the claim→register window (the caller must kill
        *proc* immediately); ``True`` otherwise."""
        with self._lock:
            self._processes[key] = proc
            if key in self._pending_cancel:
                self._pending_cancel.discard(key)
                return False
            return True

    def mark_pending_cancel(self, key: str) -> bool:
        """Record a cancel for an active job whose worker process hasn't
        registered yet. Returns ``True`` when the pending cancel was armed,
        so :func:`register_process` will kill the process on arrival."""
        with self._lock:
            if self._jobs.get(key, DownloadState("idle")).state not in _ACTIVE_STATES:
                return False
            self._pending_cancel.add(key)
            self._jobs[key] = DownloadState("cancelling")
            return True

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
            for other_key, other_transport in active.items():
                if other_key == key:
                    continue
                other_status = self._jobs.get(other_key)
                if other_status is None or other_status.state not in _ACTIVE_STATES:
                    stale_keys.append(other_key)
                    continue
                if other_transport != transport:
                    conflict_state = other_status.state
                    break
            for stale_key in stale_keys:
                active.pop(stale_key, None)
            if conflict_state is not None:
                return False, conflict_state
            current = self._jobs.get(key, DownloadState("idle")).state
            if current in _ACTIVE_STATES:
                return False, current
            self._jobs[key] = DownloadState("running")
            self._repo_active.setdefault(repo, active)[key] = transport
            return True, "running"

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

    def request_cancel(self, key: str, proc: subprocess.Popen) -> bool:
        with self._lock:
            if self._processes.get(key) is not proc:
                return False
            if self._jobs.get(key, DownloadState("idle")).state != "running":
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
        for key, proc in live:
            try:
                proc.kill()
            except ProcessLookupError:
                continue
            except Exception as e:
                logger.warning(f"shutdown: failed to kill {kind} worker for {key}: {e}")
