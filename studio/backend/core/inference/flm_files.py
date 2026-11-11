# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resumable FastFlowLM downloads into the folder it reads; renamed from .partial once hash matches"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Optional

import httpx

logger = logging.getLogger(__name__)

_CHUNK = 1 << 20
_MAX_STALLED_ATTEMPTS = 5
_RETRY_DELAY_SECONDS = 2.0
_RETRIED_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_HF_REPO_URL = re.compile(
    r"https://huggingface\.co/(?P<repo>[\w.-]+/[\w.-]+)(?:/resolve/(?P<revision>[^/?#]+))?/?$"
)
# Past the PID text: a Windows byte lock also blocks reads, and a waiter reads who holds it.
_LOCK_OFFSET = 64


class FlmDownloadError(RuntimeError):
    """A file could not be downloaded; the message is safe to show the user."""


class FlmDownloadCancelled(FlmDownloadError):
    """Stopped on request; partial files are kept so the next download resumes."""


@dataclass(frozen = True)
class FlmFile:
    name: str
    url: str
    size: int
    # sha256 of the content for LFS files, git's blob sha1 otherwise: what FastFlowLM checks.
    digest: str
    lfs: bool


@dataclass(frozen = True)
class FlmModelFiles:
    folder: Path
    files: tuple[FlmFile, ...]

    @property
    def total(self) -> int:
        return sum(file.size for file in self.files)

    def on_disk(self) -> int:
        done = 0
        for file in self.files:
            for path in (self.folder / file.name, _partial(self.folder / file.name)):
                try:
                    done += min(path.stat().st_size, file.size)
                    break
                except OSError:
                    continue
        return done

    def resume_percent(self) -> Optional[int]:
        """How much of the model an interrupted download left on disk, or None if nothing."""
        done = self.on_disk()
        return done * 100 // self.total if done and self.total else None


def _partial(path: Path) -> Path:
    return path.with_name(path.name + ".partial")


@lru_cache(maxsize = 4)
def _parsed(path: Path, stamp: tuple[int, int]) -> Any:
    return json.loads(path.read_text(encoding = "utf-8"))


def _load(path: Path) -> Any:
    """Parsed once per version of the file: the catalog reads it for every model it lists."""
    info = path.stat()
    return _parsed(path, (info.st_mtime_ns, info.st_size))


def read_model_repo(
    flm_binary: Optional[Path], checkpoint: str
) -> tuple[Optional[str], Optional[str]]:
    """(owner/name, revision or None) of the Hugging Face repo model_list.json names for
    checkpoint; (None, None) when unknown or not on huggingface.co."""
    if flm_binary is None or ":" not in checkpoint:
        return None, None
    family, size = checkpoint.split(":", 1)
    try:
        url = _load(flm_binary.parent / "model_list.json")["models"][family][size]["url"]
    except (OSError, ValueError, KeyError, TypeError):
        return None, None
    match = _HF_REPO_URL.match(url) if isinstance(url, str) else None
    if match is None:
        return None, None
    return match["repo"], match["revision"]


def read_model_files(
    flm_binary: Optional[Path], flm_model_path: Path, checkpoint: str
) -> Optional[FlmModelFiles]:
    """The files FastFlowLM expects for checkpoint (family:size), read from the manifests beside its
    binary; None when unreadable, so the caller falls back to lemond."""
    if flm_binary is None or ":" not in checkpoint:
        return None
    family, size = checkpoint.split(":", 1)
    try:
        listing = _load(flm_binary.parent / "model_list.json")
        hashes = _load(flm_binary.parent / "model_info.json")
        entry = listing["models"][family][size]
        known = {row["path"]: row for row in hashes[checkpoint]}
        base = entry["url"].rstrip("/")
        # FastFlowLM's own rule: a url naming a revision already ends in /resolve/<revision>.
        if "/resolve" not in base:
            base += "/resolve/main"
        files = []
        for name in entry["files"]:
            row = known.get(name)
            if row is None:  # FastFlowLM skips a file its hash list does not name.
                continue
            lfs = isinstance(row.get("lfs"), dict)
            files.append(
                FlmFile(
                    name = name,
                    url = f"{base}/{name}?download=true",
                    size = int(row["size"]),
                    digest = row["lfs"]["oid"] if lfs else row["oid"],
                    lfs = lfs,
                )
            )
        folder = flm_model_path / listing["model_path"] / entry["name"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("Could not read FastFlowLM's file list for %s: %s", checkpoint, exc)
        return None
    if not files:
        return None
    return FlmModelFiles(folder = folder, files = tuple(files))


def model_folder_checkpoints(flm_binary: Optional[Path]) -> tuple[Optional[str], dict[str, str]]:
    """FastFlowLM's models subfolder and each model folder's checkpoint, from its manifest."""
    if flm_binary is None:
        return None, {}
    try:
        listing = _load(flm_binary.parent / "model_list.json")
        names = {
            str(entry["name"]): f"{family}:{size}"
            for family, sizes in listing["models"].items()
            for size, entry in sizes.items()
        }
        return str(listing["model_path"]), names
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        logger.debug("Could not read FastFlowLM's model list: %s", exc)
        return None, {}


def _hasher(file: FlmFile):
    if file.lfs:
        return hashlib.sha256()
    digest = hashlib.sha1()
    digest.update(f"blob {file.size}\0".encode())
    return digest


def _check(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise FlmDownloadCancelled("The download was cancelled.")


def _matches(path: Path, file: FlmFile, cancel: threading.Event) -> bool:
    digest = _hasher(file)
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            _check(cancel)
            digest.update(chunk)
    return digest.hexdigest() == file.digest


def _try_lock(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt
        handle.seek(_LOCK_OFFSET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _acquire_lock(model: FlmModelFiles) -> BinaryIO:
    """One downloader per model across processes on this home. An OS lock, so a process that dies
    releases it and no stale-lock cleanup can race a live owner."""
    # Beside the folder, not in it: FastFlowLM owns the folder's contents.
    # Unbuffered: a buffered read would reach the locked byte.
    path = model.folder.with_name(model.folder.name + ".unsloth-download.lock")
    handle = open(path, "a+b", buffering = 0)
    try:
        _try_lock(handle)
    except OSError:
        handle.seek(0)
        owner = handle.read(32).decode("ascii", "ignore").strip()
        handle.close()
        owner = f" (PID {owner})" if owner.isdigit() else ""
        raise FlmDownloadError(
            f"This model is already downloading in another Unsloth Studio process{owner}."
        ) from None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()).encode("ascii"))
    handle.flush()
    return handle


def _gb(size: int) -> str:
    return f"{size / 1e9:.1f} GB"


def _check_free_space(model: FlmModelFiles) -> None:
    needed = model.total - model.on_disk()
    free = shutil.disk_usage(model.folder).free
    if needed > free:
        raise FlmDownloadError(
            f"Not enough disk space: this model needs {_gb(needed)} more and only "
            f"{_gb(free)} is free in {model.folder.parent}."
        )


def download_files(
    model: FlmModelFiles,
    client: Optional[httpx.Client] = None,
    cancel: Optional[threading.Event] = None,
) -> Iterator[dict[str, Any]]:
    """Raises FlmDownloadError on a stall, bad hash, full disk or cancel; partial files are kept
    so the next call resumes."""
    cancel = cancel or threading.Event()
    model.folder.mkdir(parents = True, exist_ok = True)
    lock = _acquire_lock(model)
    owned = client is None
    total = model.total
    completed = 0  # bytes of the files already in place
    # One event per (file, percent): followers replay every event of a pull that can take hours.
    reported = None
    file = None
    try:
        _check_free_space(model)
        client = client or httpx.Client(
            follow_redirects = True, timeout = httpx.Timeout(30.0, read = 120.0)
        )
        for file in model.files:
            final = model.folder / file.name
            partial = _partial(final)
            try:
                have = final.stat().st_size
            except FileNotFoundError:
                have = None
            # Rehashing what is already on disk can take minutes on a 20 GB model.
            verifying = {
                "event": "verifying",
                "file": file.name,
                "bytes_downloaded": completed,
                "bytes_total": total,
                "percent": min(99, completed * 100 // total),
            }
            if have == file.size:
                yield verifying
            # Same size is not proof: an older release's config.json differs only in its version.
            if have == file.size and _matches(final, file, cancel):
                partial.unlink(missing_ok = True)
                completed += file.size
                continue
            if have == file.size:
                final.unlink()
            elif have is not None:
                # A killed FastFlowLM pull leaves a prefix under the final name.
                if have < file.size:
                    final.replace(partial)
                else:
                    final.unlink()
            if partial.exists():
                yield verifying
            for offset in _download_file(client, file, partial, cancel):
                done = completed + offset
                percent = min(99, done * 100 // total)
                if (file.name, percent) != reported:
                    reported = (file.name, percent)
                    yield {
                        "event": "progress",
                        "file": file.name,
                        "bytes_downloaded": done,
                        "bytes_total": total,
                        "percent": percent,
                    }
            partial.replace(final)
            completed += file.size
    except OSError as exc:
        if exc.errno != errno.ENOSPC:
            raise
        name = file.name if file is not None else "the model"
        raise FlmDownloadError(
            f"Out of disk space while downloading {name}. Free up space and download again; "
            "it resumes where it stopped."
        ) from exc
    finally:
        if owned and client is not None:
            client.close()
        # Left in place: unlinking a held lock lets a newcomer lock a fresh file beside it.
        lock.close()


def _resumed_from(response: httpx.Response) -> Optional[int]:
    """The first byte a 206 answer carries, from ``Content-Range: bytes <start>-<end>/<size>``."""
    match = re.match(r"bytes (\d+)-", response.headers.get("Content-Range", ""))
    return int(match.group(1)) if match else None


def _download_file(
    client: httpx.Client, file: FlmFile, partial: Path, cancel: threading.Event
) -> Iterator[int]:
    """Bring ``partial`` up to the whole file and check its hash, yielding its size as it grows."""
    digest = _hasher(file)
    offset = 0
    try:
        with open(partial, "rb") as handle:
            while chunk := handle.read(_CHUNK):
                _check(cancel)
                digest.update(chunk)
                offset += len(chunk)
    except FileNotFoundError:
        pass
    if offset > file.size:
        partial.unlink()
        digest, offset = _hasher(file), 0
    yield offset
    # Only a byte never reached before is progress, so refetching a prefix still counts as stalled.
    furthest = offset
    stalled = 0
    while offset < file.size:
        _check(cancel)
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with client.stream("GET", file.url, headers = headers) as response:
                status = response.status_code
                if status == 200 and offset:
                    digest, offset = _hasher(file), 0
                    yield offset
                elif status == 206 and _resumed_from(response) != offset:
                    raise httpx.RemoteProtocolError(
                        f"asked for byte {offset}, got {response.headers.get('Content-Range')}"
                    )
                elif status not in (200, 206):
                    if status not in _RETRIED_STATUSES:
                        raise FlmDownloadError(f"Downloading {file.name} failed: HTTP {status}")
                    raise httpx.HTTPStatusError(
                        f"HTTP {status}", request = response.request, response = response
                    )
                with open(partial, "ab") if offset else open(partial, "wb") as handle:
                    for chunk in response.iter_bytes():
                        _check(cancel)
                        handle.write(chunk)
                        digest.update(chunk)
                        offset += len(chunk)
                        yield offset
            if offset < file.size:
                raise httpx.RemoteProtocolError("the connection closed before the file ended")
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            stalled = 0 if offset > furthest else stalled + 1
            furthest = max(furthest, offset)
            if stalled >= _MAX_STALLED_ATTEMPTS:
                raise FlmDownloadError(f"Downloading {file.name} failed: {exc}") from exc
            logger.warning(
                "Downloading %s stopped at %d of %d bytes (%s); resuming",
                file.name,
                offset,
                file.size,
                exc,
            )
            cancel.wait(_RETRY_DELAY_SECONDS * max(stalled, 1))
    if offset != file.size or digest.hexdigest() != file.digest:
        partial.unlink(missing_ok = True)
        raise FlmDownloadError(
            f"{file.name} did not match its published hash; it was removed, so try again."
        )
