# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Download a FastFlowLM model's files with resume, into the folder FastFlowLM reads.

FastFlowLM's own pull cannot resume. It writes each file straight to its final name, starts a
failed file over from zero (three tries), must finish each file within an hour, and takes a file a
killed pull left half-written as complete. Studio downloads the files here instead: each one goes
to ``<name>.partial``, continues from there with a Range request after any interruption, and is
renamed only once its hash matches. lemond's pull then finds every file present and only
registers the model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Optional

import httpx

logger = logging.getLogger(__name__)

_CHUNK = 1 << 20
_MAX_STALLED_ATTEMPTS = 5
_RETRY_DELAY_SECONDS = 2.0
_RETRIED_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class FlmDownloadError(RuntimeError):
    """A file could not be downloaded; the message is safe to show the user."""


@dataclass(frozen = True)
class FlmFile:
    name: str
    url: str
    size: int
    # sha256 for LFS files, git blob sha1 otherwise: what FastFlowLM checks.
    digest: str
    lfs: bool


@dataclass(frozen = True)
class FlmModelFiles:
    folder: Path
    files: tuple[FlmFile, ...]

    @property
    def total(self) -> int:
        return sum(file.size for file in self.files)

    def resume_percent(self) -> Optional[int]:
        """How much of the model an interrupted download left on disk, or None if nothing."""
        done = 0
        for file in self.files:
            for path in (self.folder / file.name, _partial(self.folder / file.name)):
                try:
                    done += min(path.stat().st_size, file.size)
                    break
                except OSError:
                    continue
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


def read_model_files(
    flm_binary: Optional[Path], flm_model_path: Path, checkpoint: str
) -> Optional[FlmModelFiles]:
    """The files FastFlowLM expects for ``checkpoint`` (``family:size``), from the manifests it
    ships next to its binary. None when they cannot be read, so the caller falls back to lemond.
    """
    if flm_binary is None or ":" not in checkpoint:
        return None
    family, size = checkpoint.split(":", 1)
    try:
        listing = _load(flm_binary.parent / "model_list.json")
        hashes = _load(flm_binary.parent / "model_info.json")
        entry = listing["models"][family][size]
        known = {row["path"]: row for row in hashes[checkpoint]}
        base = entry["url"].rstrip("/")
        if "/resolve" not in base:
            base += "/resolve/main"
        files = []
        for name in entry["files"]:
            row = known.get(name)
            if row is None:
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


def _hasher(file: FlmFile):
    if file.lfs:
        return hashlib.sha256()
    digest = hashlib.sha1()
    digest.update(f"blob {file.size}\0".encode())
    return digest


def _matches(path: Path, file: FlmFile) -> bool:
    digest = _hasher(file)
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest() == file.digest


def download_files(
    model: FlmModelFiles, client: Optional[httpx.Client] = None
) -> Iterator[dict[str, Any]]:
    """Download every missing file, yielding progress over the whole model as its percent moves.

    Raises FlmDownloadError when a file stops making progress or fails its hash check. Whatever
    was written stays in ``.partial`` files, so the next call continues from there.
    """
    model.folder.mkdir(parents = True, exist_ok = True)
    owned = client is None
    client = client or httpx.Client(follow_redirects = True, timeout = httpx.Timeout(30.0, read = 120.0))
    total = model.total
    completed = 0
    # One event per (file, percent): followers replay every event.
    reported = None
    try:
        for file in model.files:
            final = model.folder / file.name
            partial = _partial(final)
            try:
                have = final.stat().st_size
            except FileNotFoundError:
                have = None
            # Same size is not proof: an older release's config.json differs only in its version.
            if have == file.size and _matches(final, file):
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
            for offset in _download_file(client, file, partial):
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
    finally:
        if owned:
            client.close()


def _resumed_from(response: httpx.Response) -> Optional[int]:
    """The first byte a 206 answer carries, from ``Content-Range: bytes <start>-<end>/<size>``."""
    match = re.match(r"bytes (\d+)-", response.headers.get("Content-Range", ""))
    return int(match.group(1)) if match else None


def _download_file(client: httpx.Client, file: FlmFile, partial: Path) -> Iterator[int]:
    """Bring ``partial`` up to the whole file and check its hash, yielding its size as it grows."""
    digest = _hasher(file)
    offset = 0
    try:
        with open(partial, "rb") as handle:
            while chunk := handle.read(_CHUNK):
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
            time.sleep(_RETRY_DELAY_SECONDS * max(stalled, 1))
    if offset != file.size or digest.hexdigest() != file.digest:
        partial.unlink(missing_ok = True)
        raise FlmDownloadError(
            f"{file.name} did not match its published hash; it was removed, so try again."
        )
