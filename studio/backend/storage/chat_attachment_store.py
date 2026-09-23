# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-account store for the original bytes of chat attachments.

A blob is named by the sha256 of its bytes, so an upload needs no thread id and an identical file
is kept once. Its mtime is its last upload, and the sweep deletes only a blob older than the grace
window that no message names. One lock covers both, so an upload renews a blob before the sweep.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import BinaryIO

from utils.paths import account_path, ensure_dir

MAX_ATTACHMENT_BYTES = 200 * 1024 * 1024
# A send persists its message seconds after uploading; the window covers cancelled and failed sends.
SWEEP_GRACE_SECONDS = 24 * 60 * 60
# Deleting a chat is the only other thing that sweeps, and a server may run for weeks without one.
SWEEP_INTERVAL_SECONDS = 60 * 60

_ID = re.compile(r"[0-9a-f]{64}")
_PARTIAL_PREFIX = ".upload-"
_lock = threading.Lock()
# Per store, since each account has its own: one account's uploads must not hold off another's sweep.
_swept_at: dict[Path, float] = {}


class AttachmentTooLarge(ValueError):
    pass


class EmptyAttachment(ValueError):
    pass


def _root() -> Path:
    return account_path("chat-attachments")


def store_attachment(source: BinaryIO) -> tuple[str, int]:
    """Store a stream and return ``(id, size)``; storing bytes already present renews them."""
    root = ensure_dir(_root())
    fd, partial = tempfile.mkstemp(dir = root, prefix = _PARTIAL_PREFIX)
    try:
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(fd, "wb") as out:
            while block := source.read(1 << 20):
                size += len(block)
                if size > MAX_ATTACHMENT_BYTES:
                    raise AttachmentTooLarge(
                        f"File exceeds the {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB upload limit."
                    )
                out.write(block)
                digest.update(block)
        if size == 0:
            raise EmptyAttachment("Uploaded file is empty.")
        attachment_id = digest.hexdigest()
        with _lock:
            try:
                os.utime(root / attachment_id)
            except FileNotFoundError:
                os.replace(partial, root / attachment_id)
                partial = None
        return attachment_id, size
    finally:
        if partial is not None:
            Path(partial).unlink(missing_ok = True)


def attachment_path(attachment_id: str) -> Path | None:
    if not isinstance(attachment_id, str) or not _ID.fullmatch(attachment_id):
        return None
    path = _root() / attachment_id
    return path if path.is_file() else None


def sweep_attachments_if_due(now: float | None = None) -> int:
    """Sweep at most once an interval, so uploads abandoned on a server that deletes nothing go."""
    root = _root()
    moment = time.time() if now is None else now
    with _lock:
        if moment - _swept_at.get(root, 0.0) < SWEEP_INTERVAL_SECONDS:
            return 0
        _swept_at[root] = moment
    return sweep_attachments(now)


def sweep_attachments(now: float | None = None) -> int:
    """Delete expired blobs no stored message names, and abandoned partial uploads. Returns the count."""
    from storage.studio_db import chat_attachment_blob_is_referenced

    root = _root()
    if not root.is_dir():
        return 0
    cutoff = (time.time() if now is None else now) - SWEEP_GRACE_SECONDS
    removed = 0
    with _lock:
        for entry in os.scandir(root):
            is_blob = bool(_ID.fullmatch(entry.name))
            if not is_blob and not entry.name.startswith(_PARTIAL_PREFIX):
                continue
            try:
                if entry.stat(follow_symlinks = False).st_mtime > cutoff:
                    continue
                if is_blob and chat_attachment_blob_is_referenced(entry.name):
                    continue
                os.unlink(entry.path)
                removed += 1
            except FileNotFoundError:
                continue
    return removed
