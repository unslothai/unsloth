# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The original files of documents sent in chat.

A chat keeps a document's extracted text for the model; the file itself (a PDF, a Word document, a
workbook or a deck) is kept here so it can be opened as it looked. Each is stored once under its
SHA-256, which the attachment records as ``{"original": {"sha256", "sizeBytes"}}``: a fork or an
import copies the reference, never the bytes, and a retried send stores nothing new.

A file no attachment references any more is removed by ``sweep``, once it is older than an hour, so
one uploaded moments before its message is saved is never taken. A sweep that leaves such a file for
being too new schedules another for when it is not.
"""

from __future__ import annotations

import contextvars
import hashlib
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Iterable, Optional

from loggers import get_logger
from utils.paths.storage_roots import account_path, ensure_account_dir

logger = get_logger(__name__)

# The documents a chat can show as pages or a grid. Anything else keeps its text alone.
EXTENSIONS = frozenset({".pdf", ".docx", ".xlsx", ".xlsm", ".pptx"})
# As the chat's own document ceiling (MAX_OPEN_DOCUMENT_ARCHIVE_BYTES in the frontend).
MAX_BYTES = 50 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SWEEP_GRACE_SECONDS = 3600
_SWEEP_INTERVAL_SECONDS = 600
_sweep_lock = threading.Lock()
# Held while a save publishes a file and while a sweep checks and removes one, so a sweep never
# removes a file a save has just refreshed.
_file_lock = threading.Lock()
# Per originals folder: each account has its own, and one account's sweep must not delay another's.
_last_sweep: dict[Path, float] = {}
_scheduled: set[Path] = set()


class TooLarge(Exception):
    pass


def originals_dir() -> Path:
    """This account's originals folder. Only save() creates it, so a late sweep cannot recreate a
    deleted account's workspace."""
    return account_path("chat-originals")


def attachment_sha256(attachment: object) -> Optional[str]:
    """The hash an attachment's ``original`` names, when it is a well-formed one."""
    if not isinstance(attachment, dict):
        return None
    original = attachment.get("original")
    sha256 = original.get("sha256") if isinstance(original, dict) else None
    return sha256 if isinstance(sha256, str) and _SHA256_RE.match(sha256) else None


def attachment_size(attachment: object) -> Optional[int]:
    original = attachment.get("original") if isinstance(attachment, dict) else None
    size = original.get("sizeBytes") if isinstance(original, dict) else None
    return size if isinstance(size, int) and size >= 0 and attachment_sha256(attachment) else None


def path_for(attachment: object) -> Optional[Path]:
    """The stored original of an attachment, if it has one and it is still on disk."""
    sha256 = attachment_sha256(attachment)
    if sha256 is None:
        return None
    path = originals_dir() / sha256
    return path if path.is_file() else None


def save(chunks: Iterable[bytes]) -> tuple[str, int]:
    """Store streamed bytes under their hash: (sha256, size). Raises TooLarge past MAX_BYTES."""
    directory = ensure_account_dir(originals_dir())
    tmp_path = directory / f".{uuid.uuid4().hex}.tmp"
    digest = hashlib.sha256()
    size = 0
    try:
        with open(tmp_path, "wb") as handle:
            for chunk in chunks:
                size += len(chunk)
                if size > MAX_BYTES:
                    raise TooLarge()
                digest.update(chunk)
                handle.write(chunk)
        sha256 = digest.hexdigest()
        final_path = directory / sha256
        with _file_lock:
            if final_path.exists():
                # Already kept: refresh its age, so a sweep leaves it be.
                os.utime(final_path)
            else:
                os.replace(tmp_path, final_path)
        return sha256, size
    finally:
        tmp_path.unlink(missing_ok = True)


def sweep(force: bool = False) -> int:
    """Remove originals no attachment references, older than the grace period. At most once every
    few minutes unless ``force``. Returns how many were removed."""
    from storage.studio_db import referenced_chat_original_hashes

    now = time.time()
    try:
        directory = originals_dir()
        if not directory.is_dir():
            # Nothing kept, or the account is gone.
            return 0
    except (OSError, ValueError):
        return 0
    with _sweep_lock:
        throttled = not force and now - _last_sweep.get(directory, 0.0) < _SWEEP_INTERVAL_SECONDS
        if not throttled:
            _last_sweep[directory] = now
    if throttled:
        # A file uploaded now still needs a sweep once it ages past the grace period.
        _schedule(directory)
        return 0
    try:
        referenced = referenced_chat_original_hashes()
        removed = 0
        waiting = False
        with os.scandir(directory) as entries:
            for entry in entries:
                is_original = _SHA256_RE.match(entry.name) is not None
                if (is_original and entry.name in referenced) or not entry.is_file():
                    continue
                if not (is_original or entry.name.endswith(".tmp")):
                    continue
                # Unreferenced originals, and temp files a crashed upload left behind. Stat afresh
                # under the lock: a save may have refreshed the file since the scan began.
                with _file_lock:
                    try:
                        mtime = os.stat(entry.path).st_mtime
                    except FileNotFoundError:
                        continue
                    if now - mtime < _SWEEP_GRACE_SECONDS:
                        waiting = True
                        continue
                    os.unlink(entry.path)
                removed += 1
        if waiting:
            _schedule(directory)
        return removed
    except Exception:
        logger.debug("chat_originals.sweep_failed", exc_info = True)
        return 0


def _schedule(directory: Path) -> None:
    """Sweep ``directory`` again once the grace period has passed, in this account's context."""
    with _sweep_lock:
        if directory in _scheduled:
            return
        _scheduled.add(directory)
    context = contextvars.copy_context()

    def run() -> None:
        with _sweep_lock:
            _scheduled.discard(directory)
        context.run(sweep, True)

    timer = threading.Timer(_SWEEP_GRACE_SECONDS + 60, run)
    timer.daemon = True
    timer.start()
