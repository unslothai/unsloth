# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pin / archive flags for the image and video galleries.

Library state, NOT part of a generation recipe: a PNG's text chunk and a clip's sidecar
describe how the media was made, while "I pinned this" describes how the user files it. So
flags live in their own ``.flags.json`` beside the media rather than in the recipe, and a
record missing from the store simply has no flags.

One store per gallery directory, keyed by the same id the gallery uses (the file stem):

    {"version": 1, "items": {"<id>": {"pinned_at": 1712345678.0, "archived": true}}}

The filename is skipped by the galleries' ``*.png`` / ``*.mp4`` globs, so a store sitting in
the directory is invisible to listing. Every read fails safe: a corrupt, hand-edited or
unreadable store degrades to "no flags", never to an error, because losing a pin is a far
better outcome than a gallery that will not list.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

_SCHEMA_VERSION = 1
_STORE_NAME = ".flags.json"
_lock = threading.RLock()


def _store_path(directory: Path) -> Path:
    return directory / _STORE_NAME


def _empty() -> dict[str, Any]:
    return {"version": _SCHEMA_VERSION, "items": {}}


class FlagsUnavailable(RuntimeError):
    """The store exists but could not be trusted (unparseable, wrong shape, unreadable).

    Distinct from "no store yet", which legitimately means no flags. Callers that only order or
    display flags ignore this and fall back to no flags; callers that DELETE on the strength of a
    flag must fail closed instead, or a corrupt store silently reads every archived item as active.
    """


def _load(directory: Path) -> tuple[dict[str, Any], bool]:
    """``(data, trusted)``. ``trusted`` is False when a store is present but unusable, so a caller
    can tell "nothing is flagged" apart from "we cannot say what is flagged"."""
    try:
        with open(_store_path(directory), encoding = "utf-8-sig") as f:
            data = json.load(f)
        # Validate the shape, not just the version: a hand-edited ``items`` that is not a dict
        # (e.g. ``[]``) would otherwise crash every lookup instead of failing safe.
        if (
            isinstance(data, dict)
            and data.get("version") == _SCHEMA_VERSION
            and isinstance(data.get("items"), dict)
        ):
            return data, True
        logger.warning(
            "gallery_flags.unreadable: %s has an unrecognised shape", _store_path(directory)
        )
        return _empty(), False
    except FileNotFoundError:
        return _empty(), True  # no store yet is a legitimate "nothing is flagged"
    except Exception as exc:
        logger.warning("gallery_flags.read_failed: %s", exc)
        return _empty(), False


def _save(directory: Path, data: dict[str, Any]) -> None:
    """Atomic write (tmp + os.replace), so a crash mid-write never leaves a truncated store.

    Raises on failure. A silent miss would let the API report a pin or archive it never stored,
    which the UI has already applied optimistically, so the action would quietly undo on reload."""
    path = _store_path(directory)
    tmp = directory / f".{_STORE_NAME}.tmp-{os.getpid()}"
    try:
        with open(tmp, "w", encoding = "utf-8") as f:
            json.dump(data, f, indent = 2)
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("gallery_flags.write_failed: %s", exc)
        try:
            tmp.unlink(missing_ok = True)
        except OSError:
            pass
        raise


@contextlib.contextmanager
def _file_lock(directory: Path):
    """Best-effort cross-process exclusive lock over one directory's store. Generation runs in
    subprocesses, so the in-process RLock alone would let two processes read the same JSON and
    clobber each other on ``os.replace``. Degrades to a no-op where OS locking is unavailable
    (the consequence is only a lost flag toggle)."""
    try:
        fd = os.open(str(directory / f"{_STORE_NAME}.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    except Exception:
        yield
        return
    try:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
        except Exception:
            pass  # locking unavailable; the thread lock still applies
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                with contextlib.suppress(Exception):
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _entry(items: dict[str, Any], item_id: str) -> dict[str, Any]:
    """One id's entry, normalized. A non-dict entry (hand-edited) reads as no flags."""
    entry = items.get(item_id)
    return entry if isinstance(entry, dict) else {}


def read(directory: Path) -> dict[str, dict[str, Any]]:
    """Every id's flags for one gallery, read once so a listing pass can sort without
    re-opening the store per file. Fail-safe: an untrusted store reads as no flags, because a
    lost pin beats a gallery that will not list. Use ``read_trusted`` before destructive work."""
    with _lock:
        items = _load(directory)[0].get("items", {})
    return {k: v for k, v in items.items() if isinstance(v, dict)}


def read_trusted(directory: Path) -> dict[str, dict[str, Any]]:
    """``read``, but raises FlagsUnavailable instead of pretending nothing is flagged. For callers
    that delete based on a flag, where guessing "not archived" destroys the archive."""
    with _lock:
        data, trusted = _load(directory)
    if not trusted:
        raise FlagsUnavailable(f"{_store_path(directory)} could not be read")
    items = data.get("items", {})
    return {k: v for k, v in items.items() if isinstance(v, dict)}


def flags_for(items: dict[str, dict[str, Any]], item_id: str) -> dict[str, Any]:
    """The public record fields for one id, from an already-read ``items`` map."""
    entry = _entry(items, item_id)
    return {"pinned": entry.get("pinned_at") is not None, "archived": bool(entry.get("archived"))}


def pin_rank(items: dict[str, dict[str, Any]], item_id: str) -> float:
    """Sort key for the pinned group: most recently pinned first. Unpinned sorts last."""
    pinned_at = _entry(items, item_id).get("pinned_at")
    return float(pinned_at) if isinstance(pinned_at, (int, float)) else float("-inf")


def is_archived(items: dict[str, dict[str, Any]], item_id: str) -> bool:
    return bool(_entry(items, item_id).get("archived"))


def set_flags(
    directory: Path,
    item_id: str,
    *,
    pinned: Optional[bool] = None,
    archived: Optional[bool] = None,
) -> dict[str, Any]:
    """Patch one id's flags; ``None`` leaves that flag alone. Returns the resulting flags.

    Pinning stamps ``pinned_at`` (wall clock) so the pinned group can sort most-recent-first;
    unpinning drops the key rather than storing False, keeping the store to only what is set.
    An id whose flags all end up default is removed entirely, so toggling something on and off
    again leaves no residue."""
    import time

    with _lock, _file_lock(directory):
        # An untrusted store is REPLACED rather than merged: its contents are already unusable, and
        # refusing here would leave the user unable to pin anything until they cleaned it up by hand.
        data = _load(directory)[0]
        items = data.setdefault("items", {})
        entry = dict(_entry(items, item_id))
        if pinned is not None:
            if pinned:
                entry["pinned_at"] = time.time()
            else:
                entry.pop("pinned_at", None)
        if archived is not None:
            if archived:
                entry["archived"] = True
            else:
                entry.pop("archived", None)
        if entry:
            items[item_id] = entry
        else:
            items.pop(item_id, None)
        _save(directory, data)
    return {"pinned": entry.get("pinned_at") is not None, "archived": bool(entry.get("archived"))}


def forget(directory: Path, item_ids) -> None:
    """Drop flags for ids that no longer exist, so a deleted image cannot hand its pin to a
    future id and the store cannot grow without bound. No-op when nothing is stored."""
    ids = {i for i in item_ids if i}
    if not ids:
        return
    with _lock, _file_lock(directory):
        data = _load(directory)[0]
        items = data.get("items", {})
        if not any(i in items for i in ids):
            return  # nothing stored for these ids: skip the write entirely
        for item_id in ids:
            items.pop(item_id, None)
        try:
            _save(directory, data)
        except Exception as exc:  # noqa: BLE001 -- the media is already gone; a stale row is harmless
            logger.warning("gallery_flags.prune_failed: %s", exc)
