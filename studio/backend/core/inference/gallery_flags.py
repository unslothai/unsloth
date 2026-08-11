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
import math
import os
import threading
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

_SCHEMA_VERSION = 1
_STORE_NAME = ".flags.json"
# Marks a store written over one whose ITEMS MAP could not be read at all. Such a write cannot
# carry the old archive flags forward, because they were never legible, so the new file must not
# be taken as proof that nothing is archived. See ``_carry_taint``.
_TAINT_KEY = "unreadable"
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


def _valid_entry(entry: Any) -> bool:
    """Whether an entry is exactly the shape this module writes.

    The container being a dict is not enough. ``{"archived": null}`` is a dict, and every reader
    turns it into "not archived", which is what ``clear`` deletes on. Nothing here ever writes a
    non-bool ``archived`` or an unusable ``pinned_at``, so either one means the file was edited or
    damaged and no field in it can be taken at face value."""
    if not isinstance(entry, dict):
        return False
    if "archived" in entry and not isinstance(entry["archived"], bool):
        return False
    if "pinned_at" in entry and _pinned_at(entry) is None:
        return False
    return True


def _sanitize_entry(entry: Any) -> Optional[dict[str, Any]]:
    """The entry rewritten into a shape this module can read, or None when it held nothing.

    Damage to ``archived`` is RESOLVED to True, never dropped. Dropping it would turn "we cannot
    tell whether this was archived" into "this is active", and active is what ``clear`` deletes;
    an item wrongly moved to the archive shelf is one click to undo, an item wrongly deleted is
    gone. An ABSENT ``archived`` is not damage: unarchiving removes the key, so absent genuinely
    means active. A non-dict entry has no readable field at all and only exists because something
    was flagged, so it resolves the same safe way.

    ``pinned_at`` is dropped instead, since losing a pin costs the user an ordering, not a file."""
    if not isinstance(entry, dict):
        return {"archived": True}
    clean = dict(entry)
    if "archived" in clean and not isinstance(clean["archived"], bool):
        clean["archived"] = True
    if "pinned_at" in clean and _pinned_at(clean) is None:
        clean.pop("pinned_at")
    return clean or None


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
            # A store carrying the taint below was written over one whose contents could not be
            # read at all, so what it does NOT say is not evidence of anything.
            if data.get(_TAINT_KEY):
                return data, False
            # Every ENTRY has to be readable too, not just the container. A malformed value is
            # dropped by the readers below, which reads as "this id is not archived" -- enough for
            # clear() to delete an archived file. So one bad entry costs the store its trust, but
            # the surviving entries are still returned: listing should keep the flags it can read,
            # and only destructive callers need to refuse.
            if all(_valid_entry(v) for v in data["items"].values()):
                return data, True
            logger.warning(
                "gallery_flags.unreadable: %s has a malformed entry", _store_path(directory)
            )
            return data, False
        logger.warning(
            "gallery_flags.unreadable: %s has an unrecognised shape", _store_path(directory)
        )
        return _empty(), False
    except FileNotFoundError:
        return _empty(), True  # no store yet is a legitimate "nothing is flagged"
    except Exception as exc:
        logger.warning("gallery_flags.read_failed: %s", exc)
        return _empty(), False


def _carry_taint(data: dict[str, Any], trusted: bool) -> dict[str, Any]:
    """``data`` prepared for a rewrite, carrying the taint when the old contents were illegible.

    Entry-level damage is repaired in place by ``_sanitize_entry``, so the flags that were still
    readable survive the write and the store earns its trust back. Damage to the CONTAINER --
    truncated JSON, an ``items`` that is not a dict, a version we do not know -- leaves nothing to
    carry forward: ``_load`` substitutes an empty map, and writing that as an ordinary store turns
    "we cannot say what was archived" into "nothing is archived", which is the answer the default
    ``clear()`` deletes on. So the replacement is marked, and every destructive caller keeps
    failing closed until a human resolves it. Listing, pinning, archiving and restoring all keep
    working; ``clear(include_archived = True)`` is the deliberate way out, since it spares nothing
    and so needs no flags to be correct.
    """
    if not trusted and not data.get("items"):
        data[_TAINT_KEY] = True
    return data


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
    locked = False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
            locked = True
        except Exception:
            pass  # locking unavailable; the thread lock still applies
        yield
    finally:
        # Release only what was taken, and never let the release be the thing that fails the call.
        # A filesystem that cannot lock usually cannot unlock either, and by this point the body
        # has already written the flags or deleted the media, so raising here reports a failure for
        # work that landed.
        try:
            if locked:
                with contextlib.suppress(Exception):
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@contextlib.contextmanager
def exclusive(directory: Path):
    """Hold the store's write lock across a read-then-act sequence.

    ``clear`` decides what to delete from a snapshot of the flags and then unlinks files, so an
    archive landing in that window would be classified active from the stale snapshot and deleted
    anyway -- after the PATCH had already told the user it was archived. Taking the same lock
    ``set_flags`` takes serializes the two.
    """
    with _lock, _file_lock(directory):
        yield


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


def is_trusted(directory: Path) -> bool:
    """Whether the store can be believed about what is NOT flagged."""
    with _lock:
        return _load(directory)[1]


def reset_locked(directory: Path) -> None:
    """Replace the store with an empty, trusted one. For a caller already inside ``exclusive()``.

    Only ``clear(include_archived = True)`` does this, and only once it has removed every image we
    own. That is the way out of an unreadable store: the taint exists to protect files from a
    delete that cannot prove them active, and after this there are no such files left. Without the
    reset the escape hatch is not one, since the corrupt file survives the wipe and every later
    clear still refuses, including for media generated afterwards."""
    _save(directory, _empty())


def read_trusted(directory: Path) -> dict[str, dict[str, Any]]:
    """``read``, but raises FlagsUnavailable instead of pretending nothing is flagged. For callers
    that delete based on a flag, where guessing "not archived" destroys the archive."""
    with _lock:
        data, trusted = _load(directory)
    if not trusted:
        raise FlagsUnavailable(f"{_store_path(directory)} could not be read")
    items = data.get("items", {})
    return {k: v for k, v in items.items() if isinstance(v, dict)}


def _pinned_at(entry: dict[str, Any]) -> Optional[float]:
    """The entry's pin time as a usable float, or None when it is absent or unusable.

    JSON integers are unbounded, so a hand-edited ``pinned_at`` of a few hundred digits overflows
    ``float()``. That is read at listing time, from a store whose whole contract is to degrade to
    "no flags" rather than raise, so an unconvertible value must read as unpinned instead of
    turning every gallery request into a 500. NaN / infinity are refused for the same reason: they
    would poison the sort rather than fail it."""
    value = entry.get("pinned_at")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        pinned_at = float(value)
    except (OverflowError, ValueError):
        return None
    return pinned_at if math.isfinite(pinned_at) else None


def flags_for(items: dict[str, dict[str, Any]], item_id: str) -> dict[str, Any]:
    """The public record fields for one id, from an already-read ``items`` map."""
    entry = _entry(items, item_id)
    return {
        # Reported through the same conversion the sort uses, so a value the ordering cannot use
        # never shows as a pin the user then cannot explain.
        "pinned": _pinned_at(entry) is not None,
        "archived": bool(entry.get("archived")),
    }


def pin_rank(items: dict[str, dict[str, Any]], item_id: str) -> float:
    """Sort key for the pinned group: most recently pinned first. Unpinned sorts last."""
    pinned_at = _pinned_at(_entry(items, item_id))
    return pinned_at if pinned_at is not None else float("-inf")


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
    with _lock, _file_lock(directory):
        return set_flags_locked(directory, item_id, pinned = pinned, archived = archived)


def set_flags_locked(
    directory: Path,
    item_id: str,
    *,
    pinned: Optional[bool] = None,
    archived: Optional[bool] = None,
) -> dict[str, Any]:
    """``set_flags`` for a caller already inside ``exclusive()``, so the ownership check and the
    write land as one step. Separate for the same per-descriptor lock reason as ``forget_locked``."""
    import time

    # A write REPAIRS the store rather than preserving what made it untrusted. Merging the bad
    # entry straight back would leave every later clear() refused until someone fixed the file by
    # hand, and refusing here instead would leave the user unable to pin anything at all. Dropping
    # only the unreadable entries keeps the flags that still mean something.
    data = _carry_taint(*_load(directory))
    items: dict[str, Any] = {}
    for key, value in data.get("items", {}).items():
        clean = _sanitize_entry(value)
        if clean is not None:
            items[key] = clean
    data["items"] = items
    entry = dict(_entry(items, item_id))
    if pinned is not None:
        if pinned:
            # Strictly ahead of every stamp already stored, not just the wall clock. Windows'
            # time.time() advances in ~16 ms steps, so two pins a click apart can land on the SAME
            # value and "most recently pinned leads" quietly stops holding for them -- which is the
            # ordering the client serializes its PATCHes to preserve.
            latest = max(
                (_pinned_at(v) for v in items.values() if _pinned_at(v) is not None),
                default = float("-inf"),
            )
            now = time.time()
            nudged = math.nextafter(latest, math.inf) if latest != float("-inf") else now
            # A hand-edited store holding the largest finite float nudges to infinity, which
            # json.dump writes happily and _pinned_at then refuses, so the pin this call just
            # reported would read back as unset AND take the store's trust down with it, blocking
            # the default clear. Tie with the largest stamp instead: the group then falls back to
            # mtime for those two, which costs an ordering rather than the store.
            entry["pinned_at"] = now if now > latest else (nudged if math.isfinite(nudged) else latest)
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
    with _lock, _file_lock(directory):
        forget_locked(directory, item_ids)


def forget_locked(directory: Path, item_ids) -> None:
    """``forget`` for a caller already inside ``exclusive()``. Separate because the cross-process
    lock is per file descriptor: re-taking it on a second descriptor in the same process blocks
    against the one already held, so the nested call would deadlock rather than recurse."""
    ids = {i for i in item_ids if i}
    if not ids:
        return
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
