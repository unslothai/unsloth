# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Disk-backed persistence for generated TTS audio clips.

Each clip is a pair under ``studio_root()/audio``: ``{id}.wav`` holds the bytes and
``{id}.json`` the recipe (a WAV has no portable text chunk). A lone file is not a
valid record. Dumb storage: the route owns the schema, this reads, writes and sorts.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from core.inference import gallery_flags
from loggers import get_logger
from utils.paths import ensure_dir, studio_root

logger = get_logger(__name__)

# ids are file stems; restrict to safe chars so a crafted id cannot escape the directory
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def gallery_dir() -> Path:
    return ensure_dir(studio_root() / "audio")


def save(wav_bytes: bytes, meta: dict[str, Any]) -> dict[str, Any]:
    """Persist WAV bytes plus their recipe sidecar; return the record.

    Staged then renamed in, wav first: the sidecar is the pair's commit marker. On any
    failure every artifact is removed, so no invisible orphan wav is left behind.
    """
    audio_id = uuid.uuid4().hex
    directory = gallery_dir()
    wav_path = directory / f"{audio_id}.wav"
    wav_tmp = directory / f".{audio_id}.wav.tmp"
    sidecar = directory / f"{audio_id}.json"
    sidecar_tmp = directory / f".{audio_id}.json.tmp"
    try:
        wav_tmp.write_bytes(wav_bytes)
        sidecar_tmp.write_text(json.dumps(meta), encoding = "utf-8")
        os.replace(wav_tmp, wav_path)
        os.replace(sidecar_tmp, sidecar)
    except BaseException:
        for path in (wav_tmp, sidecar_tmp, wav_path, sidecar):
            try:
                path.unlink(missing_ok = True)
            except OSError:
                pass
        raise
    _prune_to_cap()
    return _record(audio_id, meta)


# The OpenAI-compatible /v1/audio/speech route persists every call, so an automated client
# can grow the gallery until the disk fills. Bounded here rather than at that route so the
# UI's own runaway is covered too. Generous by default: this is a convenience gallery, and
# the clip is returned to the caller either way.
_MAX_CLIPS_ENV = "UNSLOTH_AUDIO_GALLERY_MAX_CLIPS"
_DEFAULT_MAX_CLIPS = 2000

# A count alone does not bound the disk: 2000 clips of maximum-length speech is tens of
# gigabytes, and the cap exists to stop /v1/audio/speech filling the disk. Whichever limit
# binds first wins.
_MAX_BYTES_ENV = "UNSLOTH_AUDIO_GALLERY_MAX_BYTES"
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024 * 1024


def _max_clips() -> int:
    """0, or any non-numeric value such as "off", disables pruning.

    An unset variable is the only case that takes the default: restoring it for a value the
    operator did set would delete recordings they had asked to keep.
    """
    return _env_limit(_MAX_CLIPS_ENV, _DEFAULT_MAX_CLIPS)


def _max_bytes() -> int:
    return _env_limit(_MAX_BYTES_ENV, _DEFAULT_MAX_BYTES)


def _env_limit(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0, int(raw.strip()))
    except (AttributeError, ValueError):
        return 0


def _clip_bytes(audio_id: str) -> int:
    try:
        return (gallery_dir() / f"{audio_id}.wav").stat().st_size
    except OSError:
        return 0


def _prune_to_cap() -> int:
    """Drop the oldest owned pairs beyond the count or byte cap; return the count removed.

    Best-effort, so a save never fails on housekeeping. Only Unsloth-owned pairs are eligible,
    archived clips are exempt, and an unreadable flag store skips the prune rather than guess.
    """
    cap = _max_clips()
    byte_cap = _max_bytes()
    if cap <= 0 and byte_cap <= 0:
        return 0
    directory = gallery_dir()
    removed = 0
    try:
        # Select AND delete under one lock, as clear() does. Choosing victims from a snapshot and
        # unlinking after it leaves a window where an archive lands and is deleted anyway.
        with gallery_flags.exclusive(directory, require_file_lock = True):
            entries = _list_audio_entries()

            # Newest first, so the index where either budget runs out is the cut point. The newest
            # is always kept: dropping what the caller just generated looks like a silent failure.
            keep = len(entries) if cap <= 0 else min(cap, len(entries))
            if byte_cap > 0:
                running = 0
                for index, (record, _cursor) in enumerate(entries[:keep]):
                    running += _clip_bytes(record["id"])
                    if running > byte_cap and index > 0:
                        keep = index
                        break
            if keep >= len(entries):
                return 0

            # Re-read TRUSTED immediately before deleting: read() answers "nothing is archived"
            # for a store it cannot parse, which here would drop the clips the shelf exists to
            # keep. It also covers filesystems where the cross-process lock degrades to a no-op.
            flags = gallery_flags.read_trusted(directory)
            pruned: list[str] = []
            for record, _cursor in entries[keep:]:
                audio_id = record["id"]
                if gallery_flags.is_archived(flags, audio_id):
                    continue
                # Not delete(): it takes the flag lock on a second descriptor and would block here.
                path = audio_path(audio_id)
                if path is None or _read_meta(_sidecar_path(audio_id)) is None:
                    continue
                try:
                    path.unlink()
                except OSError as exc:
                    logger.warning("audio_gallery.delete_failed: %s", exc)
                    continue
                removed += 1
                pruned.append(audio_id)
                try:
                    _sidecar_path(audio_id).unlink()
                except OSError:
                    pass
            if pruned:
                gallery_flags.forget_locked(directory, pruned)
    except gallery_flags.FlagsUnavailable:
        logger.warning("audio_gallery.prune_skipped: the archive flags could not be read")
        return 0
    except Exception:  # noqa: BLE001 - never fail the save that triggered this
        return removed

    if removed:
        logger.info(
            "audio_gallery.pruned: removed %d clip(s) over the %d clip / %d byte cap",
            removed,
            cap,
            byte_cap,
        )
    return removed


def _record(
    audio_id: str,
    meta: dict[str, Any],
    flags: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    if flags is None:
        flags = gallery_flags.read(gallery_dir())
    return {
        **meta,
        "id": audio_id,
        "url": f"/api/inference/audio/gallery/{audio_id}/file",
        "archived": gallery_flags.is_archived(flags, audio_id),
    }


def audio_path(audio_id: str) -> Optional[Path]:
    """Resolve an id to its on-disk WAV, or None if missing or unsafe."""
    if not _ID_RE.match(audio_id):
        return None
    path = gallery_dir() / f"{audio_id}.wav"
    # defence in depth: confirm the resolved path is still inside the gallery
    try:
        path.resolve().relative_to(gallery_dir().resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def _sidecar_path(audio_id: str) -> Path:
    return gallery_dir() / f"{audio_id}.json"


# key-presence ownership test: a hand-dropped wav with a partial sidecar is never counted as ours nor destroyed
_REQUIRED_META = (
    "prompt",
    "model",
    "audio_type",
    "sample_rate",
    "duration_s",
    "created_at",
)


def _read_meta(sidecar: Path) -> Optional[dict[str, Any]]:
    try:
        raw = sidecar.read_text(encoding = "utf-8")
    except (OSError, UnicodeError):
        return None
    try:
        meta = json.loads(raw)
    except (ValueError, TypeError):
        return None
    # a parseable dict is not enough: a foreign or different-schema sidecar lacks the required keys
    if not isinstance(meta, dict) or any(k not in meta for k in _REQUIRED_META):
        return None
    return meta


def owned_audio_path(audio_id: str) -> Optional[Path]:
    """Resolve an id to its WAV only for an Unsloth-owned clip (readable sidecar).

    The serve route uses this rather than audio_path() so a guessed stem for a
    hand-dropped or orphan WAV cannot be streamed out."""
    path = audio_path(audio_id)
    if path is None or _read_meta(_sidecar_path(audio_id)) is None:
        return None
    return path


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


GalleryCursor = tuple[float, str]


def _list_audio_entries(
    limit: Optional[int] = None,
    offset: int = 0,
    *,
    before: Optional[GalleryCursor] = None,
    valid: Optional[Callable[[dict[str, Any]], bool]] = None,
    archived: bool = False,
) -> list[tuple[dict[str, Any], GalleryCursor]]:
    try:
        paths = list(gallery_dir().glob("*.wav"))
    except OSError:
        return []
    flags = gallery_flags.read(gallery_dir())
    paths = [p for p in paths if gallery_flags.is_archived(flags, p.stem) == archived]
    keyed_paths = [((_mtime(path), path.stem), path) for path in paths]
    keyed_paths.sort(key = lambda item: item[0], reverse = True)
    want = None if limit is None else offset + limit
    entries = []
    for cursor, path in keyed_paths:
        if before is not None and cursor >= before:
            continue
        meta = _read_meta(_sidecar_path(path.stem))
        if meta is None:
            continue
        record = _record(path.stem, meta, flags)
        if valid is not None and not valid(record):
            continue
        entries.append((record, cursor))
        if want is not None and len(entries) >= want:
            break
    return entries[offset:] if limit is None else entries[offset : offset + limit]


def list_audio(
    limit: Optional[int] = None,
    offset: int = 0,
    *,
    before: Optional[GalleryCursor] = None,
    valid: Optional[Callable[[dict[str, Any]], bool]] = None,
    archived: bool = False,
) -> list[dict[str, Any]]:
    """A newest-first window of clips for infinite scroll.

    Ordered by WAV mtime; a file without its pair is skipped. ``valid`` filters BEFORE pagination,
    so offset, limit and has_more count over the accepted records. ``before`` is an exclusive,
    stable cursor for callers that must tolerate deletions between pages, and ``archived`` picks
    the shelf."""
    return [
        record
        for record, _ in _list_audio_entries(
            limit, offset, before = before, valid = valid, archived = archived
        )
    ]


def list_audio_page(
    limit: int,
    offset: int = 0,
    *,
    before: Optional[GalleryCursor] = None,
    valid: Optional[Callable[[dict[str, Any]], bool]] = None,
    archived: bool = False,
) -> list[tuple[dict[str, Any], GalleryCursor]]:
    """Return records with their stable pagination keys for the HTTP route."""
    return _list_audio_entries(limit, offset, before = before, valid = valid, archived = archived)


def set_flags(audio_id: str, *, archived: Optional[bool] = None) -> Optional[dict[str, Any]]:
    """Archive or restore one owned clip; None when the id is not an Unsloth-owned clip."""
    with gallery_flags.exclusive(gallery_dir()):
        if owned_audio_path(audio_id) is None:
            return None
        gallery_flags.set_flags_locked(gallery_dir(), audio_id, archived = archived)
        meta = _read_meta(_sidecar_path(audio_id))
    if meta is None:
        return None
    return _record(audio_id, meta)


def delete(audio_id: str) -> bool:
    """Remove both files of an owned pair; True if the WAV existed and was ours.

    WAV first: sidecar-first then failing would lose a clip with no retry, while this
    leaves at worst an orphan sidecar list_audio ignores."""
    path = audio_path(audio_id)
    if path is None:
        return False
    # only delete a pair we own; a foreign or orphan wav must not be destroyed by a guessed id
    if _read_meta(_sidecar_path(audio_id)) is None:
        return False
    try:
        path.unlink()
    except OSError as exc:
        logger.warning("audio_gallery.delete_failed: %s", exc)
        return False
    try:
        _sidecar_path(audio_id).unlink()
    except OSError:
        pass
    gallery_flags.forget(gallery_dir(), [audio_id])
    return True


def clear(include_archived: bool = False) -> int:
    """Delete every Unsloth-owned pair (readable sidecar); return the count removed. Foreign and
    orphan WAVs are preserved, since list_audio already hides them.

    Archived clips are spared unless ``include_archived``, and sparing them raises
    FlagsUnavailable when the flag store cannot be read."""
    removed = 0
    directory = gallery_dir()
    with gallery_flags.exclusive(directory, require_file_lock = not include_archived):
        flags = {} if include_archived else gallery_flags.read_trusted(directory)
        try:
            paths = list(directory.glob("*.wav"))
        except OSError:
            return 0
        cleared: list[str] = []
        for path in paths:
            if _read_meta(_sidecar_path(path.stem)) is None:
                continue
            if not include_archived and gallery_flags.is_archived(flags, path.stem):
                continue
            # wav first; if it cannot be unlinked, leave the sidecar so the clip stays listable
            try:
                path.unlink()
            except OSError:
                continue
            removed += 1
            cleared.append(path.stem)
            try:
                _sidecar_path(path.stem).unlink()
            except OSError:
                pass
        if include_archived and not gallery_flags.is_trusted(directory):
            gallery_flags.reset_locked(directory)
        else:
            gallery_flags.forget_locked(directory, cleared)
    return removed
