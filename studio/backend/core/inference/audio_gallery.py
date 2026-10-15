# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Disk-backed persistence for generated TTS audio clips.

Each clip is a pair under ``workspace_root()/audio``: ``{id}.wav`` holds the bytes and
``{id}.json`` the recipe (a WAV has no portable text chunk). A lone file is not a
valid record. A converted upload is kept as ``{id}.source.wav`` and goes with its clip.
Dumb storage: the route owns the schema, this reads, writes and sorts.
"""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from core.inference import gallery_flags
from core.inference.audio_workflows import workflow_for_audio_type
from loggers import get_logger
from utils.account_context import is_owner_context
from utils.paths import ensure_account_dir, studio_root
from utils.paths.relocations import location_dir
from utils.paths.storage_roots import account_path

logger = get_logger(__name__)

# ids are file stems; safe chars only so a crafted id cannot escape the directory.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def gallery_dir() -> Path:
    if is_owner_context():
        return location_dir("audio", studio_root() / "audio")
    return ensure_account_dir(account_path("audio"))


def save(
    wav_bytes: bytes,
    meta: dict[str, Any],
    source_wav: Optional[Path] = None,
) -> dict[str, Any]:
    """Stages the WAV and sidecar, renaming the WAV first, since the sidecar is the commit marker."""
    audio_id = uuid.uuid4().hex
    directory = gallery_dir()
    wav_path = directory / f"{audio_id}.wav"
    wav_tmp = directory / f".{audio_id}.wav.tmp"
    sidecar = directory / f"{audio_id}.json"
    sidecar_tmp = directory / f".{audio_id}.json.tmp"
    source_path = _source_path(audio_id)
    source_tmp = directory / f".{audio_id}.source.wav.tmp"
    if source_wav is not None:
        meta = {**meta, "source_saved": True}
    try:
        wav_tmp.write_bytes(wav_bytes)
        if source_wav is not None:
            shutil.copyfile(source_wav, source_tmp)
        sidecar_tmp.write_text(json.dumps(meta), encoding = "utf-8")
        os.replace(wav_tmp, wav_path)
        if source_wav is not None:
            os.replace(source_tmp, source_path)
        os.replace(sidecar_tmp, sidecar)
    except BaseException:
        for path in (wav_tmp, sidecar_tmp, source_tmp, wav_path, source_path, sidecar):
            try:
                path.unlink(missing_ok = True)
            except OSError:
                pass
        raise
    _prune_to_cap()
    return _record(audio_id, meta)


def save_file(
    src: Path,
    meta: dict[str, Any],
    *,
    prune: bool = True,
) -> dict[str, Any]:
    """Multi-file runs pass prune=False and prune once at the end, so a cap never splits a run."""
    audio_id = uuid.uuid4().hex
    directory = gallery_dir()
    wav_path = directory / f"{audio_id}.wav"
    wav_tmp = directory / f".{audio_id}.wav.tmp"
    sidecar = directory / f"{audio_id}.json"
    sidecar_tmp = directory / f".{audio_id}.json.tmp"
    try:
        try:
            os.replace(src, wav_path)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copyfile(src, wav_tmp)
            os.replace(wav_tmp, wav_path)
        sidecar_tmp.write_text(json.dumps(meta), encoding = "utf-8")
        os.replace(sidecar_tmp, sidecar)
    except BaseException:
        for path in (wav_tmp, sidecar_tmp, wav_path, sidecar):
            try:
                path.unlink(missing_ok = True)
            except OSError:
                pass
        raise
    if prune:
        _prune_to_cap()
    return _record(audio_id, meta)


# /v1/audio/speech persists every call, so cap the gallery to avoid filling the disk.
_MAX_CLIPS_ENV = "UNSLOTH_AUDIO_GALLERY_MAX_CLIPS"
_DEFAULT_MAX_CLIPS = 2000

_MAX_BYTES_ENV = "UNSLOTH_AUDIO_GALLERY_MAX_BYTES"
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024 * 1024


def _max_clips() -> int:
    """Unset takes the default; 0 or a non-numeric value such as off disables pruning."""
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
    total = 0
    for path in (gallery_dir() / f"{audio_id}.wav", _source_path(audio_id)):
        try:
            total += path.stat().st_size
        except OSError:
            pass
    return total


def _prune_to_cap() -> int:
    """Best-effort; archived and pinned clips are exempt, and an unreadable flag store skips the prune."""
    cap = _max_clips()
    byte_cap = _max_bytes()
    if cap <= 0 and byte_cap <= 0:
        return 0
    directory = gallery_dir()
    removed = 0
    try:
        # Select AND delete under one lock, as clear() does.
        with gallery_flags.exclusive(directory, require_file_lock = True):
            entries = [e for e in _list_audio_entries() if not e[0].get("pinned")]
            entries.sort(key = lambda e: _mtime(gallery_dir() / f"{e[0]['id']}.wav"), reverse = True)

            keep = len(entries) if cap <= 0 else min(cap, len(entries))
            if byte_cap > 0:
                running = 0
                for index, (record, _cursor) in enumerate(entries[:keep]):
                    running += _clip_bytes(record["id"])
                    if running > byte_cap and index > 0:
                        keep = index
                        break
            kept_groups = {r["group_id"] for r, _ in entries[:keep] if r.get("group_id")}
            if keep >= len(entries):
                return 0

            # Re-read TRUSTED: read() reports "nothing archived" for a store it cannot parse.
            flags = gallery_flags.read_trusted(directory)
            victims = [
                record["id"]
                for record, _cursor in entries[keep:]
                if record.get("group_id") not in kept_groups
                and not gallery_flags.is_archived(flags, record["id"])
                and gallery_flags.pin_rank(flags, record["id"]) == float("-inf")
            ]
            in_use = _sources_in_use(directory, set(victims))
            pruned: list[str] = []
            for audio_id in victims:
                if audio_id in in_use:
                    continue
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
                _remove_source(audio_id)
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
        "workflow": _workflow(meta),
        "id": audio_id,
        "url": f"/api/inference/audio/gallery/{audio_id}/file",
        **gallery_flags.flags_for(flags, audio_id),
        "order_at": gallery_flags.order_rank(
            flags, audio_id, _mtime(gallery_dir() / f"{audio_id}.wav")
        ),
    }


def _workflow(meta: dict[str, Any]) -> str:
    workflow = meta.get("workflow")
    if workflow in ("speak", "clone", "edit", "convert", "music", "separate"):
        return workflow
    return workflow_for_audio_type(meta.get("audio_type"))


def audio_path(audio_id: str) -> Optional[Path]:
    """Resolve an id to its on-disk WAV, or None if missing or unsafe."""
    if not _ID_RE.match(audio_id):
        return None
    path = gallery_dir() / f"{audio_id}.wav"
    try:
        path.resolve().relative_to(gallery_dir().resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def _sidecar_path(audio_id: str) -> Path:
    return gallery_dir() / f"{audio_id}.json"


def _source_path(audio_id: str) -> Path:
    return gallery_dir() / f"{audio_id}.source.wav"


def _remove_source(audio_id: str) -> None:
    try:
        _source_path(audio_id).unlink(missing_ok = True)
    except OSError:
        pass


def _clip_wavs(directory: Path) -> list[Path]:
    return [p for p in directory.glob("*.wav") if _ID_RE.match(p.stem)]


def _sources_in_use(directory: Path, leaving: set[str]) -> set[str]:
    """Hidden Edit sources that a clip staying in the gallery still plays as its Original."""
    try:
        paths = list(directory.glob("*.wav"))
    except OSError:
        return set()
    sources: set[str] = set()
    used: set[str] = set()
    for path in paths:
        meta = _read_meta(_sidecar_path(path.stem))
        if meta is None:
            continue
        if meta.get("role") == "source":
            sources.add(path.stem)
        if path.stem not in leaving and meta.get("source_clip_id"):
            used.add(str(meta["source_clip_id"]))
    return used & sources


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
    if not isinstance(meta, dict) or any(k not in meta for k in _REQUIRED_META):
        return None
    return meta


def owned_audio_path(audio_id: str) -> Optional[Path]:
    """Serve routes use this, not audio_path(), so a guessed stem for an orphan WAV cannot stream out."""
    path = audio_path(audio_id)
    if path is None or _read_meta(_sidecar_path(audio_id)) is None:
        return None
    return path


def owned_source_path(audio_id: str) -> Optional[Path]:
    if owned_audio_path(audio_id) is None:
        return None
    path = _source_path(audio_id)
    try:
        path.resolve().relative_to(gallery_dir().resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


GalleryCursor = tuple[float, float, str]


def pin_rank(audio_id: str) -> float:
    """The clip's current pin rank (-inf if unpinned), for a cursor sent without one."""
    return gallery_flags.pin_rank(gallery_flags.read(gallery_dir()), audio_id)


def _sort_key(flags: dict[str, dict[str, Any]], path: Path) -> GalleryCursor:
    return (
        gallery_flags.pin_rank(flags, path.stem),
        gallery_flags.order_rank(flags, path.stem, _mtime(path)),
        path.stem,
    )


def _list_audio_entries(
    limit: Optional[int] = None,
    offset: int = 0,
    *,
    before: Optional[GalleryCursor] = None,
    valid: Optional[Callable[[dict[str, Any]], bool]] = None,
    archived: bool = False,
) -> list[tuple[dict[str, Any], GalleryCursor]]:
    try:
        paths = _clip_wavs(gallery_dir())
    except OSError:
        return []
    flags = gallery_flags.read(gallery_dir())
    paths = [p for p in paths if gallery_flags.is_archived(flags, p.stem) == archived]
    keyed_paths = [(_sort_key(flags, path), path) for path in paths]
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
    """valid filters before pagination, so offset, limit and has_more count accepted records only."""
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


def set_flags(
    audio_id: str,
    *,
    pinned: Optional[bool] = None,
    archived: Optional[bool] = None,
) -> Optional[dict[str, Any]]:
    """Pin or archive one owned clip; None when the id is not an Unsloth-owned clip."""
    with gallery_flags.exclusive(gallery_dir()):
        if owned_audio_path(audio_id) is None:
            return None
        gallery_flags.set_flags_locked(gallery_dir(), audio_id, pinned = pinned, archived = archived)
        meta = _read_meta(_sidecar_path(audio_id))
    if meta is None:
        return None
    return _record(audio_id, meta)


def move(audio_id: str, after_id: Optional[str]) -> Optional[dict[str, Any]]:
    """Move an active clip to just after ``after_id`` (None = front) and return its record.

    None if the id is not an owned, active clip. Raises KeyError if ``after_id`` is not in history."""
    with gallery_flags.exclusive(gallery_dir()):
        if owned_audio_path(audio_id) is None:
            return None
        flags = gallery_flags.read(gallery_dir())
        if gallery_flags.is_archived(flags, audio_id):
            return None
        try:
            paths = [
                p for p in _clip_wavs(gallery_dir()) if not gallery_flags.is_archived(flags, p.stem)
            ]
        except OSError:
            paths = []
        paths.sort(key = lambda p: _sort_key(flags, p), reverse = True)
        keyed = [(p.stem, _mtime(p)) for p in paths]
        gallery_flags.place_locked(gallery_dir(), audio_id, keyed, after_id = after_id)
        meta = _read_meta(_sidecar_path(audio_id))
    if meta is None:
        return None
    return _record(audio_id, meta)


def delete(audio_id: str) -> bool:
    """Removes the WAV first: if the sidecar went first, a failure would lose the clip with no retry."""
    path = audio_path(audio_id)
    if path is None:
        return False
    # only delete a pair we own; a foreign or orphan wav must not be destroyed by a guessed id
    meta = _read_meta(_sidecar_path(audio_id))
    if meta is None:
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
    _remove_source(audio_id)
    gallery_flags.forget(gallery_dir(), [audio_id])
    # An Edit's hidden Original goes with the last clip that plays it.
    original = str(meta.get("source_clip_id") or "")
    if (
        _ID_RE.match(original)
        and (_read_meta(_sidecar_path(original)) or {}).get("role") == "source"
        and original not in _sources_in_use(gallery_dir(), set())
    ):
        delete(original)
    return True


def delete_group(group_id: str) -> int:
    """Delete one run's active clips; archived ones are spared, as in clear()."""
    if not group_id:
        return 0
    directory = gallery_dir()
    # Locked and trusted like clear(): a racing archive or unreadable store never loses a clip.
    with gallery_flags.exclusive(directory, require_file_lock = True):
        flags = gallery_flags.read_trusted(directory)
        try:
            paths = _clip_wavs(directory)
        except OSError:
            return 0
        removed: list[str] = []
        for path in paths:
            if (_read_meta(_sidecar_path(path.stem)) or {}).get(
                "group_id"
            ) != group_id or gallery_flags.is_archived(flags, path.stem):
                continue
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("audio_gallery.delete_failed: %s", exc)
                continue
            removed.append(path.stem)
            try:
                _sidecar_path(path.stem).unlink()
            except OSError:
                pass
            _remove_source(path.stem)
        gallery_flags.forget_locked(directory, removed)
    return len(removed)


def clear(include_archived: bool = False, workflow: Optional[str] = None) -> int:
    """Archived clips are spared unless include_archived; an unreadable flag store raises
    FlagsUnavailable."""
    removed = 0
    directory = gallery_dir()
    with gallery_flags.exclusive(directory, require_file_lock = not include_archived):
        flags = {} if include_archived else gallery_flags.read_trusted(directory)
        try:
            paths = _clip_wavs(directory)
        except OSError:
            return 0
        doomed = []
        for path in paths:
            meta = _read_meta(_sidecar_path(path.stem))
            if meta is None:
                continue
            if workflow is not None and _workflow(meta) != workflow:
                continue
            if not include_archived and gallery_flags.is_archived(flags, path.stem):
                continue
            doomed.append(path)
        in_use = _sources_in_use(directory, {path.stem for path in doomed})
        cleared: list[str] = []
        for path in doomed:
            if path.stem in in_use:
                continue
            try:
                path.unlink()
            except OSError:
                continue
            removed += 1
            cleared.append(path.stem)
            _remove_source(path.stem)
            try:
                _sidecar_path(path.stem).unlink()
            except OSError:
                pass
        if include_archived and workflow is None and not gallery_flags.is_trusted(directory):
            gallery_flags.reset_locked(directory)
        else:
            gallery_flags.forget_locked(directory, cleared)
    return removed
