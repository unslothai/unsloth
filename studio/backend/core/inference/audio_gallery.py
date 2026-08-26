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

    Best-effort: a save must not fail because housekeeping did. Only Unsloth-owned pairs are
    considered, so a foreign or orphan wav is never destroyed.
    """
    cap = _max_clips()
    byte_cap = _max_bytes()
    if cap <= 0 and byte_cap <= 0:
        return 0
    try:
        entries = _list_audio_entries()
    except Exception:  # noqa: BLE001 - never fail the save that triggered this
        return 0

    # Newest first, so the index where either budget runs out is the cut point. The newest
    # clip is always kept: it is the one the caller just generated, and dropping it would
    # make a single oversized request look like a silent failure.
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

    removed = 0
    for record, _cursor in entries[keep:]:
        try:
            if delete(record["id"]):
                removed += 1
        except Exception:  # noqa: BLE001
            continue
    if removed:
        logger.info(
            "audio_gallery.pruned: removed %d clip(s) over the %d clip / %d byte cap",
            removed,
            cap,
            byte_cap,
        )
    return removed


def _record(audio_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "id": audio_id,
        "url": f"/api/inference/audio/gallery/{audio_id}/file",
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
) -> list[tuple[dict[str, Any], GalleryCursor]]:
    try:
        paths = list(gallery_dir().glob("*.wav"))
    except OSError:
        return []
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
        record = _record(path.stem, meta)
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
) -> list[dict[str, Any]]:
    """A newest-first window of clips for infinite scroll.

    Ordered by WAV mtime; only the window's sidecars are read, and a file without its
    pair is skipped. ``valid`` filters BEFORE pagination, so offset, limit and has_more
    all count over the accepted-record domain. ``before`` is an exclusive, stable cursor
    for callers that must tolerate deletions between pages."""
    return [record for record, _ in _list_audio_entries(limit, offset, before = before, valid = valid)]


def list_audio_page(
    limit: int,
    offset: int = 0,
    *,
    before: Optional[GalleryCursor] = None,
    valid: Optional[Callable[[dict[str, Any]], bool]] = None,
) -> list[tuple[dict[str, Any], GalleryCursor]]:
    """Return records with their stable pagination keys for the HTTP route."""
    return _list_audio_entries(limit, offset, before = before, valid = valid)


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
    return True


def clear() -> int:
    """Delete every Unsloth-owned pair (readable sidecar); return the count removed.
    Foreign and orphan WAVs are preserved, since list_audio already hides them."""
    removed = 0
    try:
        paths = list(gallery_dir().glob("*.wav"))
    except OSError:
        return 0
    for path in paths:
        if _read_meta(_sidecar_path(path.stem)) is None:
            continue
        # wav first; if it cannot be unlinked, leave the sidecar so the clip stays listable
        try:
            path.unlink()
        except OSError:
            continue
        removed += 1
        try:
            _sidecar_path(path.stem).unlink()
        except OSError:
            pass
    return removed
