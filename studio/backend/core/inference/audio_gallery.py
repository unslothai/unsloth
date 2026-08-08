# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Disk-backed persistence for generated TTS audio clips.

Each clip is a pair under ``studio_root()/audio``: ``{id}.wav`` holds the bytes,
``{id}.json`` holds the recipe (a WAV has no portable text chunk like a PNG).
The pair travels together; a lone file is not a valid record. Dumb storage: the
route owns the schema; this only reads, writes and sorts.
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

    Both files are staged then renamed in, wav first: the sidecar is the pair's
    commit marker (list_audio skips a wav without a readable sidecar). On any
    failure every artifact is removed so no invisible orphan wav is left behind.
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
    return _record(audio_id, meta)


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
    """Resolve an id to its WAV only when it is a Studio-owned clip (a readable
    sidecar), else None. The serve route uses this instead of audio_path() so a
    guessed stem for a hand-dropped or orphan WAV cannot be streamed out.
    Mirrors the delete/clear ownership guard."""
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

    Ordered by WAV mtime (a cheap stat close to generation order); only the
    window's sidecars are read. limit=None returns everything from ``offset``
    on. A file without its pair is skipped. ``valid`` filters records BEFORE
    pagination so offset, limit and has_more all count over the accepted-record
    domain; pass the route's schema validator. ``before`` is an exclusive,
    stable continuation cursor for callers that must tolerate deletions between
    pages."""
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

    The WAV is unlinked first: dropping the sidecar first and failing on the wav
    would leave a clip that vanished from the gallery with no retry, while
    wav-first leaves at worst an orphan sidecar list_audio ignores."""
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
    """Delete every Studio-owned gallery pair (readable sidecar); return the
    count removed. Foreign and orphan WAVs are preserved: list_audio already
    hides them, so clear must not destroy them."""
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
