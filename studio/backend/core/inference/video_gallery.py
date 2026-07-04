# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Disk-backed persistence for generated videos.

Each video is an MP4 under ``studio_root()/videos``. Unlike a PNG, an MP4 has no
portable text-chunk we can embed the generation recipe into, so every video is
stored as a pair: ``{id}.mp4`` holds the encoded bytes and ``{id}.json`` holds
the full recipe as a UTF-8 JSON sidecar (the source of truth the gallery reads
back). The pair travels together on disk; a lone file is not a valid record.

The gallery is intentionally dumb storage: the route owns the metadata schema
and passes a plain dict; this module only writes/reads/sorts files.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger
from utils.paths import ensure_dir, studio_root

logger = get_logger(__name__)

# Video ids are file stems; restrict to filename-safe chars so a crafted id
# can't escape the gallery directory.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def gallery_dir() -> Path:
    return ensure_dir(studio_root() / "videos")


def save(mp4_bytes: bytes, meta: dict[str, Any]) -> dict[str, Any]:
    """Persist encoded MP4 bytes plus their recipe sidecar; return the record."""
    video_id = uuid.uuid4().hex
    directory = gallery_dir()
    (directory / f"{video_id}.mp4").write_bytes(mp4_bytes)
    # Write the sidecar via a tmp file + os.replace so a reader never sees a
    # half-written recipe (an incomplete json would make the pair unlistable).
    sidecar = directory / f"{video_id}.json"
    tmp = directory / f"{video_id}.json.tmp"
    tmp.write_text(json.dumps(meta), encoding = "utf-8")
    os.replace(tmp, sidecar)
    return _record(video_id, meta)


def _record(video_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "id": video_id,
        "url": f"/api/inference/video/gallery/{video_id}/file",
    }


def video_path(video_id: str) -> Optional[Path]:
    """Resolve an id to its on-disk MP4, or None if missing / unsafe."""
    if not _ID_RE.match(video_id):
        return None
    path = gallery_dir() / f"{video_id}.mp4"
    # Defence in depth: confirm the resolved path is still inside the gallery.
    try:
        path.resolve().relative_to(gallery_dir().resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def _sidecar_path(video_id: str) -> Path:
    return gallery_dir() / f"{video_id}.json"


def _read_meta(sidecar: Path) -> Optional[dict[str, Any]]:
    try:
        raw = sidecar.read_text(encoding = "utf-8")
    except OSError:
        return None
    try:
        meta = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return meta if isinstance(meta, dict) else None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def list_videos(limit: Optional[int] = None, offset: int = 0) -> list[dict[str, Any]]:
    """A newest-first window of videos for infinite scroll.

    Ordered by MP4 mtime (a cheap stat, ~= generation order) so a months-old
    gallery isn't opened in full just to sort it; only the window's sidecars are
    read. limit=None returns everything from ``offset`` on. A file without its
    pair (an MP4 with no readable json sidecar, or a sidecar with no MP4) is not
    a valid record and is skipped."""
    try:
        paths = list(gallery_dir().glob("*.mp4"))
    except OSError:
        return []
    paths.sort(key = _mtime, reverse = True)
    # Page over READABLE records, not raw files: filtering an orphan MP4 out of an
    # already-sliced window would drop valid videos that sort after it and make the
    # route's has_more wrong. Read only as far as needed to fill the requested window.
    want = None if limit is None else offset + limit
    records = []
    for path in paths:
        meta = _read_meta(_sidecar_path(path.stem))
        if meta is None:  # no readable sidecar (orphan mp4) — skip
            continue
        records.append(_record(path.stem, meta))
        if want is not None and len(records) >= want:
            break
    return records[offset:] if limit is None else records[offset : offset + limit]


def delete(video_id: str) -> bool:
    """Remove both files of a pair; True if the MP4 existed."""
    path = video_path(video_id)
    if path is None:
        return False
    # Best-effort on the sidecar: a leftover json without its mp4 is skipped by
    # list_videos anyway, so a failed sidecar unlink must not fail the delete.
    try:
        _sidecar_path(video_id).unlink()
    except OSError:
        pass
    try:
        path.unlink()
        return True
    except OSError as exc:
        logger.warning("video_gallery.delete_failed: %s", exc)
        return False


def clear() -> int:
    """Delete every gallery pair; return how many videos were removed."""
    removed = 0
    try:
        paths = list(gallery_dir().glob("*.mp4"))
    except OSError:
        return 0
    for path in paths:
        # Drop the sidecar first; an orphaned json is harmless (skipped by list).
        try:
            _sidecar_path(path.stem).unlink()
        except OSError:
            pass
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed
