# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Disk-backed persistence for generated videos.

Each video is a pair under ``studio_root()/videos``: ``{id}.mp4`` holds the bytes, ``{id}.json``
holds the recipe (an MP4 has no portable text-chunk like a PNG). The pair travels together; a lone
file is not a valid record. Dumb storage: the route owns the schema; this only reads/writes/sorts.
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

# Video ids are file stems; restrict to safe chars so a crafted id can't escape the directory.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def gallery_dir() -> Path:
    return ensure_dir(studio_root() / "videos")


def save(mp4_bytes: bytes, meta: dict[str, Any]) -> dict[str, Any]:
    """Persist encoded MP4 bytes plus their recipe sidecar; return the record."""
    video_id = uuid.uuid4().hex
    directory = gallery_dir()
    mp4_path = directory / f"{video_id}.mp4"
    mp4_tmp = directory / f".{video_id}.mp4.tmp"
    sidecar = directory / f"{video_id}.json"
    sidecar_tmp = directory / f".{video_id}.json.tmp"
    # Stage both files, rename the MP4 in, then the sidecar (the pair's commit marker: list_videos
    # skips an mp4 without a readable sidecar). On any failure remove every artifact, else a sidecar
    # failure would leave an invisible, undeletable orphan MP4.
    try:
        mp4_tmp.write_bytes(mp4_bytes)
        sidecar_tmp.write_text(json.dumps(meta), encoding = "utf-8")
        os.replace(mp4_tmp, mp4_path)
        os.replace(sidecar_tmp, sidecar)
    except BaseException:
        for path in (mp4_tmp, sidecar_tmp, mp4_path, sidecar):
            try:
                path.unlink(missing_ok = True)
            except OSError:
                pass
        raise
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


def transcode(video_id: str, fmt: str) -> Optional[bytes]:
    """Re-encode a stored MP4 for the Download menu: "webm" (VP9) or "gif". Returns the bytes, or
    None when the id doesn't resolve. Raises RuntimeError on missing codec/deps (route 501s). MP4
    downloads stream the original via /file, not here."""
    path = video_path(video_id)
    if path is None:
        return None
    normalized = fmt.strip().lower()
    if normalized == "webm":
        return _transcode_webm(path)
    if normalized == "gif":
        return _transcode_gif(path)
    raise ValueError(f"Unsupported export format '{fmt}'. Use webm or gif.")


def _transcode_webm(path: Path) -> bytes:
    import io

    try:
        import av
    except Exception as exc:  # noqa: BLE001 -- no PyAV -> no transcode
        raise RuntimeError("WebM export needs the 'av' package (PyAV).") from exc
    buf = io.BytesIO()
    try:
        with av.open(str(path)) as src, av.open(buf, "w", format = "webm") as dst:
            if not src.streams.video:
                raise RuntimeError("WebM export failed: the clip has no video stream.")
            in_v = src.streams.video[0]
            rate = in_v.average_rate or 24
            out_v = dst.add_stream("libvpx-vp9", rate = rate)
            out_v.width = in_v.codec_context.width
            out_v.height = in_v.codec_context.height
            out_v.pix_fmt = "yuv420p"
            # Realtime settings: VP9's default "good" profile is slow; cpu-used 8 + row-mt is much
            # faster at a small quality cost, right for a download button.
            out_v.options = {"deadline": "realtime", "cpu-used": "8", "row-mt": "1"}
            for frame in src.decode(in_v):
                for packet in out_v.encode(frame.reformat(format = "yuv420p")):
                    dst.mux(packet)
            for packet in out_v.encode():
                dst.mux(packet)
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 -- surface as "encoder unavailable"
        raise RuntimeError(f"WebM export failed (libvpx-vp9 unavailable?): {exc}") from exc
    # Audio dropped: Opus muxing needs a 48 kHz resample chain and most clips are silent (the
    # original MP4 keeps the audio).
    return buf.getvalue()


def _transcode_gif(path: Path) -> bytes:
    import io

    try:
        import av
        from PIL import Image
    except Exception as exc:  # noqa: BLE001 -- missing deps -> no transcode
        raise RuntimeError("GIF export needs the 'av' and 'Pillow' packages.") from exc
    frames: list[Any] = []
    try:
        with av.open(str(path)) as src:
            if not src.streams.video:
                raise RuntimeError("GIF export failed: the clip has no video stream.")
            in_v = src.streams.video[0]
            rate = float(in_v.average_rate or 24)
            # Full-rate GIFs are huge and stutter; ~12 fps (skipping source frames) is the sweet spot.
            step = max(1, round(rate / 12))
            for i, frame in enumerate(src.decode(in_v)):
                if i % step:
                    continue
                frames.append(frame.to_image().convert("P", palette = Image.Palette.ADAPTIVE))
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 -- surface as "decoder unavailable"
        raise RuntimeError(f"GIF export failed to decode the clip: {exc}") from exc
    if not frames:
        raise RuntimeError("GIF export decoded no frames.")
    duration_ms = max(20, int(1000 * step / rate))
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format = "GIF",
        save_all = True,
        append_images = frames[1:],
        duration = duration_ms,
        loop = 0,
    )
    return buf.getvalue()


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


def list_videos(
    limit: Optional[int] = None,
    offset: int = 0,
    *,
    valid: Optional[Callable[[dict[str, Any]], bool]] = None,
) -> list[dict[str, Any]]:
    """A newest-first window of videos for infinite scroll.

    Ordered by MP4 mtime (a cheap stat ~= generation order); only the window's sidecars are read.
    limit=None returns everything from ``offset`` on. A file without its pair is skipped.

    ``valid`` (optional) filters records BEFORE pagination, so ``offset`` / ``limit`` and has_more
    all count over the accepted-record domain. Pass the route's schema validator: a sidecar that
    parses as JSON but fails the response schema would otherwise be counted here yet dropped after
    slicing, stalling infinite scroll."""
    try:
        paths = list(gallery_dir().glob("*.mp4"))
    except OSError:
        return []
    paths.sort(key = _mtime, reverse = True)
    # Page over READABLE records, not raw files: filtering an orphan MP4 out of an already-sliced
    # window would drop valid videos and make has_more wrong. Read only as far as needed.
    want = None if limit is None else offset + limit
    records = []
    for path in paths:
        meta = _read_meta(_sidecar_path(path.stem))
        if meta is None:  # orphan mp4 (no readable sidecar)
            continue
        record = _record(path.stem, meta)
        if valid is not None and not valid(record):  # parses but schema-invalid
            continue
        records.append(record)
        if want is not None and len(records) >= want:
            break
    return records[offset:] if limit is None else records[offset : offset + limit]


def delete(video_id: str) -> bool:
    """Remove both files of a pair; True if the MP4 existed."""
    path = video_path(video_id)
    if path is None:
        return False
    # Delete the MP4 FIRST: if the sidecar were dropped first and the mp4 unlink then failed (lock /
    # permission), the still-present mp4 would vanish from the gallery with no retry. mp4-first means
    # the worst case is an orphaned sidecar, which list_videos ignores.
    try:
        path.unlink()
    except OSError as exc:
        logger.warning("video_gallery.delete_failed: %s", exc)
        return False
    # Best-effort sidecar unlink: a leftover json is skipped by list_videos anyway.
    try:
        _sidecar_path(video_id).unlink()
    except OSError:
        pass
    return True


def clear() -> int:
    """Delete every gallery pair; return how many videos were removed."""
    removed = 0
    try:
        paths = list(gallery_dir().glob("*.mp4"))
    except OSError:
        return 0
    for path in paths:
        # mp4 first; if it can't be unlinked, leave the sidecar so the video stays listable.
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
