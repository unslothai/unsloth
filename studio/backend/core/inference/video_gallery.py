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


def transcode(video_id: str, fmt: str) -> Optional[bytes]:
    """Re-encode a stored MP4 for the Download menu: "webm" (VP9, web embeds)
    or "gif" (shareable preview). Returns the encoded bytes, or None when the
    id doesn't resolve. Raises RuntimeError when the codec/deps are missing so
    the route can 501 with a clear message. MP4 downloads never come through
    here -- the /file route streams the original bytes."""
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
            # Realtime-oriented settings: VP9's default "good" profile encodes a
            # few frames per second; cpu-used 8 + row-mt is many times faster at
            # a small quality cost, right for a download button.
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
    # Audio is intentionally dropped: Opus muxing needs a 48 kHz resample chain
    # and most exported clips are silent; the original MP4 keeps the audio.
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
            # Full-rate GIFs are enormous and stutter in chat apps; ~12 fps is
            # the sweet spot, achieved by skipping source frames.
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
    # Delete the MP4 payload FIRST. list_videos globs *.mp4 but requires a readable sidecar, so
    # a video whose sidecar is gone is skipped as an orphan. If the sidecar were dropped first and
    # the mp4 unlink then failed (a Windows lock from a concurrent stream/transcode, or a
    # permission change), the still-present mp4 would vanish from the gallery with no way to retry
    # the delete. Ordering mp4-first means the worst case is an orphaned sidecar, which
    # list_videos ignores.
    try:
        path.unlink()
    except OSError as exc:
        logger.warning("video_gallery.delete_failed: %s", exc)
        return False
    # Best-effort on the sidecar: a leftover json without its mp4 is skipped by list_videos
    # anyway, so a failed sidecar unlink must not fail the delete.
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
        # Delete the mp4 first; if it can't be unlinked, leave the sidecar so the video stays
        # listable (an orphaned mp4 would vanish from the gallery). An orphaned json is harmless.
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
