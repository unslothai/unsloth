# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Disk-backed persistence for generated videos.

Each video is a pair under ``workspace_root()/videos``: ``{id}.mp4`` holds the bytes, ``{id}.json``
holds the recipe (an MP4 has no portable text-chunk like a PNG). The pair travels together; a lone
file is not a valid record. Dumb storage: the route owns the schema; this only reads/writes/sorts.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

from core.inference import gallery_flags
from loggers import get_logger
from utils.account_context import is_owner_context
from utils.paths import ensure_account_dir, ensure_dir, studio_root
from utils.paths.relocations import location_dir
from utils.paths.storage_roots import account_path

logger = get_logger(__name__)

# Video ids are file stems; restrict to safe chars so a crafted id can't escape the directory.
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_JOB_OUTCOME_KEY = "_worker_outcome"
_job_lock = threading.Lock()
_THUMBNAIL_WIDTH = 192


def gallery_dir() -> Path:
    if is_owner_context():
        return location_dir("videos", studio_root() / "videos")
    return ensure_account_dir(account_path("videos"))


def _job_dir() -> Path:
    return ensure_dir(gallery_dir() / ".jobs")


def _job_path(video_id: str) -> Optional[Path]:
    if not _ID_RE.match(video_id):
        return None
    return _job_dir() / f"{video_id}.json"


def _read_job(path: Path, video_id: str) -> Optional[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return None
    return raw if isinstance(raw, dict) and raw.get("id") == video_id else None


def _write_job(path: Path, job: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(job), encoding = "utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok = True)


def save_job(video_id: str, job: dict[str, Any]) -> None:
    """Atomically persist an OpenAI job, preserving any worker outcome that raced it."""
    path = _job_path(video_id)
    if path is None:
        raise ValueError("Invalid video id.")
    with _job_lock:
        existing = _read_job(path, video_id) or {}
        stored = dict(job)
        if isinstance(existing.get(_JOB_OUTCOME_KEY), dict):
            stored[_JOB_OUTCOME_KEY] = existing[_JOB_OUTCOME_KEY]
        _write_job(path, stored)


def record_job_outcome(
    video_id: str,
    *,
    completed_at: int,
    error: Optional[str] = None,
) -> None:
    """Persist a worker result by id before the backend can accept another generation."""
    path = _job_path(video_id)
    if path is None:
        raise ValueError("Invalid video id.")
    with _job_lock:
        stored = _read_job(path, video_id) or {"id": video_id}
        stored[_JOB_OUTCOME_KEY] = {"completed_at": completed_at, "error": error}
        _write_job(path, stored)


def get_job(video_id: str) -> Optional[dict[str, Any]]:
    path = _job_path(video_id)
    if path is None:
        return None
    with _job_lock:
        return _read_job(path, video_id)


def list_jobs() -> list[dict[str, Any]]:
    try:
        paths = list(_job_dir().glob("*.json"))
    except OSError:
        return []
    jobs = []
    for path in paths:
        job = get_job(path.stem)
        if job is not None:
            jobs.append(job)
    return jobs


def forget_job(video_id: str) -> bool:
    """Remove a persisted OpenAI job; missing state already satisfies the request."""
    path = _job_path(video_id)
    if path is None:
        return False
    with _job_lock:
        try:
            path.unlink(missing_ok = True)
        except OSError as exc:
            logger.warning("video_gallery.forget_job_failed: %s", exc)
            return False
    return True


def save(
    mp4_bytes: bytes,
    meta: dict[str, Any],
    video_id: Optional[str] = None,
) -> dict[str, Any]:
    """Persist encoded MP4 bytes plus their recipe sidecar; return the record."""
    if video_id is None:
        video_id = uuid.uuid4().hex
    elif not _ID_RE.match(video_id):
        raise ValueError("Invalid video id.")
    directory = gallery_dir()
    mp4_path = directory / f"{video_id}.mp4"
    mp4_tmp = directory / f".{video_id}.mp4.tmp"
    sidecar = directory / f"{video_id}.json"
    sidecar_tmp = directory / f".{video_id}.json.tmp"
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


def _record(
    video_id: str,
    meta: dict[str, Any],
    flags: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    # flags are library state, not recipe: they come from the .flags.json store, never the sidecar
    if flags is None:
        flags = gallery_flags.read(gallery_dir())
    return {
        **meta,
        "id": video_id,
        "url": f"/api/inference/video/gallery/{video_id}/file",
        **gallery_flags.flags_for(flags, video_id),
        "order_at": gallery_flags.order_rank(
            flags, video_id, _mtime(path) if (path := video_path(video_id)) else 0.0
        ),
    }


def video_path(video_id: str) -> Optional[Path]:
    """Resolve an id to its on-disk MP4, or None if missing / unsafe."""
    if not _ID_RE.match(video_id):
        return None
    path = gallery_dir() / f"{video_id}.mp4"
    try:
        path.resolve().relative_to(gallery_dir().resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def thumbnail(video_id: str) -> Optional[bytes]:
    """Encode the first frame of an owned clip as the OpenAI-compatible WebP preview."""
    path = owned_video_path(video_id)
    if path is None:
        return None
    return first_frame_webp(path)


def first_frame_webp(
    source: Union[Path, BinaryIO],
    width: int = _THUMBNAIL_WIDTH,
    *,
    container: Optional[str] = None,
    max_pixels: Optional[int] = None,
    max_height: Optional[int] = None,
) -> bytes:
    """First frame as WebP; a forced container reads only the stream, not files a probed playlist names."""
    import io

    try:
        import av
        from PIL import Image
    except Exception as exc:  # noqa: BLE001 -- a missing decoder dependency makes thumbnails unavailable
        raise RuntimeError("Thumbnail generation needs the 'av' and 'Pillow' packages.") from exc
    if isinstance(source, Path):
        target, protocols = str(source), "file"
    else:
        target, protocols = source, "none"
    options = {"protocol_whitelist": protocols} if container else {}
    try:
        with av.open(target, format = container, options = options) as src:
            if not src.streams.video:
                raise RuntimeError("Thumbnail generation failed: the clip has no video stream.")
            stream = src.streams.video[0]
            if max_pixels is not None:
                if stream.width * stream.height > max_pixels:
                    raise RuntimeError(f"{stream.width}x{stream.height} is too large to thumbnail.")
                stream.codec_context.options = {"max_pixels": str(max_pixels)}
            frame = next(src.decode(stream), None)
            if frame is None:
                raise RuntimeError("Thumbnail generation failed: the clip has no decodable frames.")
            image = frame.to_image()
            scale = min(1, width / image.width, (max_height or width * 4) / image.height)
            if scale < 1:
                image = image.resize(
                    (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                    Image.LANCZOS,
                )
            buf = io.BytesIO()
            image.save(buf, format = "WEBP", quality = 85, method = 4)
            return buf.getvalue()
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 -- surface decoder / WebP encoder failures
        raise RuntimeError(f"Thumbnail generation failed to decode the clip: {exc}") from exc


def transcode_to_file(video_id: str, fmt: str) -> Optional[Path]:
    """Re-encodes to a temp file, not memory; the caller must delete it after serving."""
    # only transcode an Unsloth-owned clip, so a guessed stem for a foreign MP4 cannot be re-encoded out
    path = owned_video_path(video_id)
    if path is None:
        return None
    normalized = fmt.strip().lower()
    if normalized not in ("webm", "gif"):
        raise ValueError(f"Unsupported export format '{fmt}'. Use webm or gif.")
    import tempfile

    fd, tmp_name = tempfile.mkstemp(prefix = f"unsloth-export-{video_id}-", suffix = f".{normalized}")
    os.close(fd)
    dest = Path(tmp_name)
    try:
        if normalized == "webm":
            _transcode_webm(path, dest)
        else:
            dest.write_bytes(_transcode_gif(path))
    except BaseException:
        dest.unlink(missing_ok = True)
        raise
    return dest


def transcode(video_id: str, fmt: str) -> Optional[bytes]:
    """``transcode_to_file`` read back into memory. Kept for callers that want the bytes; the route
    uses the file form so a large export is never fully resident."""
    dest = transcode_to_file(video_id, fmt)
    if dest is None:
        return None
    try:
        return dest.read_bytes()
    finally:
        dest.unlink(missing_ok = True)


def _transcode_webm(path: Path, dest: Path) -> None:
    """Transcode ``path`` to VP9 (+ Opus when the clip has audio) at ``dest``."""
    try:
        import av
    except Exception as exc:  # noqa: BLE001 -- no PyAV -> no transcode
        raise RuntimeError("WebM export needs the 'av' package (PyAV).") from exc
    try:
        with av.open(str(path)) as src, av.open(str(dest), "w", format = "webm") as dst:
            if not src.streams.video:
                raise RuntimeError("WebM export failed: the clip has no video stream.")
            in_v = src.streams.video[0]
            rate = in_v.average_rate or 24
            out_v = dst.add_stream("libvpx-vp9", rate = rate)
            out_v.width = in_v.codec_context.width
            out_v.height = in_v.codec_context.height
            out_v.pix_fmt = "yuv420p"
            # VP9 realtime: cpu-used 8 + row-mt is much faster at a small quality cost.
            out_v.options = {"deadline": "realtime", "cpu-used": "8", "row-mt": "1"}
            # Keep LTX-2 audio: Opus needs 48 kHz and whole 960-sample frames via a FIFO.
            in_a = src.streams.audio[0] if src.streams.audio else None
            out_a = fifo = resampler = None
            if in_a is not None:
                try:
                    stereo = (getattr(in_a.codec_context.layout, "nb_channels", 1) or 1) > 1
                    layout = "stereo" if stereo else "mono"
                    out_a = dst.add_stream("libopus", rate = 48000, layout = layout)
                    resampler = av.audio.resampler.AudioResampler(
                        format = out_a.format.name, layout = layout, rate = 48000
                    )
                    fifo = av.audio.fifo.AudioFifo()
                except Exception:  # noqa: BLE001 -- a build without libopus still exports the video
                    out_a = fifo = resampler = None

            def _drain_audio(flush: bool = False) -> None:
                size = out_a.frame_size or 960
                while True:
                    frame = fifo.read(size, partial = flush)
                    if frame is None:
                        break
                    for packet in out_a.encode(frame):
                        dst.mux(packet)

            for packet in src.demux(*([in_v] + ([in_a] if out_a is not None else []))):
                if packet.dts is None:
                    continue
                if packet.stream is in_v:
                    for frame in packet.decode():
                        for out_packet in out_v.encode(frame.reformat(format = "yuv420p")):
                            dst.mux(out_packet)
                    continue
                for frame in packet.decode():
                    for resampled in resampler.resample(frame):
                        # FIFO times output: resampler frames do not match Opus' frame size.
                        resampled.pts = None
                        fifo.write(resampled)
                    _drain_audio()
            for packet in out_v.encode():
                dst.mux(packet)
            if out_a is not None:
                _drain_audio(flush = True)
                for packet in out_a.encode():
                    dst.mux(packet)
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 -- surface as "encoder unavailable"
        raise RuntimeError(f"WebM export failed (libvpx-vp9 unavailable?): {exc}") from exc


# GIF holds every frame in memory: 720 px / 300 frames bounds it at ~150 MB.
_GIF_MAX_EDGE = 720
_GIF_MAX_FRAMES = 300


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
            step = max(1, round(rate / 12))
            # Bound both axes or a long clip is a memory bomb (>4 GB); MP4 keeps the full clip.
            total = in_v.frames or 0
            kept = (total + step - 1) // step if total else 0
            if kept > _GIF_MAX_FRAMES:
                step = -(-total // _GIF_MAX_FRAMES)
            for i, frame in enumerate(src.decode(in_v)):
                if i % step:
                    continue
                if len(frames) >= _GIF_MAX_FRAMES:
                    break
                image = frame.to_image()
                if max(image.size) > _GIF_MAX_EDGE:
                    scale = _GIF_MAX_EDGE / max(image.size)
                    image = image.resize(
                        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                        Image.Resampling.LANCZOS,
                    )
                frames.append(image.convert("P", palette = Image.Palette.ADAPTIVE))
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


# Keys every genuine Unsloth sidecar carries; delete()/clear() only own pairs that have all.
_REQUIRED_META = (
    "prompt",
    "width",
    "height",
    "num_frames",
    "fps",
    "duration_s",
    "steps",
    "guidance",
    "seed",
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
    # A foreign or other-schema sidecar lacks these keys and must never be deleted.
    if not isinstance(meta, dict) or any(k not in meta for k in _REQUIRED_META):
        return None
    return meta


def get_record(video_id: str) -> Optional[dict[str, Any]]:
    if owned_video_path(video_id) is None:
        return None
    meta = _read_meta(_sidecar_path(video_id))
    if meta is None:
        return None
    return _record(video_id, meta)


def owned_video_path(video_id: str) -> Optional[Path]:
    """Owned clips only (readable sidecar); a guessed stem for a hand-dropped MP4 must not be streamed."""
    path = video_path(video_id)
    if path is None or _read_meta(_sidecar_path(video_id)) is None:
        return None
    return path


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
    archived: bool = False,
) -> list[dict[str, Any]]:
    """Pinned then newest-first; valid filters run before slicing so has_more counts accepted clips."""
    try:
        paths = list(gallery_dir().glob("*.mp4"))
    except OSError:
        return []
    flags = gallery_flags.read(gallery_dir())
    # Shelf split and pin sort use stems only, before any sidecar read, keeping the early break.
    paths = [p for p in paths if gallery_flags.is_archived(flags, p.stem) == archived]
    paths.sort(
        key = lambda p: (
            gallery_flags.pin_rank(flags, p.stem),
            gallery_flags.order_rank(flags, p.stem, _mtime(p)),
        ),
        reverse = True,
    )
    # Page over READABLE records, else orphans skew the window and has_more.
    want = None if limit is None else offset + limit
    records = []
    for path in paths:
        meta = _read_meta(_sidecar_path(path.stem))
        if meta is None:
            continue
        record = _record(path.stem, meta, flags)
        if valid is not None and not valid(record):
            continue
        records.append(record)
        if want is not None and len(records) >= want:
            break
    return records[offset:] if limit is None else records[offset : offset + limit]


def set_flags(
    video_id: str,
    *,
    pinned: Optional[bool] = None,
    archived: Optional[bool] = None,
) -> Optional[dict[str, Any]]:
    """Patch one clip's pin/archive flags and return its updated record, or None when the id is
    not an Unsloth-owned clip. Ownership-gated like delete: a guessed stem for a hand-dropped or
    orphan MP4 must not become flaggable."""
    # Ownership check and write under one lock so a concurrent clear cannot race them.
    with gallery_flags.exclusive(gallery_dir()):
        if owned_video_path(video_id) is None:
            return None
        gallery_flags.set_flags_locked(gallery_dir(), video_id, pinned = pinned, archived = archived)
        meta = _read_meta(_sidecar_path(video_id))
    if meta is None:  # raced a delete between the guard and the read
        return None
    return _record(video_id, meta)


def move(video_id: str, after_id: Optional[str]) -> Optional[dict[str, Any]]:
    """Move an active video to just after ``after_id`` (None = front) and return its record.

    None if the id is not an owned, active video. Raises KeyError if ``after_id`` is not on the shelf."""
    with gallery_flags.exclusive(gallery_dir()):
        if owned_video_path(video_id) is None:
            return None
        flags = gallery_flags.read(gallery_dir())
        if gallery_flags.is_archived(flags, video_id):
            return None
        try:
            paths = [
                p
                for p in gallery_dir().glob("*.mp4")
                if not gallery_flags.is_archived(flags, p.stem)
            ]
        except OSError:
            paths = []
        keyed = [(p.stem, _mtime(p)) for p in paths]
        keyed.sort(
            key = lambda pair: (
                gallery_flags.pin_rank(flags, pair[0]),
                gallery_flags.order_rank(flags, pair[0], pair[1]),
            ),
            reverse = True,
        )
        gallery_flags.place_locked(gallery_dir(), video_id, keyed, after_id = after_id)
        meta = _read_meta(_sidecar_path(video_id))
    if meta is None:  # raced a delete between the guard and the read
        return None
    return _record(video_id, meta)


def delete(video_id: str) -> bool:
    """Remove both files of an owned pair; True if the MP4 existed and was ours."""
    path = video_path(video_id)
    if path is None:
        return False
    # Only delete pairs we own; a guessed id must not destroy a foreign MP4.
    if _read_meta(_sidecar_path(video_id)) is None:
        return False
    # Delete the MP4 FIRST: sidecar-first plus a failed unlink leaves an unretryable orphan.
    try:
        path.unlink()
    except OSError as exc:
        logger.warning("video_gallery.delete_failed: %s", exc)
        return False
    try:
        _sidecar_path(video_id).unlink()
    except OSError:
        pass
    gallery_flags.forget(gallery_dir(), [video_id])
    return True


def clear(include_archived: bool = False, *, return_ids: bool = False) -> int | list[str]:
    """Delete owned clips, sparing archived ones unless asked; fail closed if flags are unreadable."""
    removed = 0
    directory = gallery_dir()
    # Hold the flag lock across read-then-delete or a mid-loop archive is deleted after success.
    with gallery_flags.exclusive(directory):
        # Read flags BEFORE listing: nothing is unlinked if the store is untrusted.
        flags = {} if include_archived else gallery_flags.read_trusted(directory)
        try:
            paths = list(directory.glob("*.mp4"))
        except OSError:
            return [] if return_ids else 0
        cleared: list[str] = []
        for path in paths:
            if _read_meta(_sidecar_path(path.stem)) is None:
                continue
            if not include_archived and gallery_flags.is_archived(flags, path.stem):
                continue
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
        # Replace the unreadable store now, or every later default clear still refuses.
        if include_archived and not gallery_flags.is_trusted(directory):
            gallery_flags.reset_locked(directory)
        else:
            gallery_flags.forget_locked(directory, cleared)
    return cleared if return_ids else removed
