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
from fractions import Fraction
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
# A frame PNG is decoded and encoded whole, so refuse anything past 8K square.
_FRAME_MAX_PIXELS = 8192 * 8192


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
        # The server's own sort key, so a client re-sort agrees with it (created_at can differ).
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
    """First frame as WebP; forced container reads only the stream. max_pixels refuses before decode."""
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


def _frame_time(frame) -> float:
    return float(frame.time) if frame.time is not None else 0.0


def frame_png(
    source: Path,
    at: str = "first",
    max_px: Optional[int] = None,
) -> bytes:
    """One frame of a clip as PNG: ``first``, ``last`` or the frame shown at ``at`` seconds,
    scaled to fit ``max_px`` when given. ValueError for a bad or past-end ``at`` or a clip PyAV
    cannot decode; FileNotFoundError when the file is gone."""
    import io
    import math

    text = str(at).strip().lower()
    seconds: Optional[float] = None
    if text not in ("first", "last"):
        try:
            seconds = float(text)
        except ValueError:
            seconds = None
        if seconds is None or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("at must be 'first', 'last' or a time in seconds.")
    try:
        import av
        from PIL import Image
    except Exception as exc:  # noqa: BLE001 -- a missing decoder dependency
        raise RuntimeError("Frame extraction needs the 'av' and 'Pillow' packages.") from exc
    try:
        with av.open(str(source)) as src:
            if not src.streams.video:
                raise ValueError("This clip has no video stream.")
            stream = src.streams.video[0]
            if stream.width * stream.height > _FRAME_MAX_PIXELS:
                raise ValueError(f"{stream.width}x{stream.height} is too large to extract.")
            duration = (src.duration or 0) / 1_000_000
            if not duration and stream.duration is not None and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            # Frame times and seeks are absolute; ``at`` counts from the stream's first timestamp.
            origin = float(stream.start_time * stream.time_base) if stream.start_time else 0.0
            frame = None
            if text == "first":
                frame = next(src.decode(stream), None)
            elif seconds is not None:
                if duration and seconds > duration:
                    raise ValueError(
                        f"at={seconds:g} is past the end of the clip ({duration:.2f} s)."
                    )
                src.seek(int((origin + seconds) / stream.time_base), stream = stream, backward = True)
                for candidate in src.decode(stream):
                    if frame is not None and _frame_time(candidate) > origin + seconds + 1e-6:
                        break
                    frame = candidate
            else:
                # Seek close to the end and keep the final frame; a failed seek falls back to a full decode.
                for start in (max(0.0, duration - 2.0), 0.0):
                    src.seek(int((origin + start) / stream.time_base), stream = stream, backward = True)
                    for candidate in src.decode(stream):
                        frame = candidate
                    if frame is not None or start == 0.0:
                        break
            if frame is None:
                raise ValueError("This clip has no decodable frame there.")
            image = frame.to_image()
    except FileNotFoundError:
        raise
    except av.error.FFmpegError:
        raise ValueError("This clip could not be decoded.") from None
    if max_px:
        image.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    # Pillow's default level on purpose: MCP callers gate on these bytes (INLINE_CAP, MAX_IMAGE_BYTES).
    image.save(buf, format = "PNG")
    return buf.getvalue()


def transcode_to_file(video_id: str, fmt: str) -> Optional[Path]:
    """Re-encodes to a temp file, not memory (concurrent exports exhaust RAM); caller deletes after serving; None if id unresolved; RuntimeError without codec"""
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
            # Keep LTX-2 audio (dropping it halves the WebM result); Opus needs 48 kHz and whole 960-sample frames via a FIFO
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
                # frame_size is 0 until the container starts writing; 960 is libopus' own frame.
                size = out_a.frame_size or 960
                while True:
                    frame = fifo.read(size, partial = flush)
                    if frame is None:
                        break
                    for packet in out_a.encode(frame):
                        dst.mux(packet)

            # demux both streams together so the muxer sees them interleaved rather than buffering every video packet
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


_GIF_MAX_EDGE = 720
_GIF_MAX_FRAMES = 300
_GIF_PALETTEGEN_ARGS = "reserve_transparent=on:transparency_color=ffffff"
_GIF_PALETTEUSE_ARGS = "dither=sierra2_4a"
# palettegen needs every frame before paletteuse; above this budget decode twice instead of holding frames.
_GIF_HOLD_BYTES = 160 * 1024 * 1024


def _gif_plan(in_v) -> tuple[Fraction, int, int, int]:
    """(source fps, frame step, output width, output height) for a GIF of ``in_v``."""
    rate = Fraction(in_v.average_rate or 24)
    step = max(1, round(float(rate) / 12))
    total = in_v.frames or 0
    kept = (total + step - 1) // step if total else 0
    if kept > _GIF_MAX_FRAMES:
        step = -(-total // _GIF_MAX_FRAMES)
    width, height = in_v.codec_context.width, in_v.codec_context.height
    if max(width, height) > _GIF_MAX_EDGE:
        scale = _GIF_MAX_EDGE / max(width, height)
        width, height = max(1, round(width * scale)), max(1, round(height * scale))
    return rate, step, width, height


def _gif_scaled_frames(path: Path, step: int, width: int, height: int):
    import av
    with av.open(str(path)) as src:
        in_v = src.streams.video[0]
        in_v.thread_type = "AUTO"
        kept = 0
        for i, frame in enumerate(src.decode(in_v)):
            if i % step:
                continue
            if kept >= _GIF_MAX_FRAMES:
                break
            kept += 1
            yield frame.reformat(
                width = width, height = height, format = "rgb24", interpolation = "LANCZOS"
            )


def _gif_palette_supported() -> bool:
    try:
        import av
        import av.filter

        filters = av.filter.filters_available
        if "palettegen" not in filters or "paletteuse" not in filters:
            return False
        # codecs_available lists a codec usable in either direction; only an encoder writes the GIF
        av.codec.Codec("gif", "w")
        return True
    except Exception:  # noqa: BLE001 -- an old or partial build: use the Pillow path
        return False


def _drain_sink(sink) -> list[Any]:
    import av
    out = []
    while True:
        try:
            out.append(sink.pull())
        except (av.error.BlockingIOError, av.error.EOFError):
            return out


def _transcode_gif_palette(
    path: Path, rate: Fraction, step: int, width: int, height: int, kept: int
) -> bytes:
    """GIF via one clip-wide palette (palettegen + paletteuse); per-frame quantizing was slow and flickered."""
    import io

    import av
    import av.filter

    hold = kept > 0 and kept * width * height * 3 <= _GIF_HOLD_BYTES
    held: Optional[list[Any]] = [] if hold else None
    time_base = Fraction(1, 100)

    graph = av.filter.Graph()
    buffer = graph.add_buffer(width = width, height = height, format = "rgb24", time_base = time_base)
    palettegen = graph.add("palettegen", _GIF_PALETTEGEN_ARGS)
    sink = graph.add("buffersink")
    buffer.link_to(palettegen)
    palettegen.link_to(sink)
    graph.configure()
    count = 0
    for frame in _gif_scaled_frames(path, step, width, height):
        frame.pts = count
        frame.time_base = time_base
        buffer.push(frame)
        # palettegen emits only at EOF, but pulling lets FFmpeg consume each frame instead of queueing the whole clip
        _drain_sink(sink)
        if held is not None:
            held.append(frame)
        count += 1
    if not count:
        raise RuntimeError("GIF export decoded no frames.")
    buffer.push(None)
    palettes = _drain_sink(sink)
    if not palettes:
        raise RuntimeError("GIF export failed: palettegen produced no palette.")
    palette = palettes[-1]

    graph = av.filter.Graph()
    frames_in = graph.add_buffer(width = width, height = height, format = "rgb24", time_base = time_base)
    palette_in = graph.add_buffer(
        width = palette.width, height = palette.height, format = palette.format.name, time_base = time_base
    )
    paletteuse = graph.add("paletteuse", _GIF_PALETTEUSE_ARGS)
    sink = graph.add("buffersink")
    frames_in.link_to(paletteuse, 0, 0)
    palette_in.link_to(paletteuse, 0, 1)
    paletteuse.link_to(sink)
    graph.configure()
    palette.pts = 0
    palette.time_base = time_base
    palette_in.push(palette)
    palette_in.push(None)

    # Frame k at pts k on the rate/step clock; the muxer derives centisecond delays from pts gaps (no drift).
    out_rate = rate / step
    buf = io.BytesIO()
    with av.open(buf, "w", format = "gif", options = {"loop": "0"}) as dst:
        stream = dst.add_stream("gif", rate = out_rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "pal8"
        written = 0

        def _emit(frames: list[Any]) -> None:
            nonlocal written
            for out in frames:
                out.pts = written
                out.time_base = 1 / out_rate
                written += 1
                for packet in stream.encode(out):
                    packet.duration = 1
                    dst.mux(packet)

        source = held if held is not None else _gif_scaled_frames(path, step, width, height)
        for k, frame in enumerate(source):
            if k >= count:
                break
            frame.pts = k
            frame.time_base = time_base
            frames_in.push(frame)
            _emit(_drain_sink(sink))
        frames_in.push(None)
        _emit(_drain_sink(sink))
        for packet in stream.encode():
            packet.duration = 1
            dst.mux(packet)
    return buf.getvalue()


def _transcode_gif_pillow(path: Path, rate: Fraction, step: int, width: int, height: int) -> bytes:
    """Fallback for a PyAV build without palettegen / paletteuse / the GIF encoder: Pillow, one palette per frame."""
    import io

    from PIL import Image

    frames = [
        frame.to_image().convert("P", palette = Image.Palette.ADAPTIVE)
        for frame in _gif_scaled_frames(path, step, width, height)
    ]
    if not frames:
        raise RuntimeError("GIF export decoded no frames.")
    duration_ms = max(20, int(1000 * step / float(rate)))
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


def _transcode_gif(path: Path) -> bytes:
    try:
        import av
        from PIL import Image  # noqa: F401 -- the fallback encoder; checked up front so the error names it
    except Exception as exc:  # noqa: BLE001 -- missing deps -> no transcode
        raise RuntimeError("GIF export needs the 'av' and 'Pillow' packages.") from exc
    try:
        with av.open(str(path)) as src:
            if not src.streams.video:
                raise RuntimeError("GIF export failed: the clip has no video stream.")
            in_v = src.streams.video[0]
            rate, step, width, height = _gif_plan(in_v)
            total = in_v.frames or 0
        kept = min(_GIF_MAX_FRAMES, (total + step - 1) // step) if total else 0
        if _gif_palette_supported():
            return _transcode_gif_palette(path, rate, step, width, height, kept)
        return _transcode_gif_pillow(path, rate, step, width, height)
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 -- surface as "decoder unavailable"
        raise RuntimeError(f"GIF export failed to decode the clip: {exc}") from exc


def _sidecar_path(video_id: str) -> Path:
    return gallery_dir() / f"{video_id}.json"


# Keys every genuine Unsloth sidecar carries; delete()/clear() own only pairs with all, so a partial hand-dropped MP4 is never destroyed. Key-presence only.
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
    """Pinned then newest-first; archived selects one shelf, does not widen it; valid filters run before slicing so has_more counts accepted clips"""
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
    """Ownership check and write under one lock so a concurrent clear cannot delete the pair and leave this reporting success for a gone clip."""
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
        # Full shelf in listing order, so neighbours past the client's loaded window are known.
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
    # Drop the flags with the pair, so the id cannot hand a stale pin to anything.
    gallery_flags.forget(gallery_dir(), [video_id])
    return True


def clear(include_archived: bool = False, *, return_ids: bool = False) -> int | list[str]:
    """Delete owned clips; archived spared unless include_archived=True; raises FlagsUnavailable when archive must be spared but flags unreadable (fail closed)."""
    removed = 0
    directory = gallery_dir()
    # Hold the flag lock across read-then-delete or a mid-loop archive is deleted after success.
    with gallery_flags.exclusive(directory):
        # read flags BEFORE listing: nothing is unlinked if the store turns out to be untrusted
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
        # Once every owned clip is gone nothing is left to protect, so replace an unreadable store now or every later default clear still refuses.
        if include_archived and not gallery_flags.is_trusted(directory):
            gallery_flags.reset_locked(directory)
        else:
            gallery_flags.forget_locked(directory, cleared)
    return cleared if return_ids else removed
