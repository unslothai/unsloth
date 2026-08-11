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

from core.inference import gallery_flags
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
    # Stage both files, rename the MP4 in, then the sidecar (the pair's commit marker: list_videos skips an mp4 without a
    # readable sidecar). On any failure remove every artifact, else an invisible, undeletable orphan MP4 is left behind.
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
    # Flags are library state, not recipe: they come from the .flags.json store, never the sidecar.
    return {
        **meta,
        "id": video_id,
        "url": f"/api/inference/video/gallery/{video_id}/file",
        **gallery_flags.flags_for(
            flags if flags is not None else gallery_flags.read(gallery_dir()), video_id
        ),
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


def transcode_to_file(video_id: str, fmt: str) -> Optional[Path]:
    """Re-encode a stored MP4 for the Download menu into a TEMP FILE and return its path, or None
    when the id doesn't resolve. Raises RuntimeError on missing codec/deps (route 501s). The caller
    owns the file and must delete it after serving.

    A file rather than a buffer because the request caps allow 2048x2048 x 1024 frames: a VP9
    export of a clip that size runs to hundreds of MB, and holding it as one ``bytes`` (then again
    in the response) let a couple of concurrent export clicks exhaust the process. The MP4 route
    already streams from disk; this makes the transcodes behave the same way."""
    # Ownership-gate like /file: only transcode a Studio-owned clip (readable sidecar), so a guessed stem for a foreign MP4 cannot be re-encoded out either.
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
            # GIF is already bounded by _GIF_MAX_FRAMES / _GIF_MAX_EDGE, so it is built in memory and written out.
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
            # Realtime settings: VP9's default "good" profile is slow; cpu-used 8 + row-mt is much faster at a small quality cost.
            out_v.options = {"deadline": "realtime", "cpu-used": "8", "row-mt": "1"}
            # An LTX-2 clip carries a synchronized audio track and WebM is the web-embed format, so dropping it would hand back half the
            # result. Opus is WebM's audio codec: resample to its 48 kHz grid and feed whole frames through a FIFO (960 samples per frame).
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

            # Demux both streams together so the muxer sees them interleaved rather than buffering every video packet.
            for packet in src.demux(*([in_v] + ([in_a] if out_a is not None else []))):
                if packet.dts is None:  # flush packet from the demuxer
                    continue
                if packet.stream is in_v:
                    for frame in packet.decode():
                        for out_packet in out_v.encode(frame.reformat(format = "yuv420p")):
                            dst.mux(out_packet)
                    continue
                for frame in packet.decode():
                    for resampled in resampler.resample(frame):
                        # Let the FIFO time the output: the resampler's frames do not line up with Opus' fixed frame size.
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


# Ceilings for a GIF export, which must hold every kept frame in memory before encoding. 720 px and 300 frames (25s at the
# 12 fps target) bound that at roughly 150 MB for the widest clip a generate request allows.
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
            # Full-rate GIFs are huge and stutter; ~12 fps (skipping source frames) is the sweet spot.
            step = max(1, round(rate / 12))
            # Every kept frame is held as a paletted image until the encoder runs, so an unbounded walk is a memory bomb (a 2048x2048 clip of 1024
            # frames is >4 GB). Bound both axes: downscale past _GIF_MAX_EDGE and widen the step to at most _GIF_MAX_FRAMES. MP4 keeps the full clip.
            total = in_v.frames or 0
            kept = (total + step - 1) // step if total else 0
            if kept > _GIF_MAX_FRAMES:
                step = -(-total // _GIF_MAX_FRAMES)
            for i, frame in enumerate(src.decode(in_v)):
                if i % step:
                    continue
                if len(frames) >= _GIF_MAX_FRAMES:
                    # Frame count unknown up front (no stream metadata): stop at the cap.
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


# Sidecar keys every genuine Studio record carries. delete()/clear() own a pair only when its sidecar has all of these, so a
# hand-dropped MP4 with a partial sidecar is neither counted as ours nor destroyed. Key-presence only.
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
        # Invalid UTF-8 is a corrupt sidecar, not a listing failure.
        return None
    try:
        meta = json.loads(raw)
    except (ValueError, TypeError):
        return None
    # A parseable dict is not enough: a foreign ("{}") or different-schema sidecar lacks these keys, and delete()/clear() must never destroy a clip the gallery never surfaced.
    if not isinstance(meta, dict) or any(k not in meta for k in _REQUIRED_META):
        return None
    return meta


def owned_video_path(video_id: str) -> Optional[Path]:
    """Resolve an id to its MP4 only when it is a Studio-owned clip (a readable sidecar), else
    None. The serve and export routes use this instead of video_path() so a guessed stem for a
    hand-dropped/orphan MP4 -- which list_videos/delete/clear already treat as not ours -- can't
    be streamed or transcoded out. Mirrors the delete/clear ownership guard."""
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
    """A window of videos for infinite scroll: pinned first (most recently pinned leading), then
    newest-first by MP4 mtime.

    mtime is a cheap stat ~= generation order; only the window's sidecars are read. limit=None
    returns everything from ``offset`` on. A file without its pair is skipped.

    ``archived`` selects WHICH shelf to page over, it does not widen one: False lists only active
    clips, True lists only archived ones. The archived section needs its own scrollable page.

    ``valid`` (optional) filters records BEFORE pagination, so ``offset`` / ``limit`` and has_more
    all count over the accepted-record domain. Pass the route's schema validator: a sidecar that
    parses as JSON but fails the response schema would otherwise be counted here yet dropped after
    slicing, stalling infinite scroll."""
    try:
        paths = list(gallery_dir().glob("*.mp4"))
    except OSError:
        return []
    flags = gallery_flags.read(gallery_dir())
    # Shelf split and pin sort run on file stems, BEFORE any sidecar is read, so they cost one
    # dict lookup per file and leave the early break below intact.
    paths = [p for p in paths if gallery_flags.is_archived(flags, p.stem) == archived]
    paths.sort(key = lambda p: (gallery_flags.pin_rank(flags, p.stem), _mtime(p)), reverse = True)
    # Page over READABLE records, not raw files: filtering an orphan MP4 out of an already-sliced window would drop valid videos and make has_more wrong.
    want = None if limit is None else offset + limit
    records = []
    for path in paths:
        meta = _read_meta(_sidecar_path(path.stem))
        if meta is None:  # orphan mp4 (no readable sidecar)
            continue
        record = _record(path.stem, meta, flags)
        if valid is not None and not valid(record):  # parses but schema-invalid
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
    not a Studio-owned clip. Ownership-gated like delete: a guessed stem for a hand-dropped or
    orphan MP4 must not become flaggable."""
    # Ownership check and write under one lock, so a concurrent clear cannot delete the pair
    # between them and leave this reporting success for a clip that is already gone.
    with gallery_flags.exclusive(gallery_dir()):
        if owned_video_path(video_id) is None:
            return None
        gallery_flags.set_flags_locked(gallery_dir(), video_id, pinned = pinned, archived = archived)
        meta = _read_meta(_sidecar_path(video_id))
    if meta is None:  # raced a delete between the guard and the read
        return None
    return _record(video_id, meta)


def delete(video_id: str) -> bool:
    """Remove both files of an owned pair; True if the MP4 existed and was ours."""
    path = video_path(video_id)
    if path is None:
        return False
    # Only delete a pair we own (a readable sidecar); a foreign/orphan MP4 is invisible to list_videos, so a guessed id must not destroy it.
    if _read_meta(_sidecar_path(video_id)) is None:
        return False
    # Delete the MP4 FIRST: dropping the sidecar first and failing to unlink the mp4 would leave a clip that vanished from the
    # gallery with no retry. mp4-first leaves at worst an orphan sidecar, which list_videos ignores.
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
    # Drop the flags with the pair, so the id cannot hand a stale pin to anything.
    gallery_flags.forget(gallery_dir(), [video_id])
    return True


def clear(include_archived: bool = False) -> int:
    """Delete Studio-owned gallery pairs (readable sidecar); return how many were removed.

    Archived clips are SPARED by default: archiving is how a user sets something aside, so a
    "clear the gallery" action that destroyed the archive would defeat it. Pass
    include_archived=True to remove those too.

    Raises FlagsUnavailable when the archive has to be spared but the flag store cannot be read.
    Fail CLOSED: read() answers "nothing is archived" for an unreadable store, which here would
    quietly delete the very archive this promises to keep.

    Foreign/orphan MP4s are preserved: list_videos already hides them, so clear must not destroy them."""
    removed = 0
    directory = gallery_dir()
    # Hold the flag lock across the whole read-then-delete: an archive landing mid-loop would
    # otherwise be judged active from the stale snapshot and deleted, after its PATCH had already
    # reported success.
    with gallery_flags.exclusive(directory):
        # Read flags BEFORE listing: nothing is unlinked if the store turns out to be untrusted.
        flags = {} if include_archived else gallery_flags.read_trusted(directory)
        try:
            paths = list(directory.glob("*.mp4"))
        except OSError:
            return 0
        cleared: list[str] = []
        for path in paths:
            if _read_meta(_sidecar_path(path.stem)) is None:  # orphan / not ours
                continue
            if not include_archived and gallery_flags.is_archived(flags, path.stem):
                continue
            # mp4 first; if it can't be unlinked, leave the sidecar so the video stays listable.
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
        # An unreadable store has nothing left to protect once every clip we own is gone, so this
        # is where the escape hatch actually escapes: replace it, or the corrupt file survives the
        # wipe and every later default clear still refuses, new clips included.
        if include_archived and not gallery_flags.is_trusted(directory):
            gallery_flags.reset_locked(directory)
        else:
            gallery_flags.forget_locked(directory, cleared)
    return removed
