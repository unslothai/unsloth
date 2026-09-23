# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rework GGUF chat videos before llama-server samples their frames."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = logging.getLogger(__name__)

# An area, not an edge, so portrait and landscape clips get the same budget.
MAX_FRAME_PIXELS = 640 * 360
# llama-server's --video-fps default: encoding above it buys frames mtmd drops,
# and those bytes count against max_bytes.
MAX_FRAME_RATE = 4
# `fps` places its first output frame half an interval in, so at --video-fps 1 a
# clip under ~0.5s decodes to NO frames. A whole second leaves user-rate headroom.
MIN_SAMPLED_SECONDS = 1.0
_PROBE_TIMEOUT_S = 30
_SHRINK_TIMEOUT_S = 300


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=timeout,
        env=child_env_without_native_path_secret(),
        **windows_hidden_subprocess_kwargs(),
    )


def _frame_geometry(
    ffprobe: str, clip: Path
) -> tuple[Optional[int], Optional[float], Optional[float]]:
    """Any field is None when ffprobe cannot answer; the caller then leaves the clip alone."""
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(clip),
        ],
        _PROBE_TIMEOUT_S,
    )
    if result.returncode != 0:
        return None, None, None
    # A build answering `-of json` with a non-object must fall back, not raise
    # AttributeError past the caller's except clause.
    payload = json.loads(result.stdout or b"{}")
    if not isinstance(payload, dict):
        return None, None, None
    streams = payload.get("streams") or []
    if not streams or not isinstance(streams[0], dict):
        return None, None, None
    width = int(streams[0].get("width") or 0)
    height = int(streams[0].get("height") or 0)
    area = width * height if width > 0 and height > 0 else None
    container = payload.get("format")
    duration = None
    if isinstance(container, dict):
        try:
            duration = float(container.get("duration"))
        except (TypeError, ValueError):
            duration = None
    if duration is None:
        duration = _packet_duration(ffprobe, clip)
    return area, _frame_rate(streams[0].get("avg_frame_rate")), duration


def _packet_duration(ffprobe: str, clip: Path) -> Optional[float]:
    """Duration from packet timestamps, for containers that carry none: a raw
    elementary stream has no `format.duration`, so the sub-second floor never fired
    on one. Bounded to twice the floor, past which a clip is long enough anyway."""
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-read_intervals",
            f"%+{MIN_SAMPLED_SECONDS * 2:g}",
            "-show_entries",
            "packet=pts_time,duration_time",
            "-of",
            "json",
            str(clip),
        ],
        _PROBE_TIMEOUT_S,
    )
    if result.returncode != 0:
        return None
    payload = json.loads(result.stdout or b"{}")
    if not isinstance(payload, dict):
        return None
    end = 0.0
    total = 0.0
    for packet in payload.get("packets") or []:
        if not isinstance(packet, dict):
            continue
        try:
            span = float(packet.get("duration_time"))
        except (TypeError, ValueError):
            span = 0.0
        total += span
        try:
            end = max(end, float(packet.get("pts_time")) + span)
        except (TypeError, ValueError):
            # Annex B carries no timestamps, only durations: summing is the only
            # reading available for a raw stream.
            continue
    return (end or total) or None


def _frame_rate(value: object) -> Optional[float]:
    """ffprobe's `num/den` rate as a float; None when it is absent or 0/0."""
    if not isinstance(value, str) or "/" not in value:
        return None
    numerator, _, denominator = value.partition("/")
    try:
        den = float(denominator)
        return float(numerator) / den if den else None
    except ValueError:
        return None


def _scale_filter(max_pixels: int) -> str:
    # ffmpeg applies display rotation before filters, so use filter-time dimensions.
    factor = f"sqrt({max_pixels}/(iw*ih))"
    return f"scale=w='max(2,trunc(iw*{factor}/2)*2)':h='max(2,trunc(ih*{factor}/2)*2)':flags=area"


def _rate_ceiling(sampled_fps: Optional[float]) -> float:
    """The highest rate worth encoding. A user asking for more than MAX_FRAME_RATE
    raises it: the server samples AFTER this transcode and can only duplicate what
    was dropped here."""
    if sampled_fps is not None and sampled_fps > MAX_FRAME_RATE:
        return float(sampled_fps)
    return float(MAX_FRAME_RATE)


def _filter_chain(
    max_pixels: int,
    source_rate: Optional[float],
    duration: Optional[float],
    oversized: bool,
    sampled_fps: Optional[float] = None,
) -> str:
    """Tail pad, then rate cap, then scale, each only when needed. Order matters:
    the cap shares the blind spot the pad covers (`fps=4` on a 0.1s clip also emits
    nothing, leaving tpad no frame to clone), the cap only goes downwards since
    `fps` DUPLICATES when asked for more than the source has, and scale is skipped
    inside the budget where it would UPSCALE."""
    ceiling = _rate_ceiling(sampled_fps)
    chain = []
    if duration is not None and 0 < duration < MIN_SAMPLED_SECONDS:
        chain.append(f"tpad=stop_mode=clone:stop_duration={MIN_SAMPLED_SECONDS - duration:.3f}")
    if source_rate is not None and source_rate > ceiling:
        chain.append(f"fps={ceiling:g}")
    if oversized:
        chain.append(_scale_filter(max_pixels))
    return ",".join(chain)


def shrink_video_for_llama(
    video_b64: str,
    max_bytes: int,
    max_pixels: int = MAX_FRAME_PIXELS,
    sampled_fps: Optional[float] = None,
) -> str:
    """Make a clip cheap and safe for llama-server to sample, as bare base64.

    ``sampled_fps`` raises the rate ceiling. Keeps only video and the timing of every
    real frame. Missing tools, failures and an over-``max_bytes`` result pass through.
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        return video_b64
    try:
        raw = base64.b64decode(video_b64)
    except (binascii.Error, ValueError):
        return video_b64
    try:
        with tempfile.TemporaryDirectory(prefix="unsloth-video-") as tmp:
            source = Path(tmp) / "clip"
            source.write_bytes(raw)
            area, rate, duration = _frame_geometry(ffprobe, source)
            if area is None:
                return video_b64
            oversized = area > max_pixels
            too_short = duration is not None and 0 < duration < MIN_SAMPLED_SECONDS
            if not oversized and not too_short:
                return video_b64
            # Keeping the decoded copy alive just to log its length costs 64 MiB.
            raw_bytes = len(raw)
            del raw
            shrunk = Path(tmp) / "shrunk.mkv"
            result = _run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-sn",
                    "-dn",
                    "-vf",
                    _filter_chain(max_pixels, rate, duration, oversized, sampled_fps),
                    "-c:v",
                    "mpeg4",
                    # Widely available, but outgrows a low-bitrate source: -fs
                    # bounds at the upload cap, not near the original.
                    "-q:v",
                    "5",
                    "-fs",
                    str(max_bytes),
                    "-f",
                    "matroska",
                    str(shrunk),
                ],
                _SHRINK_TIMEOUT_S,
            )
            if result.returncode != 0 or not shrunk.is_file() or shrunk.stat().st_size == 0:
                logger.warning(
                    "Could not shrink the video clip, forwarding it unchanged: %s",
                    (result.stderr or b"").decode("utf-8", "replace").strip()[-500:],
                )
                return video_b64
            if shrunk.stat().st_size >= max_bytes:
                logger.warning(
                    "The shrunk video clip reached the %d byte cap, forwarding it unchanged",
                    max_bytes,
                )
                return video_b64
            logger.info(
                "Reworked video clip for sampling: %d -> at most %d pixels, "
                "%.2fs -> at least %.2fs (%d to %d bytes)",
                area,
                max_pixels if oversized else area,
                duration if duration is not None else -1.0,
                MIN_SAMPLED_SECONDS if too_short else (duration or -1.0),
                raw_bytes,
                shrunk.stat().st_size,
            )
            return base64.b64encode(shrunk.read_bytes()).decode("ascii")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.warning("Could not shrink the video clip, forwarding it unchanged: %s", exc)
        return video_b64
