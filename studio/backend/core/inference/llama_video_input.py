# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resize GGUF chat videos before llama-server samples their frames.

Smaller frames reduce prompt tokens and projector prefill time. Conversion
failures leave the clip unchanged.
"""

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
# llama-server's own --video-fps default. Re-encoding above the rate anything
# will sample spends bytes on frames mtmd drops, and those bytes count against
# max_bytes: a legal 53 MB 12-minute 1080p30 upload transcodes to over 64 MB,
# trips the cap guard below and is forwarded at full size after a 25s ffmpeg
# pass. Capping at llama-server's default keeps every configuration whole --
# Studio asks for 1 fps, a build too old for --video-fps samples at 4.
MAX_FRAME_RATE = 4
_PROBE_TIMEOUT_S = 30
_SHRINK_TIMEOUT_S = 300


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        stdin = subprocess.DEVNULL,
        capture_output = True,
        timeout = timeout,
        env = child_env_without_native_path_secret(),
        **windows_hidden_subprocess_kwargs(),
    )


def _frame_geometry(ffprobe: str, clip: Path) -> tuple[Optional[int], Optional[float]]:
    """The first video stream's pixel area and its average frame rate.

    Either may be None when ffprobe cannot answer, which the caller treats the
    same way it treats a failed conversion: leave the clip alone.
    """
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "json",
            str(clip),
        ],
        _PROBE_TIMEOUT_S,
    )
    if result.returncode != 0:
        return None, None
    # ffprobe answers `-of json` with an object, but a build that hands back
    # anything else must fall back rather than raise AttributeError past the
    # caller's except clause.
    payload = json.loads(result.stdout or b"{}")
    streams = payload.get("streams") or [] if isinstance(payload, dict) else []
    if not streams or not isinstance(streams[0], dict):
        return None, None
    width = int(streams[0].get("width") or 0)
    height = int(streams[0].get("height") or 0)
    area = width * height if width > 0 and height > 0 else None
    return area, _frame_rate(streams[0].get("avg_frame_rate"))


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


def _filter_chain(max_pixels: int, source_rate: Optional[float]) -> str:
    """Drop to MAX_FRAME_RATE before scaling, but only ever downwards.

    `fps` DUPLICATES frames when asked for more than the source has, so a
    timelapse recorded at 1 fps would come back four times heavier. Only cap a
    rate we have measured to be above the ceiling.
    """
    scale = _scale_filter(max_pixels)
    if source_rate is not None and source_rate > MAX_FRAME_RATE:
        return f"fps={MAX_FRAME_RATE},{scale}"
    return scale


def shrink_video_for_llama(
    video_b64: str,
    max_bytes: int,
    max_pixels: int = MAX_FRAME_PIXELS,
) -> str:
    """Shrink oversized frames and return the clip as bare base64.

    Keeps only video, preserves timing, and caps the frame rate at the highest
    rate llama-server will sample. Missing tools, conversion failures, and a
    result that would not fit in ``max_bytes`` return the input unchanged.
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
        with tempfile.TemporaryDirectory(prefix = "unsloth-video-") as tmp:
            source = Path(tmp) / "clip"
            source.write_bytes(raw)
            area, rate = _frame_geometry(ffprobe, source)
            if area is None or area <= max_pixels:
                return video_b64
            # The source bytes are on disk now, and holding the decoded copy to
            # the end just to log its length keeps 64 MiB alive alongside the
            # output and both base64 strings.
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
                    _filter_chain(max_pixels, rate),
                    "-c:v",
                    "mpeg4",
                    # Widely available, but still outgrows a low-bitrate source:
                    # -fs only bounds the result at the upload cap, it does not
                    # keep it near the original size.
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
                "Shrunk video clip frames from %d to at most %d pixels (%d to %d bytes)",
                area,
                max_pixels,
                raw_bytes,
                shrunk.stat().st_size,
            )
            return base64.b64encode(shrunk.read_bytes()).decode("ascii")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.warning("Could not shrink the video clip, forwarding it unchanged: %s", exc)
        return video_b64
