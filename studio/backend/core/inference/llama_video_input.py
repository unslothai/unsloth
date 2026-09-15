# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rework GGUF chat videos before llama-server samples their frames.

Smaller frames reduce prompt tokens and projector prefill time, and dropping
the frames the sampler will discard keeps the result inside the upload cap.
A clip too short for the sampler to land on is padded so it still arrives.
Conversion failures leave the clip unchanged.
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
# Studio pins --video-fps 1, and ffmpeg's fps filter places its first output
# frame half an interval in, so a clip shorter than ~0.5s decodes to NO frames
# and the model answers about a video it never saw. Measured against
# llama-server's own `-vf fps=1` call: 0.4s emits 0 frames, 0.49s emits 1. At
# the previous 4 fps default those clips worked, so the floor is a regression
# guard, not a new feature. One whole second, not 0.5, to leave room for the
# rate a user can still pass in the advanced arguments.
MIN_SAMPLED_SECONDS = 1.0
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


def _frame_geometry(
    ffprobe: str, clip: Path
) -> tuple[Optional[int], Optional[float], Optional[float]]:
    """The first video stream's pixel area, average frame rate and duration.

    Any of them may be None when ffprobe cannot answer, which the caller treats
    the same way it treats a failed conversion: leave the clip alone.
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
    # ffprobe answers `-of json` with an object, but a build that hands back
    # anything else must fall back rather than raise AttributeError past the
    # caller's except clause.
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
    return area, _frame_rate(streams[0].get("avg_frame_rate")), duration


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


def _filter_chain(
    max_pixels: int, source_rate: Optional[float], duration: Optional[float], oversized: bool
) -> str:
    """Tail pad, then rate cap, then scale -- each only when it is needed.

    The pad comes FIRST because the rate cap has the same blind spot the pad
    exists to cover: `fps=4` on a 0.1s clip also lands its first frame past the
    end and emits nothing, leaving tpad with no frame to clone and the result
    empty. Padding to a full second first gives both filters something to work
    on. It holds the LAST frame rather than stretching timestamps, so every
    real frame keeps the moment it was shot at and only the tail is invented.

    The rate cap only ever goes downwards: `fps` DUPLICATES frames when asked
    for more than the source has, so a timelapse recorded at 1 fps would come
    back four times heavier.

    Scaling is skipped for a clip already inside the budget, since the factor
    would be greater than one and `scale` would happily UPSCALE it.
    """
    chain = []
    if duration is not None and 0 < duration < MIN_SAMPLED_SECONDS:
        chain.append(f"tpad=stop_mode=clone:stop_duration={MIN_SAMPLED_SECONDS - duration:.3f}")
    if source_rate is not None and source_rate > MAX_FRAME_RATE:
        chain.append(f"fps={MAX_FRAME_RATE}")
    if oversized:
        chain.append(_scale_filter(max_pixels))
    return ",".join(chain)


def shrink_video_for_llama(
    video_b64: str,
    max_bytes: int,
    max_pixels: int = MAX_FRAME_PIXELS,
) -> str:
    """Make a clip cheap and safe for llama-server to sample, as bare base64.

    Shrinks oversized frames, caps the frame rate at the highest rate anything
    will sample, and pads a sub-second clip out far enough that one frame still
    survives the sampler. Keeps only video and preserves the timing of every
    real frame. Missing tools, conversion failures, and a result that would not
    fit in ``max_bytes`` return the input unchanged.
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
            area, rate, duration = _frame_geometry(ffprobe, source)
            if area is None:
                return video_b64
            oversized = area > max_pixels
            too_short = duration is not None and 0 < duration < MIN_SAMPLED_SECONDS
            if not oversized and not too_short:
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
                    _filter_chain(max_pixels, rate, duration, oversized),
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
