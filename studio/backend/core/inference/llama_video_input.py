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


def _frame_area(ffprobe: str, clip: Path) -> Optional[int]:
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(clip),
        ],
        _PROBE_TIMEOUT_S,
    )
    if result.returncode != 0:
        return None
    streams = json.loads(result.stdout or b"{}").get("streams") or []
    if not streams:
        return None
    width = int(streams[0].get("width") or 0)
    height = int(streams[0].get("height") or 0)
    return width * height if width > 0 and height > 0 else None


def _scale_filter(max_pixels: int) -> str:
    # ffmpeg applies display rotation before filters, so use filter-time dimensions.
    factor = f"sqrt({max_pixels}/(iw*ih))"
    return f"scale=w='max(2,trunc(iw*{factor}/2)*2)':h='max(2,trunc(ih*{factor}/2)*2)':flags=area"


def shrink_video_for_llama(
    video_b64: str,
    max_bytes: int,
    max_pixels: int = MAX_FRAME_PIXELS,
) -> str:
    """Shrink oversized frames and return the clip as bare base64.

    Keeps only video and preserves timing. Missing tools, conversion failures,
    and a result that would not fit in ``max_bytes`` return the input unchanged.
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
            area = _frame_area(ffprobe, source)
            if area is None or area <= max_pixels:
                return video_b64
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
                    _scale_filter(max_pixels),
                    "-c:v",
                    "mpeg4",
                    # Widely available, but can outgrow a low-bitrate source, hence -fs.
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
                len(raw),
                shrunk.stat().st_size,
            )
            return base64.b64encode(shrunk.read_bytes()).decode("ascii")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.warning("Could not shrink the video clip, forwarding it unchanged: %s", exc)
        return video_b64
