# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for Studio's llama-server video frame budget."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from core.inference import llama_video_input  # noqa: E402
from core.inference.llama_cpp import _LLAMA_VIDEO_FPS, _video_fps_flags  # noqa: E402
from core.inference.llama_video_input import (  # noqa: E402
    MAX_FRAME_PIXELS,
    shrink_video_for_llama,
)
from test_llama_flag_catalog import _probe_with_help  # noqa: E402
from test_mmproj_placement_policy import _backend, _launch, _write_gguf  # noqa: E402

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason = "ffmpeg and ffprobe are not on PATH")
_CAP = 64 * 1024 * 1024


def _video_backend(tmp_path, **caps):
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])
    probe = backend.probe_server_capabilities

    def with_caps(_binary = None):
        return {**probe(), **caps}

    backend.probe_server_capabilities = with_caps
    return backend, gguf


def test_a_vision_load_samples_video_at_one_frame_per_second(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_ARG_VIDEO_FPS", raising = False)
    backend, gguf = _video_backend(tmp_path, supports_video_fps = True)

    cmd = _launch(backend, gguf)["cmd"]

    assert cmd.count("--video-fps") == 1
    assert cmd[cmd.index("--video-fps") + 1] == _LLAMA_VIDEO_FPS == "1"


def test_a_hand_typed_frame_rate_still_wins(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_ARG_VIDEO_FPS", raising = False)
    backend, gguf = _video_backend(tmp_path, supports_video_fps = True)

    cmd = _launch(backend, gguf, extra_args = ["--video-fps", "4"])["cmd"]

    positions = [i for i, token in enumerate(cmd) if token == "--video-fps"]
    assert cmd[positions[-1] + 1] == "4"


def test_an_inherited_frame_rate_is_not_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_VIDEO_FPS", "2")
    backend, gguf = _video_backend(tmp_path, supports_video_fps = True)

    assert "--video-fps" not in _launch(backend, gguf)["cmd"]


def test_a_build_without_the_flag_is_not_sent_it(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_ARG_VIDEO_FPS", raising = False)
    backend, gguf = _video_backend(tmp_path)

    assert "--video-fps" not in _launch(backend, gguf)["cmd"]


def _without_studio_projector(tmp_path):
    backend, gguf = _video_backend(tmp_path, supports_video_fps = True)
    backend._resolve_launch_mmproj_path = lambda **_kw: None
    return backend, gguf


def test_a_projector_from_mmproj_auto_is_sampled_at_the_default_rate(tmp_path, monkeypatch):
    """--mmproj itself is denied in the advanced arguments; --mmproj-auto is the
    allowed way to have llama-server find a projector on its own."""
    monkeypatch.delenv("LLAMA_ARG_VIDEO_FPS", raising = False)
    backend, gguf = _without_studio_projector(tmp_path)

    cmd = _launch(backend, gguf, extra_args = ["--mmproj-auto"])["cmd"]

    assert "--mmproj" not in cmd
    assert cmd.index("--video-fps") < cmd.index("--mmproj-auto")
    assert cmd[cmd.index("--video-fps") + 1] == "1"


def test_a_projector_from_the_environment_is_sampled_at_the_default_rate(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_ARG_VIDEO_FPS", raising = False)
    backend, gguf = _without_studio_projector(tmp_path)
    monkeypatch.setenv("LLAMA_ARG_MMPROJ", str(_write_gguf(tmp_path / "env-mmproj.gguf")))

    result = _launch(backend, gguf)

    assert "--mmproj" not in result["cmd"]
    assert result["env"].get("LLAMA_ARG_MMPROJ")
    assert result["cmd"][result["cmd"].index("--video-fps") + 1] == "1"


def test_the_flag_helper_reads_the_environment_it_is_given():
    caps = {"supports_video_fps": True}
    assert _video_fps_flags(caps, env = {}) == ["--video-fps", "1"]
    assert _video_fps_flags(caps, env = {"LLAMA_ARG_VIDEO_FPS": ""}) == []
    assert _video_fps_flags({}, env = {}) == []


def test_the_probe_reads_the_flag_from_help(monkeypatch, tmp_path):
    with_flag = _probe_with_help(
        monkeypatch,
        tmp_path,
        "--video-fps N                           target video frame rate (default: 4.0)\n",
    )
    assert with_flag["supports_video_fps"] is True

    older = tmp_path / "older"
    older.mkdir()
    without = _probe_with_help(
        monkeypatch, older, "--top-k N                               top-k sampling (default: 40)\n"
    )
    assert without["supports_video_fps"] is False


def _clip(
    tmp_path: Path,
    name: str,
    size: str,
    rotation: int = 0,
) -> str:
    path = tmp_path / name
    ffmpeg = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y"]
    subprocess.run(
        [
            *ffmpeg,
            *("-f", "lavfi", "-i", f"testsrc2=size={size}:rate=10", "-t", "2"),
            *("-pix_fmt", "yuv420p", str(path)),
        ],
        check = True,
    )
    if rotation:
        # Match phone orientation metadata without re-encoding.
        rotated = tmp_path / f"rotated-{name}"
        subprocess.run(
            [
                *ffmpeg,
                "-display_rotation",
                str(rotation),
                "-i",
                str(path),
                "-c",
                "copy",
                str(rotated),
            ],
            check = True,
        )
        path = rotated
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _probe(tmp_path: Path, clip_b64: str) -> dict:
    path = tmp_path / "probe-input"
    path.write_bytes(base64.b64decode(clip_b64))
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:stream_side_data=rotation:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check = True,
        capture_output = True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    rotation = [s.get("rotation") for s in stream.get("side_data_list") or [] if "rotation" in s]
    return {
        "width": stream["width"],
        "height": stream["height"],
        "rotation": rotation[0] if rotation else 0,
        "duration": float(data["format"]["duration"]),
    }


@needs_ffmpeg
def test_a_720p_clip_is_shrunk_to_the_frame_budget(tmp_path):
    clip = _clip(tmp_path, "wide.mp4", "1280x720")

    shrunk = shrink_video_for_llama(clip, _CAP)

    info = _probe(tmp_path, shrunk)
    assert info["width"] * info["height"] <= MAX_FRAME_PIXELS
    assert (info["width"], info["height"]) == (640, 360)
    # Sampling stays llama-server's responsibility.
    assert info["duration"] == pytest.approx(2.0, abs = 0.2)


@needs_ffmpeg
def test_a_rotated_phone_clip_comes_out_upright(tmp_path):
    clip = _clip(tmp_path, "portrait.mp4", "1280x720", rotation = 90)
    assert _probe(tmp_path, clip)["rotation"] != 0

    info = _probe(tmp_path, shrink_video_for_llama(clip, _CAP))

    assert (info["width"], info["height"]) == (360, 640)
    assert info["rotation"] == 0


@needs_ffmpeg
def test_a_clip_already_within_the_budget_is_forwarded_untouched(tmp_path):
    clip = _clip(tmp_path, "small.mp4", "640x360")

    assert shrink_video_for_llama(clip, _CAP) is clip


@needs_ffmpeg
def test_a_shrunk_clip_that_outgrows_the_cap_is_forwarded_untouched(tmp_path):
    clip = _clip(tmp_path, "wide.mp4", "1280x720")

    assert shrink_video_for_llama(clip, 20_000) is clip


@needs_ffmpeg
def test_an_undecodable_clip_is_forwarded_untouched(tmp_path):
    clip = base64.b64encode(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64).decode("ascii")

    assert shrink_video_for_llama(clip, _CAP) is clip


def test_a_machine_without_ffmpeg_forwards_the_clip_untouched(monkeypatch):
    monkeypatch.setattr(llama_video_input.shutil, "which", lambda _name: None)
    called = []
    monkeypatch.setattr(llama_video_input, "_run", lambda *a, **k: called.append(a))

    assert shrink_video_for_llama("AAAA", _CAP) == "AAAA"
    assert called == []


def test_a_timed_out_shrink_forwards_the_clip_untouched(monkeypatch):
    monkeypatch.setattr(llama_video_input.shutil, "which", lambda name: f"/bin/{name}")

    def timeout(argv, _timeout):
        raise subprocess.TimeoutExpired(argv, _timeout)

    monkeypatch.setattr(llama_video_input, "_run", timeout)

    assert shrink_video_for_llama("AAAA", _CAP) == "AAAA"


def test_the_gguf_route_shrinks_the_clip_after_the_size_check_and_before_injection():
    source = (Path(__file__).resolve().parent.parent / "routes" / "inference.py").read_text(
        encoding = "utf-8"
    )
    start = source.index('"Video provided but the current GGUF model cannot take video input. "')
    check = source.index("_video_b64_rejection(payload.video_base64)", start)
    shrink = source.index(
        "await asyncio.to_thread(shrink_video_for_llama, video_b64, _MAX_VIDEO_BYTES)", start
    )
    inject = source.index("_inject_video_part(gguf_messages, video_b64)", start)
    assert check < shrink < inject
