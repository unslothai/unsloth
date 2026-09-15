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
    MAX_FRAME_RATE,
    MIN_SAMPLED_SECONDS,
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
    rate: int = 10,
    seconds: int = 2,
) -> str:
    path = tmp_path / name
    ffmpeg = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y"]
    subprocess.run(
        [
            *ffmpeg,
            *("-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}"),
            *("-t", str(seconds)),
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


def _frame_count(tmp_path: Path, clip_b64: str, name: str) -> int:
    path = tmp_path / name
    path.write_bytes(base64.b64decode(clip_b64))
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check = True,
        capture_output = True,
    )
    return int(json.loads(result.stdout)["streams"][0]["nb_read_frames"])


@needs_ffmpeg
def test_frames_beyond_the_sampled_rate_are_not_re_encoded(tmp_path):
    # Re-encoding at the source rate spends the byte budget on frames mtmd drops,
    # which pushed a long clip over the cap and threw the conversion away.
    clip = _clip(tmp_path, "fast.mp4", "1280x720", rate = 30, seconds = 2)
    assert _frame_count(tmp_path, clip, "fast-source") == 60

    shrunk = shrink_video_for_llama(clip, _CAP)

    assert _frame_count(tmp_path, shrunk, "fast-shrunk") == MAX_FRAME_RATE * 2
    # Same span, so llama-server samples the same moments it would have before.
    assert _probe(tmp_path, shrunk)["duration"] == pytest.approx(2.0, abs = 0.3)


@needs_ffmpeg
def test_a_clip_slower_than_the_cap_keeps_every_frame(tmp_path):
    # `fps` DUPLICATES when asked for more than the source has, so a timelapse
    # must not be padded up to the ceiling.
    clip = _clip(tmp_path, "slow.mp4", "1280x720", rate = 2, seconds = 3)
    assert _frame_count(tmp_path, clip, "slow-source") == 6

    shrunk = shrink_video_for_llama(clip, _CAP)

    assert _frame_count(tmp_path, shrunk, "slow-shrunk") == 6


@needs_ffmpeg
def test_a_long_clip_still_fits_the_cap_and_is_really_shrunk(tmp_path):
    # The regression that motivated the cap: a source-rate re-encode outgrows
    # max_bytes, so the guard forwarded the original after paying to convert it.
    clip = _clip(tmp_path, "long.mp4", "1280x720", rate = 30, seconds = 30)

    shrunk = shrink_video_for_llama(clip, _CAP)

    assert shrunk is not clip
    assert len(base64.b64decode(shrunk)) < len(base64.b64decode(clip))
    assert _probe(tmp_path, shrunk)["width"] == 640


def _frames_at(tmp_path: Path, clip_b64: str, name: str, fps: int) -> int:
    """Frames llama-server would actually get, via the same `-vf fps=` it runs."""
    path = tmp_path / name
    path.write_bytes(base64.b64decode(clip_b64))
    # Dimensions only: a raw stream has no container duration, so _probe would
    # KeyError before the frames could be counted.
    dims = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            *("-show_entries", "stream=width,height"),
            *("-of", "json", str(path)),
        ],
        check = True,
        capture_output = True,
    )
    stream = json.loads(dims.stdout)["streams"][0]
    raw = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vf",
            f"fps={fps}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check = True,
        capture_output = True,
    ).stdout
    return len(raw) // (stream["width"] * stream["height"] * 3)


@needs_ffmpeg
@pytest.mark.parametrize("seconds", [0.1, 0.3, 0.4])
def test_a_sub_second_clip_still_reaches_the_model(tmp_path, seconds):
    # At the 1 fps Studio pins, anything under ~0.5s decodes to NOTHING. These
    # worked at the old 4 fps default, so this is a regression guard.
    clip = _clip(tmp_path, f"tiny{seconds}.mp4", "320x180", rate = 30, seconds = seconds)
    assert _frames_at(tmp_path, clip, f"tiny-src{seconds}", 1) == 0

    shrunk = shrink_video_for_llama(clip, _CAP)

    assert _frames_at(tmp_path, shrunk, f"tiny-out{seconds}", 1) >= 1
    assert _probe(tmp_path, shrunk)["duration"] == pytest.approx(MIN_SAMPLED_SECONDS, abs = 0.15)


@needs_ffmpeg
def test_a_long_enough_clip_within_the_budget_is_still_left_alone(tmp_path):
    clip = _clip(tmp_path, "ok.mp4", "640x360", rate = 10, seconds = 2)

    assert shrink_video_for_llama(clip, _CAP) is clip


def test_a_probe_that_is_not_an_object_forwards_the_clip_untouched(monkeypatch):
    # A build answering `-of json` with a non-object must fall back, not raise
    # AttributeError past the caller's except.
    class _Result:
        returncode = 0
        stdout = b"[]"
        stderr = b""

    monkeypatch.setattr(llama_video_input, "_run", lambda *a, **k: _Result())
    monkeypatch.setattr(llama_video_input.shutil, "which", lambda name: f"/usr/bin/{name}")

    assert shrink_video_for_llama("AAAA", _CAP) == "AAAA"


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
    # Bare name: the call has been reflowed twice, and each time an anchor on the
    # full call raised ValueError instead of reporting an ordering problem.
    shrink = source.index("shrink_video_for_llama,", start)
    inject = source.index("_inject_video_part(gguf_messages, video_b64)", start)
    assert check < shrink < inject


def test_the_conversions_result_is_what_gets_injected():
    """The ordering test above matches source text, so it cannot see whether the
    conversion's RESULT is used. Dropping just the assignment silently disables
    the whole feature -- the clip is still converted, then thrown away and the
    original injected -- while every substring it searches for stays put.

    Read the structure instead: the awaited call must be bound to a name, and
    that same name must be the one handed to _inject_video_part.
    """
    import ast

    source = (Path(__file__).resolve().parent.parent / "routes" / "inference.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(source)

    def is_shrink_call(node) -> bool:
        call = node.value if isinstance(node, ast.Await) else node
        if not isinstance(call, ast.Call):
            return False
        return any(isinstance(a, ast.Name) and a.id == "shrink_video_for_llama" for a in call.args)

    bound_to = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and is_shrink_call(node.value)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert bound_to, (
        "shrink_video_for_llama's result is not assigned to anything: the clip "
        "is converted and then discarded, leaving the original to be injected"
    )

    injected = {
        arg.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_inject_video_part"
        for arg in node.args
        if isinstance(arg, ast.Name)
    }
    assert bound_to & injected, (
        f"the shrunk clip is bound to {sorted(bound_to)} but _inject_video_part "
        f"is given {sorted(injected)}"
    )


def test_the_route_hands_the_helper_bare_base64_with_any_data_header_gone():
    """`shrink_video_for_llama` documents a bare-base64 contract, and the route
    is what has to honour it. A data: URL reaching ffmpeg would decode to
    garbage and be forwarded unchanged, losing the speedup silently.
    """
    import routes.inference as inference_route

    payload = base64.b64encode(b"not-really-a-clip").decode("ascii")

    bare, rejection = inference_route._video_b64_rejection(f"data:video/mp4;base64,{payload}")
    assert rejection is None
    assert bare == payload

    already_bare, rejection = inference_route._video_b64_rejection(payload)
    assert rejection is None
    assert already_bare == payload


def test_an_oversize_upload_is_refused_before_any_conversion_runs():
    """The 413 has to come first: spawning ffmpeg for a clip the route is about
    to reject burns a worker thread and up to the shrink timeout for nothing.
    """
    import routes.inference as inference_route

    too_big = "A" * (inference_route._MAX_VIDEO_B64_CHARS + 4)

    _, rejection = inference_route._video_b64_rejection(too_big)

    assert rejection is not None
    assert rejection[0] == 413


def test_the_requested_rate_is_read_from_the_extras_and_the_environment():
    from core.inference.llama_cpp import requested_video_fps

    assert requested_video_fps(None, env = {}) is None
    assert requested_video_fps(["--video-fps", "8"], env = {}) == 8
    assert requested_video_fps(["--video-fps=8"], env = {}) == 8
    # llama.cpp resolves a repeated flag last-wins, so the encoder must agree.
    assert requested_video_fps(["--video-fps", "8", "--video-fps", "2"], env = {}) == 2
    # argv beats the environment: llama.cpp reads the env var when it registers
    # the option, and argv overrides it afterwards (measured on b10976).
    assert requested_video_fps(["--video-fps", "8"], env = {"LLAMA_ARG_VIDEO_FPS": "1"}) == 8
    assert requested_video_fps([], env = {"LLAMA_ARG_VIDEO_FPS": "8"}) == 8
    # llama.cpp accepts the underscore spelling too, so an exact match on the
    # hyphen would silently miss it.
    assert requested_video_fps(["--video_fps", "8"], env = {}) == 8
    assert requested_video_fps(["--video_fps=8"], env = {}) == 8
    assert requested_video_fps(["--video_fps", "8", "--video-fps", "2"], env = {}) == 2
    assert requested_video_fps(["--video-fps", "abc"], env = {}) is None
    assert requested_video_fps(["--video-fps", "0"], env = {}) is None
    assert requested_video_fps(["--video-fps"], env = {}) is None


def test_a_higher_requested_rate_raises_the_encode_ceiling():
    # The server samples AFTER this transcode, so a dropped rate can only come
    # back as duplicates. Below the default the ceiling holds.
    chain = llama_video_input._filter_chain(MAX_FRAME_PIXELS, 30.0, 5.0, True, 8.0)
    assert "fps=8" in chain
    assert llama_video_input._rate_ceiling(None) == MAX_FRAME_RATE
    assert llama_video_input._rate_ceiling(1.0) == MAX_FRAME_RATE
    assert llama_video_input._rate_ceiling(2.0) == MAX_FRAME_RATE
    assert llama_video_input._rate_ceiling(24.0) == 24.0


@needs_ffmpeg
def test_an_eight_fps_override_keeps_eight_fps_of_frames(tmp_path):
    clip = _clip(tmp_path, "fast8.mp4", "1280x720", rate = 30, seconds = 2)

    default = shrink_video_for_llama(clip, _CAP)
    override = shrink_video_for_llama(clip, _CAP, sampled_fps = 8.0)

    assert _frame_count(tmp_path, default, "d8") == MAX_FRAME_RATE * 2
    assert _frame_count(tmp_path, override, "o8") == 8 * 2


@needs_ffmpeg
def test_a_short_clip_with_no_container_duration_is_still_padded(tmp_path):
    # A raw stream carries no format.duration, so the sub-second floor never
    # fired. Annex B has no packet timestamps either, only durations, so the
    # fallback has to sum those rather than read a presentation time.
    path = tmp_path / "raw.h264"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            *("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30"),
            *("-t", "0.3", "-c:v", "libx264", "-bsf:v", "h264_mp4toannexb"),
            *("-f", "h264", str(path)),
        ],
        check = True,
    )
    clip = base64.b64encode(path.read_bytes()).decode("ascii")
    ffprobe = shutil.which("ffprobe")
    assert llama_video_input._frame_geometry(ffprobe, path)[2] == pytest.approx(0.3, abs = 0.05)
    assert _frames_at(tmp_path, clip, "raw-src", 1) == 0

    shrunk = shrink_video_for_llama(clip, _CAP)

    assert _frames_at(tmp_path, shrunk, "raw-out", 1) >= 1


@needs_ffmpeg
def test_the_packet_fallback_is_only_consulted_when_the_container_is_silent(tmp_path, monkeypatch):
    # It reads packets, so a long ordinary clip must never reach it.
    called = []
    real = llama_video_input._packet_duration
    monkeypatch.setattr(
        llama_video_input,
        "_packet_duration",
        lambda *a, **k: called.append(1) or real(*a, **k),
    )
    clip = _clip(tmp_path, "ordinary.mp4", "1280x720", rate = 30, seconds = 2)

    shrink_video_for_llama(clip, _CAP)

    assert not called
