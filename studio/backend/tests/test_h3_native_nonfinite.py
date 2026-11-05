# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 native (sd.cpp) refuses a NaN render (flat black/white frames + constant full-scale PCM).

Clips mimic sd.cpp: MJPEG + PCM s16 in AVI (sd-server) and VP8 + PCM s16 in Matroska (sd-cli WebM)."""

import sys
from fractions import Fraction
from pathlib import Path

import pytest

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")

import core.inference.video as video_module  # noqa: E402
from core.inference import video_minimax_h3 as h3_mod  # noqa: E402
from core.inference.video_frames import (  # noqa: E402
    NON_FINITE_FRAMES_MSG,
    NonFiniteFramesError,
)

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

_ENV = "UNSLOTH_VIDEO_NONFINITE_CHECK"
_SIDE, _FRAMES, _FPS, _RATE = 32, 5, 8, 16_000


def _probe(path):
    probe = getattr(h3_mod, "h3_native_clip_is_nan_render", None)
    return False if probe is None else probe(path)


def _write_clip(path: Path, frames, audio) -> Path:
    """``frames``: uint8 (H, W, 3) arrays; ``audio``: int16 mono samples, or None for no soundtrack."""
    is_avi = path.suffix == ".avi"
    with av.open(str(path), mode = "w", format = "avi" if is_avi else "matroska") as output:
        video = output.add_stream("mjpeg" if is_avi else "libvpx", rate = _FPS)
        video.width = video.height = _SIDE
        video.pix_fmt = "yuvj420p" if is_avi else "yuv420p"
        stream = None
        if audio is not None:
            stream = output.add_stream("pcm_s16le", rate = _RATE)
            stream.layout = "mono"
        for pixels in frames:
            frame = av.VideoFrame.from_ndarray(pixels, format = "rgb24")
            for packet in video.encode(frame):
                output.mux(packet)
        if stream is not None:
            frame = av.AudioFrame.from_ndarray(
                np.asarray(audio, dtype = np.int16).reshape(1, -1), format = "s16", layout = "mono"
            )
            frame.sample_rate = _RATE
            frame.pts = 0
            frame.time_base = Fraction(1, _RATE)
            for packet in stream.encode(frame):
                output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        for packet in video.encode():
            output.mux(packet)
    return path


def _flat(value):
    return [np.full((_SIDE, _SIDE, 3), value, dtype = np.uint8) for _ in range(_FRAMES)]


def _dark_noise():
    # A night scene: every pixel 1-3 levels above black. The lossy encode flattens it to almost one value.
    rng = np.random.default_rng(0)
    return [rng.integers(1, 4, (_SIDE, _SIDE, 3), dtype = np.uint8) for _ in range(_FRAMES)]


def _samples(level = None):
    count = _RATE * _FRAMES // _FPS
    if level is not None:
        return np.full(count, level, dtype = np.int16)
    return (1600 * np.sin(2 * np.pi * 220 * np.arange(count) / _RATE)).astype(np.int16)


@pytest.mark.parametrize(
    "suffix, value, level",
    [(".avi", 0, -32767), (".mkv", 0, -32767), (".avi", 255, 32767), (".mkv", 255, 32767)],
    ids = ["cuda-avi", "cuda-webm", "cpu-avi", "cpu-webm"],
)
def test_a_nan_render_is_flagged_in_both_containers(tmp_path, suffix, value, level):
    path = _write_clip(tmp_path / f"nan{suffix}", _flat(value), _samples(level))
    assert _probe(path) is True


@pytest.mark.parametrize("suffix", [".avi", ".mkv"])
@pytest.mark.parametrize(
    "case",
    [
        "dark_noise_with_sound",
        "flat_black_with_sound",
        "flat_black_silent",
        "flat_white_with_sound",
        "dark_noise_full_scale_audio",
        "flat_black_no_audio",
        "flat_black_half_scale_dc",
        "fade_from_black_full_scale_audio",
    ],
)
def test_legitimate_and_ambiguous_clips_are_kept(tmp_path, suffix, case):
    frames, audio = {
        "dark_noise_with_sound": (_dark_noise(), _samples()),
        # A clip that really is black (or saturated to it) keeps its soundtrack, so it is never refused.
        "flat_black_with_sound": (_flat(0), _samples()),
        "flat_black_silent": (_flat(0), _samples(0)),
        "flat_white_with_sound": (_flat(255), _samples()),
        # Only the soundtrack is broken: the frames are real, so this is not the NaN render fingerprint.
        "dark_noise_full_scale_audio": (_dark_noise(), _samples(-32767)),
        "flat_black_no_audio": (_flat(0), None),
        "flat_black_half_scale_dc": (_flat(0), _samples(-16000)),
        "fade_from_black_full_scale_audio": (
            _flat(0)[:2]
            + [np.full((_SIDE, _SIDE, 3), 40 * i, dtype = np.uint8) for i in range(1, 4)],
            _samples(-32767),
        ),
    }[case]
    path = _write_clip(tmp_path / f"clip{suffix}", frames, audio)
    assert _probe(path) is False


def _native_backend(monkeypatch, clip: Path):
    from test_video_backend import _h3_native_backend

    calls: list = []
    backend = _h3_native_backend(monkeypatch, calls)
    monkeypatch.setattr(
        backend._state.pipe.engine, "generate_video", lambda *args, **kwargs: clip, raising = False
    )
    return backend


def test_the_native_route_refuses_a_nan_render(monkeypatch, tmp_path):
    monkeypatch.delenv(_ENV, raising = False)
    clip = _write_clip(tmp_path / "nan.avi", _flat(0), _samples(-32767))
    backend = _native_backend(monkeypatch, clip)

    with pytest.raises(NonFiniteFramesError) as raised:
        backend.generate(prompt = "a fox runs through snow", width = 960, height = 544)

    assert str(raised.value) == NON_FINITE_FRAMES_MSG
    assert NON_FINITE_FRAMES_MSG in video_module.video_failure_detail(raised.value)


def test_the_native_route_keeps_a_dark_clip_and_honours_the_kill_switch(monkeypatch, tmp_path):
    monkeypatch.delenv(_ENV, raising = False)
    dark = _write_clip(tmp_path / "dark.avi", _dark_noise(), _samples())
    result = _native_backend(monkeypatch, dark).generate(prompt = "p", width = 960, height = 544)
    assert result["mp4_bytes"] == b"MP4"

    monkeypatch.setenv(_ENV, "0")
    nan = _write_clip(tmp_path / "nan.avi", _flat(0), _samples(-32767))
    result = _native_backend(monkeypatch, nan).generate(prompt = "p", width = 960, height = 544)
    assert result["mp4_bytes"] == b"MP4"


def test_an_unreadable_clip_never_fails_the_render(monkeypatch, tmp_path):
    monkeypatch.delenv(_ENV, raising = False)
    broken = tmp_path / "broken.avi"
    broken.write_bytes(b"not a container")
    result = _native_backend(monkeypatch, broken).generate(prompt = "p", width = 960, height = 544)
    assert result["mp4_bytes"] == b"MP4"
