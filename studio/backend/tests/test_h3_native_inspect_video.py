# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""H3 native ``inspect_video`` counts sd-cli WebM from VP8 headers and otherwise equals a full decode."""

from fractions import Fraction
from pathlib import Path

import pytest

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")

from core.inference import video_minimax_h3 as h3_mod  # noqa: E402

_W, _H, _FPS, _RATE = 64, 48, 8, 16_000


def _clip(
    path: Path,
    frames: int,
    audio: bool = False,
    codec: str = "libvpx",
    gop: int = 1,
) -> Path:
    rng = np.random.default_rng(0)
    with av.open(str(path), mode = "w", format = "matroska") as output:
        video = output.add_stream(codec, rate = _FPS)
        video.width, video.height, video.pix_fmt = _W, _H, "yuv420p"
        video.codec_context.gop_size = gop
        sound = None
        if audio:
            sound = output.add_stream("pcm_s16le", rate = _RATE)
            sound.layout = "mono"
        for _ in range(frames):
            pixels = rng.integers(0, 255, (_H, _W, 3), dtype = np.uint8)
            for packet in video.encode(av.VideoFrame.from_ndarray(pixels, format = "rgb24")):
                output.mux(packet)
        for packet in video.encode():
            output.mux(packet)
        if sound is not None:
            frame = av.AudioFrame.from_ndarray(
                np.zeros((1, _RATE * frames // _FPS), dtype = np.int16), format = "s16", layout = "mono"
            )
            frame.sample_rate, frame.pts, frame.time_base = _RATE, 0, Fraction(1, _RATE)
            for packet in sound.encode(frame):
                output.mux(packet)
            for packet in sound.encode():
                output.mux(packet)
    return path


def _decoded(path: Path):
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            frames = int(stream.frames or 0)
            out = (
                int(stream.codec_context.width),
                int(stream.codec_context.height),
                frames if frames > 0 else sum(1 for _ in container.decode(video = 0)),
                bool(container.streams.audio),
            )
        return out
    except Exception as exc:  # noqa: BLE001
        return type(exc)


def _inspected(path: Path):
    try:
        return h3_mod.inspect_video(path)
    except Exception as exc:  # noqa: BLE001
        return type(exc)


@pytest.fixture
def decodes(monkeypatch):
    calls = []
    real_open = av.open

    class _Counted:
        def __init__(self, container):
            self._container = container

        def __getattr__(self, name):
            return getattr(self._container, name)

        def __enter__(self):
            self._container.__enter__()
            return self

        def __exit__(self, *exc):
            return self._container.__exit__(*exc)

        def decode(self, *args, **kwargs):
            calls.append(args or kwargs)
            return self._container.decode(*args, **kwargs)

    monkeypatch.setattr(av, "open", lambda *a, **k: _Counted(real_open(*a, **k)))
    return calls


def _edit_first_frame_tag(
    path: Path,
    out: Path,
    edit,
    at: int = -3,
) -> Path:
    """Edit one byte of the middle key frame: its frame tag by default, ``at = 0`` its start code."""
    data = bytearray(path.read_bytes())
    start = data.find(b"\x9d\x01\x2a", len(data) // 2)
    assert start > 3
    data[start + at] = edit(data[start + at])
    out.write_bytes(bytes(data))
    return out


@pytest.mark.parametrize("audio", [False, True])
@pytest.mark.parametrize("frames", [1, 9])
def test_sd_cli_webm_is_counted_without_decoding(tmp_path, decodes, frames, audio):
    path = _clip(tmp_path / "generated.webm", frames, audio)
    want = _decoded(path)
    decodes.clear()
    assert h3_mod.inspect_video(path) == want == (_W, _H, frames, audio)
    assert decodes == []


def test_a_hidden_frame_is_not_counted(tmp_path, decodes):
    # show_frame = 0 decodes to nothing: a packet count is one too many
    path = _edit_first_frame_tag(
        _clip(tmp_path / "a.webm", 9), tmp_path / "hidden.webm", lambda b: b & ~0x10
    )
    assert _decoded(path) == (_W, _H, 8, False)
    decodes.clear()
    assert h3_mod.inspect_video(path) == (_W, _H, 8, False)
    assert len(decodes) == 1


@pytest.mark.parametrize(
    "edit, at",
    [(lambda b: b | 0x01, -3), (lambda b: b | 0xE0, -3), (lambda b: 0, 0)],
    ids = ["inter-frame-flag", "partition-size", "start-code"],
)
def test_a_bad_frame_header_raises_what_the_decode_raises(tmp_path, edit, at):
    path = _edit_first_frame_tag(_clip(tmp_path / "a.webm", 9), tmp_path / "bad.webm", edit, at)
    want = _decoded(path)
    assert _inspected(path) == want


def test_truncated_and_other_codecs_match_the_decode(tmp_path):
    clips = [
        _clip(tmp_path / "sd.webm", 9, audio = True),
        _clip(tmp_path / "inter.webm", 9, gop = 4),
        _clip(tmp_path / "vp9.webm", 9, codec = "libvpx-vp9"),
    ]
    data = clips[0].read_bytes()
    for i, cut in enumerate(
        sorted({len(data) * k // 13 for k in range(1, 13)} | {len(data) - 1, len(data) - 40})
    ):
        clip = tmp_path / f"cut{i}.webm"
        clip.write_bytes(data[:cut])
        clips.append(clip)
    for clip in clips:
        assert _inspected(clip) == _decoded(clip), clip.name
