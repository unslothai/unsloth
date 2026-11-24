# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Streamed mp4 == stock export on every render in a process (libx264 AVX-512 heap reads); 208 px wide hits it."""

from __future__ import annotations

import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")
av = pytest.importorskip("av")
pytest.importorskip("diffusers.utils.export_utils")

from core.inference import video_encode as ve  # noqa: E402
from core.inference import video_stream_encode as vse  # noqa: E402

_H, _W, _N, _FPS, _RATE = 128, 208, 41, 24, 24000
# seeds per render: the repeats must give the bytes of their first render
_RENDERS = (0, 1, 2, 0, 1, 2)


def _clip(seed: int):
    g = np.random.default_rng(seed)
    base = g.integers(0, 256, (_H // 8, _W // 8, 3), dtype = np.uint8).repeat(8, 0).repeat(8, 1)
    return torch.from_numpy(np.stack([np.roll(base, 5 * i, axis = 1) for i in range(_N)]).copy())


def _audio(seed: int):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(2, _RATE * _N // _FPS, generator = g) * 2 - 1) * 0.3


class _Vae:
    config = type("C", (), {"patch_size": None})()


def _streamed(frames, audio):
    rate = _RATE if audio is not None else None
    session = vse.StreamEncode(_Vae(), None, _FPS, audio_sample_rate = rate)
    session._decoded = True
    pixels = frames.permute(0, 3, 1, 2).float() / 255
    for start, stop in [(0, 1)] + [(i, i + 4) for i in range(1, _N, 4)]:
        session._push_uint8(pixels[start:stop])
    if audio is None:
        return session.take(frames, _FPS)
    return session.take(frames, _FPS, audio, _RATE)


def _stock(frames, audio, path):
    assert ve.encode_x264(frames, _FPS, str(path), audio, _RATE if audio is not None else None)
    return path.read_bytes()


@pytest.fixture(autouse = True)
def _defaults(monkeypatch):
    for env in (ve.FRAME_THREADS_ENV, vse.STREAM_ENCODE_ENV):
        monkeypatch.delenv(env, raising = False)


@pytest.mark.parametrize("with_audio", [False, True])
def test_streamed_mp4_is_the_stock_mp4_on_every_render(tmp_path, with_audio):
    first: dict = {}
    for render, seed in enumerate(_RENDERS):
        frames = _clip(seed)
        audio = _audio(seed) if with_audio else None
        streamed = _streamed(frames, audio)
        assert streamed is not None, vse.STATS["last"]
        stock = _stock(frames, audio, tmp_path / f"stock_{render}.mp4")
        assert (
            streamed == stock
        ), f"render {render} (seed {seed}): streamed mp4 is not the stock mp4"
        assert (
            first.setdefault(seed, stock) == stock
        ), f"render {render}: seed {seed} encoded differently"


def test_stock_export_does_not_depend_on_the_thread_or_history(tmp_path):
    frames = _clip(0)
    want = _stock(frames, None, tmp_path / "first.mp4")
    for i in range(4):
        _stock(_clip(i + 1), None, tmp_path / "other.mp4")  # leave a different heap behind
        got: list = []
        worker = threading.Thread(
            target = lambda: got.append(_stock(frames, None, tmp_path / "t.mp4"))
        )
        worker.start()
        worker.join()
        assert got == [want], f"pass {i}: the clip encoded on another thread differs"
        assert _stock(frames, None, tmp_path / "again.mp4") == want, f"pass {i}: re-encode differs"


def test_asm_cap_only_where_avx512_would_run(monkeypatch):
    monkeypatch.delenv(ve.X264_ASM_CAP_ENV, raising = False)
    monkeypatch.setattr(ve, "_cpu_runs_x264_avx512", lambda: True)
    assert ve.x264_options() == {"x264-params": "asm=AVX2"}
    for value in ("0", "false", "off"):
        monkeypatch.setenv(ve.X264_ASM_CAP_ENV, value)
        assert ve.x264_options() == {}
    monkeypatch.delenv(ve.X264_ASM_CAP_ENV)
    monkeypatch.setattr(ve, "_cpu_runs_x264_avx512", lambda: False)
    assert ve.x264_options() == {}


def test_both_paths_open_the_same_stream(monkeypatch, tmp_path):
    opened: list = []
    real = ve.add_x264_stream

    def spy(container, fps, width, height):
        stream = real(container, fps, width, height)
        opened.append((int(fps), int(width), int(height), dict(stream.codec_context.options or {})))
        return stream

    monkeypatch.setattr(ve, "add_x264_stream", spy)
    frames = _clip(3)
    assert _streamed(frames, None) is not None
    _stock(frames, None, tmp_path / "s.mp4")
    assert len(opened) == 2 and opened[0] == opened[1]
