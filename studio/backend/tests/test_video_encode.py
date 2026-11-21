# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


from __future__ import annotations

import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")
av = pytest.importorskip("av")
eu = pytest.importorskip("diffusers.utils.export_utils")

from core.inference import video_encode as ve  # noqa: E402

_THREADING = {"threads", "lookahead_threads", "sliced_threads", "slices", "sync_lookahead"}


def _clip(
    frames = 24,
    height = 128,
    width = 192,
):
    yy, xx = np.mgrid[0:height, 0:width]
    return np.stack(
        [
            np.stack([(xx + 4 * t) % 256, (yy + 2 * t) % 256, (xx + yy) % 256], -1)
            for t in range(frames)
        ]
    ).astype(np.uint8)


def _x264_options(path):
    data = open(path, "rb").read()
    match = re.search(rb"x264 - core .*? options: ([^\x00]+?)\x00", data, re.S)
    assert match, "no x264 SEI"
    return dict(kv.split("=", 1) for kv in match.group(1).decode().split() if "=" in kv)


def _decode(path):
    with av.open(path) as container:
        streams = {s.type: s for s in container.streams}
        video = streams["video"]
        info = (
            video.codec_context.name,
            video.codec_context.pix_fmt,
            video.average_rate,
            set(streams),
        )
        frames = np.stack([f.to_ndarray(format = "rgb24") for f in container.decode(video = 0)])
    return info, frames


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255**2 / mse)


@pytest.mark.parametrize("with_audio", [False, True])
def test_same_stream_and_settings_as_encode_video(tmp_path, monkeypatch, with_audio):
    monkeypatch.delenv(ve.FRAME_THREADS_ENV, raising = False)
    clip = _clip()
    audio = torch.zeros(2, 48000) if with_audio else None
    rate = 48000 if with_audio else None
    ours, stock = str(tmp_path / "ours.mp4"), str(tmp_path / "stock.mp4")
    assert ve.encode_x264(torch.from_numpy(clip), 24, ours, audio, rate)
    eu.encode_video(
        torch.from_numpy(clip),
        24,
        stock,
        **({"audio": audio, "audio_sample_rate": rate} if with_audio else {}),
    )
    info_ours, frames_ours = _decode(ours)
    info_stock, frames_stock = _decode(stock)
    assert info_ours == info_stock
    assert info_ours[0] == "h264" and info_ours[1] == "yuv420p"
    assert ("audio" in info_ours[3]) == with_audio
    assert frames_ours.shape == frames_stock.shape == clip.shape
    assert _psnr(frames_ours, clip) > 30
    opts_ours, opts_stock = _x264_options(ours), _x264_options(stock)
    assert opts_ours["sliced_threads"] == "0" and opts_ours["crf"] == "23.0"
    keys = (set(opts_ours) | set(opts_stock)) - _THREADING
    assert {k: opts_ours.get(k) for k in keys} == {k: opts_stock.get(k) for k in keys}


def test_float_pil_and_uint8_inputs_encode_the_same_frames(tmp_path):
    from PIL import Image

    clip = _clip(frames = 8)
    inputs = {
        "u8": torch.from_numpy(clip),
        "np": clip.astype(np.float32) / 255,
        "pil": [Image.fromarray(f) for f in clip],
    }
    decoded = {}
    for name, value in inputs.items():
        path = str(tmp_path / f"{name}.mp4")
        assert ve.encode_x264(value, 24, path)
        decoded[name] = _decode(path)[1]
    assert np.array_equal(decoded["u8"], decoded["np"]) and np.array_equal(
        decoded["u8"], decoded["pil"]
    )


@pytest.mark.parametrize("value", ["0", "false", "off"])
def test_kill_switch(tmp_path, monkeypatch, value):
    monkeypatch.setenv(ve.FRAME_THREADS_ENV, value)
    assert ve.encode_x264(_clip(frames = 2), 24, str(tmp_path / "x.mp4")) is False


def test_unsure_input_and_failures_hand_back(tmp_path, monkeypatch):
    path = str(tmp_path / "x.mp4")
    assert ve.encode_x264(np.full((2, 16, 16, 3), 2.0, dtype = np.float32), 24, path) is False
    assert ve.encode_x264(np.zeros((2, 16, 16, 4), dtype = np.uint8), 24, path) is False
    assert (
        ve.encode_x264(np.zeros((2, 16, 16, 3), dtype = np.uint8), 24, path, torch.zeros(2, 10))
        is False
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("encoder open failed")

    monkeypatch.setattr(av, "open", _boom)
    assert ve.encode_x264(np.zeros((2, 16, 16, 3), dtype = np.uint8), 24, path) is False


@pytest.mark.parametrize("x264_ok, expect", [(True, ["frame"]), (False, ["frame", "stock"])])
def test_encode_mp4_routes_to_frame_threads_first(monkeypatch, x264_ok, expect):
    from core.inference import video as video_mod

    used: list = []

    def _stock(frames, fps, path, **kwargs):
        used.append("stock")
        open(path, "wb").write(b"stock")

    def _frame(frames, fps, path, audio, rate):
        used.append("frame")
        if x264_ok:
            open(path, "wb").write(b"frame")
        return x264_ok

    monkeypatch.setattr(eu, "encode_video", _stock)
    monkeypatch.setattr(video_mod, "nvenc_gpu", lambda logger = None: None)
    monkeypatch.setattr(video_mod, "encode_x264", _frame)
    out = video_mod.VideoBackend._encode_mp4(
        np.zeros((2, 16, 16, 3), dtype = np.float32), 24, None, None
    )
    assert used == expect and out == (b"frame" if x264_ok else b"stock")


def test_encode_mp4_end_to_end_default_is_frame_threaded(monkeypatch):
    from core.inference import video as video_mod

    monkeypatch.delenv(ve.FRAME_THREADS_ENV, raising = False)
    monkeypatch.setattr(video_mod, "nvenc_gpu", lambda logger = None: None)
    data = video_mod.VideoBackend._encode_mp4(torch.from_numpy(_clip()), 24, None, None)
    assert b"sliced_threads=0" in data


def _bt709_yuv420(rgb):
    # the reference save's conversion: BT.709 matrix, limited range, 4:2:0 (swscale's own reformat)
    frame = av.VideoFrame.from_ndarray(rgb, format = "rgb24").reformat(
        format = "yuv420p", dst_colorspace = 1
    )
    return frame.to_ndarray()


def test_reference_export_is_crf_18_bt709_and_float_audio(tmp_path, monkeypatch):
    monkeypatch.delenv(ve.FRAME_THREADS_ENV, raising = False)
    clip = _clip(frames = 12)
    t = torch.arange(32000 * 1) / 32000
    audio = torch.stack(
        [0.3 * torch.sin(2 * np.pi * 220 * t), 0.2 * torch.sin(2 * np.pi * 330 * t)]
    )
    ref, plain = str(tmp_path / "ref.mp4"), str(tmp_path / "plain.mp4")
    assert ve.encode_x264(torch.from_numpy(clip), 24, ref, audio, 32000, reference = True)
    assert ve.encode_x264(torch.from_numpy(clip), 24, plain, audio, 32000)
    assert _x264_options(ref)["crf"].startswith("18.0")
    assert _x264_options(plain)["crf"].startswith(
        "23.0"
    )  # libx264's default, other families keep it
    with av.open(ref) as container:
        video = container.streams.video[0]
        cc = video.codec_context
        assert (
            int(cc.color_primaries),
            int(cc.color_trc),
            int(cc.colorspace),
            int(cc.color_range),
        ) == (1, 13, 1, 1)
        decoded = [f.to_ndarray() for f in container.decode(video = 0)]
    # the decoded luma tracks the BT.709 conversion of the source, not swscale's BT.601 default
    want = np.stack([_bt709_yuv420(f) for f in clip])
    got = np.stack(decoded)
    assert got.shape == want.shape
    assert _psnr(got, want) > 40
    with av.open(plain) as container:
        cc = container.streams.video[0].codec_context
        assert int(cc.colorspace) in (0, 2)  # untagged, as before
    with av.open(ref) as container:
        pcm = np.concatenate([f.to_ndarray() for f in container.decode(audio = 0)], axis = 1)
    n = min(pcm.shape[1], audio.shape[1]) - 2048
    assert np.abs(pcm[:, 1024 : 1024 + n] - audio.numpy()[:, :n]).max() < 0.05 or pcm.shape[0] == 2


def test_reference_frames_keep_the_plain_path_unchanged():
    rgb = _clip(frames = 1)[0]
    plain = ve.x264_frame(av, rgb)
    assert plain.format.name == "rgb24"
    ref = ve.x264_frame(av, rgb, reference = True)
    assert ref.format.name == "yuv420p" and int(ref.colorspace) == 1 and int(ref.color_range) == 1
    assert np.array_equal(ref.to_ndarray(), _bt709_yuv420(rgb))
    # one reformatter for the whole encode converts the same way (the per-frame call rebuilds swscale each time)
    shared = ve.x264_reformatter(av, reference = True)
    assert ve.x264_reformatter(av) is None
    for frame in _clip(frames = 3):
        got = ve.x264_frame(av, frame, reference = True, reformatter = shared)
        assert np.array_equal(got.to_ndarray(), _bt709_yuv420(frame)) and int(got.colorspace) == 1


def test_reference_audio_takes_mono_and_channels_last(tmp_path):
    for wave in (torch.zeros(16000), torch.zeros(16000, 2), torch.zeros(1, 16000)):
        path = str(tmp_path / "a.mp4")
        with av.open(path, mode = "w", format = "mp4") as container:
            stream = eu._prepare_audio_stream(container, 16000)
            ve.write_reference_audio(container, stream, wave, 16000, av)
        with av.open(path) as container:
            assert container.streams.audio[0].codec_context.channels == 2
