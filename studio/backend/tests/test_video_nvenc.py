# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in NVENC export (``video_nvenc.py``): off unless asked for, never probed on GPUs without an encoder, probed once
per GPU, and libx264 whenever NVENC is unusable or fails."""

from __future__ import annotations


import numpy as np
import pytest

torch = pytest.importorskip("torch")

from core.inference import video_nvenc as vn  # noqa: E402


@pytest.fixture(autouse = True)
def _fresh(monkeypatch):
    monkeypatch.setattr(vn, "_probed", {})


def _fake_cuda(
    monkeypatch,
    name,
    *,
    hip = None,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index = None: name)
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    probes: list = []

    def _probe(gpu):
        probes.append(gpu)
        return True

    monkeypatch.setattr(vn, "_probe", _probe)
    return probes


def test_off_unless_requested(monkeypatch):
    monkeypatch.delenv(vn.ENCODER_ENV, raising = False)
    probes = _fake_cuda(monkeypatch, "NVIDIA L4")
    assert vn.nvenc_gpu() is None and probes == []
    monkeypatch.setenv(vn.ENCODER_ENV, "x264")
    assert vn.nvenc_gpu() is None and probes == []


@pytest.mark.parametrize(
    "name",
    [
        "NVIDIA B200",
        "NVIDIA B100",
        "NVIDIA A100-SXM4-40GB",
        "NVIDIA H100 80GB HBM3",
        "NVIDIA GB200",
        "NVIDIA H20",
        "NVIDIA GH200 480GB",
    ],
)
def test_gpus_without_an_encoder_are_never_probed(monkeypatch, name):
    monkeypatch.setenv(vn.ENCODER_ENV, "nvenc")
    probes = _fake_cuda(monkeypatch, name)
    assert vn.nvenc_gpu() is None and probes == []


def test_probed_once_per_gpu(monkeypatch):
    monkeypatch.setenv(vn.ENCODER_ENV, " NVENC ")
    probes = _fake_cuda(monkeypatch, "NVIDIA RTX A3000 Laptop GPU")
    assert vn.nvenc_gpu() == 0
    assert vn.nvenc_gpu() == 0
    assert probes == [0]


def test_a_failed_probe_is_retried_after_the_window(monkeypatch):
    monkeypatch.setenv(vn.ENCODER_ENV, "nvenc")
    _fake_cuda(monkeypatch, "NVIDIA GeForce RTX 4090")
    results = iter([False, True])
    probes: list = []
    monkeypatch.setattr(vn, "_probe", lambda gpu: probes.append(gpu) or next(results))
    now = [1000.0]
    monkeypatch.setattr(vn.time, "monotonic", lambda: now[0])
    assert vn.nvenc_gpu() is None
    now[0] += vn._PROBE_RETRY_S - 1
    assert vn.nvenc_gpu() is None and probes == [0]
    now[0] += 1
    assert vn.nvenc_gpu() == 0 and probes == [0, 0]
    now[0] += 10 * vn._PROBE_RETRY_S
    assert vn.nvenc_gpu() == 0 and probes == [0, 0]


def test_rocm_is_never_tried(monkeypatch):
    monkeypatch.setenv(vn.ENCODER_ENV, "nvenc")
    probes = _fake_cuda(monkeypatch, "AMD Radeon 8060S", hip = "7.1")
    assert vn.nvenc_gpu() is None and probes == []


def test_frames_match_what_encode_video_would_encode():
    rng = np.random.default_rng(0)
    clip = rng.random((3, 8, 12, 3), dtype = np.float32)
    assert np.array_equal(vn._uint8_frames(clip), (clip * 255).round().astype("uint8"))
    u8 = (clip * 255).round().astype("uint8")
    assert np.array_equal(vn._uint8_frames(torch.from_numpy(u8)), u8)
    from PIL import Image

    assert np.array_equal(vn._uint8_frames([Image.fromarray(f) for f in u8]), u8)
    clip[0, 0, 0, 0] = np.nan
    assert vn._uint8_frames(clip) is None


def test_a_failed_encode_asks_for_libx264(monkeypatch, tmp_path):
    def _boom(*args, **kwargs):
        raise RuntimeError("OpenEncodeSessionEx failed: incompatible client key (21)")

    monkeypatch.setattr(vn, "_encode", _boom)
    clip = np.zeros((2, 16, 16, 3), dtype = np.uint8)
    assert vn.encode_nvenc(clip, 24, str(tmp_path / "x.mp4"), 0) is False
    assert (
        vn.encode_nvenc(
            np.full((2, 16, 16, 3), 2.0, dtype = np.float32), 24, str(tmp_path / "x.mp4"), 0
        )
        is False
    )


@pytest.mark.parametrize(
    "gpu, nvenc_ok, expect", [(None, None, "x264"), (0, True, "nvenc"), (0, False, "x264")]
)
def test_encode_mp4_falls_back_to_libx264(monkeypatch, gpu, nvenc_ok, expect):
    eu = pytest.importorskip("diffusers.utils.export_utils")

    from core.inference import video as video_mod

    used: list = []

    def _x264(frames, fps, path, **kwargs):
        used.append("x264")
        open(path, "wb").write(b"x264")

    def _nvenc(frames, fps, path, gpu_index, audio, sample_rate):
        if nvenc_ok:
            used.append("nvenc")
            open(path, "wb").write(b"nvenc")
        return nvenc_ok

    monkeypatch.setattr(eu, "encode_video", _x264)
    monkeypatch.setattr(video_mod, "nvenc_gpu", lambda logger = None: gpu)
    monkeypatch.setattr(video_mod, "encode_nvenc", _nvenc)
    out = video_mod.VideoBackend._encode_mp4(
        np.zeros((2, 16, 16, 3), dtype = np.float32), 24, None, None
    )
    assert used == [expect] and out == expect.encode()


def _nvenc_here():
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    import re
    return not set(re.split(r"[\s\-_/]+", torch.cuda.get_device_name(0).upper())) & set(
        vn._NO_NVENC_GPUS
    )


@pytest.mark.skipif(not _nvenc_here(), reason = "needs an NVIDIA GPU with an NVENC engine")
def test_real_nvenc_clip_round_trips(monkeypatch, tmp_path):
    av = pytest.importorskip("av")
    monkeypatch.setenv(vn.ENCODER_ENV, "nvenc")
    gpu = vn.nvenc_gpu()
    if gpu is None:
        pytest.skip("NVENC did not open here (driver / container)")
    yy, xx = np.mgrid[0:128, 0:192]
    clip = np.stack(
        [np.stack([(xx + 4 * t) % 256, (yy + 2 * t) % 256, (xx + yy) % 256], -1) for t in range(24)]
    )
    clip = clip.astype(np.uint8)
    audio = torch.zeros(2, 48000)
    path = str(tmp_path / "nv.mp4")
    assert vn.encode_nvenc(clip, 24, path, gpu, audio, 48000)
    with av.open(path) as container:
        codecs = {s.type: s.codec_context.name for s in container.streams}
        frames = [f.to_ndarray(format = "rgb24") for f in container.decode(video = 0)]
    assert codecs["video"] == "h264" and "audio" in codecs and len(frames) == 24
    mse = np.mean((np.stack(frames).astype(np.float64) - clip) ** 2)
    assert 10 * np.log10(255**2 / mse) > 30
