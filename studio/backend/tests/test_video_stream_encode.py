# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The streamed encode must hand back exactly the frames the pipeline exports, and fall back otherwise."""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
av = pytest.importorskip("av")
diffusers = pytest.importorskip("diffusers")

from diffusers import AutoencoderKLWan  # noqa: E402
from diffusers.video_processor import VideoProcessor  # noqa: E402

from core.inference import video_stream_encode as vse  # noqa: E402
from core.inference.video_frames import uint8_video_frames  # noqa: E402


@pytest.fixture(autouse = True)
def _few_threads():
    # The tiny convs run in milliseconds on a few threads and minutes on a loaded many-core host.
    threads = torch.get_num_threads()
    torch.set_num_threads(min(4, threads))
    yield
    torch.set_num_threads(threads)


def _vae(patch = False):
    torch.manual_seed(0)
    kw = dict(
        base_dim = 16,
        z_dim = 4,
        dim_mult = [1, 1, 1, 1],
        num_res_blocks = 1,
        temperal_downsample = [False, True, True],
        latents_mean = [0.0] * 4,
        latents_std = [1.0] * 4,
    )
    if patch:
        kw.update(
            decoder_base_dim = 16, is_residual = True, in_channels = 12, out_channels = 12, patch_size = 2
        )
    vae = AutoencoderKLWan(**kw).eval()
    vae.disable_tiling()
    return vae


class _Pipe:
    def __init__(self, vae):
        self.vae = vae
        self.video_processor = VideoProcessor(vae_scale_factor = 8)

    def __call__(self, z):
        # WanPipeline's tail: decode, then postprocess_video to np.
        video = self.vae.decode(z, return_dict = False)[0]
        return self.video_processor.postprocess_video(video, output_type = "np")[0]


def _render(pipe, session, z):
    with torch.inference_mode(), uint8_video_frames(pipe), session.hooks():
        return pipe(z)


def _mp4_frames(data, tmp_path):
    path = tmp_path / "out.mp4"
    path.write_bytes(data)
    with av.open(str(path)) as c:
        return [f.to_ndarray(format = "rgb24") for f in c.decode(video = 0)]


@pytest.mark.parametrize("patch", [False, True])
def test_streamed_frames_are_the_exported_frames(patch, tmp_path):
    pipe = _Pipe(_vae(patch))
    z = torch.randn(1, 4, 4, 8, 8)
    with torch.inference_mode():
        reference = pipe.video_processor.postprocess_video(
            pipe.vae.decode(z, return_dict = False)[0], output_type = "np"
        )[0]
    session = vse.StreamEncode(pipe.vae, pipe.video_processor, 8)
    frames = _render(pipe, session, z)
    assert isinstance(frames, torch.Tensor) and frames.dtype == torch.uint8
    # 1 + 3 x 4 frames, streamed as 4 decoder chunks
    assert len(session.chunks) == 4 and len(frames) == 13
    assert np.array_equal(frames.numpy(), (reference * 255).round().astype(np.uint8))
    before = vse.STATS["streamed"]
    data = session.take(frames, 8)
    assert data is not None and vse.STATS["streamed"] == before + 1
    assert len(_mp4_frames(data, tmp_path)) == 13
    assert not os.path.exists(session.path)
    # hooks are removed with the session
    assert "decode" not in vars(pipe.vae) and not pipe.vae.decoder._forward_hooks


def test_tiled_decode_falls_back():
    vae = _vae()
    vae.enable_tiling(tile_sample_min_height = 32, tile_sample_min_width = 32)
    pipe = _Pipe(vae)
    session = vse.StreamEncode(vae, pipe.video_processor, 8)
    frames = _render(pipe, session, torch.randn(1, 4, 2, 8, 8))
    assert session.take(frames, 8) is None
    assert vse.STATS["last"] == "tiled decode"
    assert not os.path.exists(session.path)


def test_any_differing_byte_or_fps_falls_back():
    pipe = _Pipe(_vae())
    z = torch.randn(1, 4, 3, 8, 8)
    session = vse.StreamEncode(pipe.vae, pipe.video_processor, 8)
    frames = _render(pipe, session, z).clone()
    frames[5, 3, 3, 1] ^= 1
    assert session.take(frames, 8) is None and vse.STATS["last"] == "frames differ"
    session = vse.StreamEncode(pipe.vae, pipe.video_processor, 8)
    frames = _render(pipe, session, z)
    assert session.take(frames, 16) is None and vse.STATS["last"] == "fps differs"
    session = vse.StreamEncode(pipe.vae, pipe.video_processor, 8)
    frames = _render(pipe, session, z)
    assert session.take(frames[:-1], 8) is None and vse.STATS["last"] == "frame count differs"


def test_a_restarted_decode_falls_back():
    # The fp16 decode reruns in fp32 when its output is not finite: the first pass's chunks are stale.
    pipe = _Pipe(_vae())
    original = pipe.vae.decode

    def decode_twice(z, *args, **kwargs):
        original(z, *args, **kwargs)
        return original(z, *args, **kwargs)

    pipe.vae.decode = decode_twice
    session = vse.StreamEncode(pipe.vae, pipe.video_processor, 8)
    frames = _render(pipe, session, torch.randn(1, 4, 2, 8, 8))
    assert session.take(frames, 8) is None and vse.STATS["last"] == "decoder restarted"
    assert pipe.vae.decode is decode_twice


def test_failure_inside_the_render_closes_the_session():
    pipe = _Pipe(_vae())
    session = vse.StreamEncode(pipe.vae, pipe.video_processor, 8)
    with pytest.raises(RuntimeError):
        with torch.inference_mode(), uint8_video_frames(pipe), session.hooks():
            pipe(torch.randn(1, 4, 2, 8, 8))
            raise RuntimeError("cancelled")
    assert not session.valid and session._thread is None
    assert not os.path.exists(session.path)


def test_engagement_gates(monkeypatch):
    pipe = _Pipe(_vae())
    monkeypatch.setenv(vse.STREAM_ENCODE_ENV, "0")
    assert vse.open_stream_encode(pipe, 8, engage = True) is None
    monkeypatch.delenv(vse.STREAM_ENCODE_ENV)
    assert vse.open_stream_encode(pipe, 8, engage = False) is None
    monkeypatch.setenv("UNSLOTH_STUDIO_VIDEO_FRAME_THREADS", "0")
    assert vse.open_stream_encode(pipe, 8, engage = True) is None
    monkeypatch.delenv("UNSLOTH_STUDIO_VIDEO_FRAME_THREADS")
    # a CPU decoder never engages; the hook is only worth it beside a GPU decode
    assert vse.open_stream_encode(pipe, 8, engage = True) is None


def test_a_streamed_file_that_does_not_hold_the_clip_is_never_used(monkeypatch):
    pipe = _Pipe(_vae())
    session = vse.StreamEncode(pipe.vae, pipe.video_processor, 8)
    frames = _render(pipe, session, torch.randn(1, 4, 3, 8, 8))
    join = session._join

    def join_then_truncate():
        join()
        with open(session.path, "r+b") as f:
            f.truncate(os.path.getsize(session.path) // 2)

    monkeypatch.setattr(session, "_join", join_then_truncate)
    assert session.take(frames, 8) is None
    assert vse.STATS["last"].startswith("streamed file holds")


def test_encode_mp4_claims_an_offered_stream_once():
    from core.inference.video import VideoBackend

    pipe = _Pipe(_vae())
    session = vse.StreamEncode(pipe.vae, pipe.video_processor, 8)
    frames = _render(pipe, session, torch.randn(1, 4, 3, 8, 8))
    before = vse.STATS["streamed"]
    with vse.offer_stream(session):
        data = VideoBackend._encode_mp4(frames, 8, None, None)
        assert vse.STATS["streamed"] == before + 1
        # a second export in the same scope encodes normally
        again = VideoBackend._encode_mp4(frames, 8, None, None)
    assert vse._mp4_frame_count(data, frames.shape[1:3]) == len(frames)
    assert vse._mp4_frame_count(again, frames.shape[1:3]) == len(frames)
    assert vse.STATS["streamed"] == before + 1
