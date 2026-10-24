# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The streamed encode must hand back exactly the frames the pipeline exports, and fall back otherwise."""

from __future__ import annotations

import io
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
av = pytest.importorskip("av")
diffusers = pytest.importorskip("diffusers")

# Lazy diffusers imports can raise RuntimeError (broken bitsandbytes): skip, not abort collection.
try:
    from diffusers import AutoencoderKLWan  # noqa: E402
    from diffusers.video_processor import VideoProcessor  # noqa: E402
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"diffusers is not importable here: {type(exc).__name__}", allow_module_level = True)

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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_a_compiled_decode_runs_the_chunk_hook_as_one_eager_island(device):
    """UNSLOTH_DIFFUSION_COMPILE_VAE=1 compiles ``vae.decode`` (fullgraph=False, dynamic=True): Dynamo must not trace
    into the hook's thread, queue and CUDA stream code, and the streamed frames are still the exported ones."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    pipe = _Pipe(_vae().to(device))
    z = torch.randn(1, 4, 4, 8, 8, device = device)
    with torch.inference_mode():
        reference = pipe.video_processor.postprocess_video(
            pipe.vae.decode(z, return_dict = False)[0], output_type = "np"
        )[0]
    pipe.vae.decode = torch.compile(pipe.vae.decode, backend = "eager", fullgraph = False, dynamic = True)
    try:
        for _ in range(2):
            session = vse.StreamEncode(pipe.vae, pipe.video_processor, 8)
            frames = _render(pipe, session, z)
            assert len(session.chunks) == 4
            assert np.array_equal(frames.cpu().numpy(), (reference * 255).round().astype(np.uint8))
            assert session.take(frames, 8) is not None
        reasons = {k.splitlines()[0] for k in counters["graph_break"]}
        assert all("torch.compiler.disable" in r for r in reasons), reasons
        assert sum(counters["graph_break"].values()) <= 2
    finally:
        del pipe.vae.decode
        torch._dynamo.reset()


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


def test_an_offloaded_vae_engages_when_the_pipeline_executes_on_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    pipe = _Pipe(_vae())
    assert vse.open_stream_encode(pipe, 8, engage = True) is None
    pipe._execution_device = torch.device("cpu")
    assert vse.open_stream_encode(pipe, 8, engage = True) is None
    pipe._execution_device = torch.device("cuda", 0)
    session = vse.open_stream_encode(pipe, 8, engage = True)
    try:
        assert isinstance(session, vse.StreamEncode)
    finally:
        session.close()
    monkeypatch.setenv(vse.STREAM_ENCODE_ENV, "0")
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


def test_a_failed_final_flush_falls_back_without_hanging(monkeypatch):
    """The flush runs after the encoder thread took the stop marker; a failure there must not wait for another."""
    import threading

    class _Stream:
        width = height = 0
        pix_fmt = None
        codec_context = type("Ctx", (), {"thread_type": None})()

        def encode(self, frame = None):
            if frame is None:
                raise OSError(28, "No space left on device")
            return []

    class _Container:
        def add_stream(self, *args, **kwargs):
            return _Stream()

        def mux(self, packet):
            pass

        def close(self):
            pass

    monkeypatch.setattr(av, "open", lambda *args, **kwargs: _Container())
    pipe = _Pipe(_vae())
    session = vse.StreamEncode(pipe.vae, pipe.video_processor, 8)
    frames = _render(pipe, session, torch.randn(1, 4, 2, 8, 8))
    result = {}
    worker = threading.Thread(
        target = lambda: result.setdefault("data", session.take(frames, 8)), daemon = True
    )
    worker.start()
    worker.join(timeout = 60)
    assert not worker.is_alive(), "take() hung after the encoder's final flush failed"
    assert result["data"] is None
    session.close()
    assert not os.path.exists(session.path)


def test_rocm_bf16_decode_streams_the_exported_frames(monkeypatch, tmp_path):
    """ROCm gfx11 / gfx12 decode the Wan VAE in bf16 by default and widen the result to fp32: the streamed chunks
    must be widened the same way, or the session falls back ("frames differ") on every clip."""
    from types import SimpleNamespace

    from core.inference.diffusion_device import install_rocm_vae_bf16_decode

    monkeypatch.setenv(
        "UNSLOTH_VIDEO_VAE_BF16_DECODE", "1"
    )  # forced: the CPU stands in for the ROCm card
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    for patch, outer in ((False, False), (True, False), (False, True), (True, True)):
        pipe = _Pipe(_vae(patch))
        target = SimpleNamespace(device = "cuda", backend = "rocm", dtype = torch.bfloat16)
        assert install_rocm_vae_bf16_decode(pipe, target) == "weights"
        if outer:
            # Studio's default video load wraps the bf16 decode again (video_vae_untiled's untiled-when-fits layer):
            # the outer function carries none of the inner wrapper's attributes.
            inner = pipe.vae.decode

            def untiled_when_fits(
                *args,
                _inner = inner,
                **kwargs,
            ):
                return _inner(*args, **kwargs)

            pipe.vae.decode = untiled_when_fits
        assert pipe.vae.decoder.conv_in.weight.dtype == torch.bfloat16
        z = torch.randn(1, 4, 4, 8, 8)
        session = vse.StreamEncode(pipe.vae, pipe.video_processor, 8)
        frames = _render(pipe, session, z)
        assert len(session.chunks) == 4 and len(frames) == 13
        before = vse.STATS["streamed"]
        data = session.take(frames, 8)
        assert data is not None, vse.STATS["last"]
        assert vse.STATS["streamed"] == before + 1
        assert len(_mp4_frames(data, tmp_path)) == 13


# MiniMax-H3: temporal clips cross-faded into each other, a soundtrack muxed at the end.
_H3_RATE = 16000
_H3_PIXEL = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def _h3_vae():
    cls = getattr(diffusers, "AutoencoderKLMiniMaxH3", None)
    if cls is None:
        pytest.skip("diffusers without AutoencoderKLMiniMaxH3")
    torch.manual_seed(0)
    vae = cls(
        block_out_channels = (16, 32),
        layers_per_block = 1,
        spatial_downsample_factors = (2, 2),
        temporal_downsample_factors = (2, 1),
        norm_num_groups = 8,
        latent_channels = 8,
        decoder_num_layers = 1,
        decoder_num_attention_heads = 2,
        decoder_attention_head_dim = 16,
        decoder_num_register_tokens = 2,
        decoder_ffn_mult = 2,
        clip_length = 9,
        token_drop = 1,
    ).eval()
    vae.tile_sample_min_height = vae.tile_sample_min_width = 16
    vae.tile_sample_min_overlap_height = vae.tile_sample_min_overlap_width = 4
    return vae


class _H3Pipe:
    """MiniMaxH3VideoDecodeStep's tail: decode, ImageNet denormalise, postprocess_video to np."""

    pixel_mean, pixel_std = _H3_PIXEL
    audio_sampling_rate = _H3_RATE

    def __init__(self, vae):
        self.vae = vae
        self.video_processor = VideoProcessor(vae_scale_factor = 16, do_normalize = False)

    def __call__(self, z):
        video = self.vae.decode(z, return_dict = False)[0]
        mean = torch.tensor(self.pixel_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.pixel_std).view(1, -1, 1, 1, 1)
        video = (video.float() * std + mean).clamp(0, 1)
        return self.video_processor.postprocess_video(video, output_type = "np")[0]


def _h3_session(pipe, **kw):
    return vse.H3StreamEncode(
        pipe.vae,
        pipe.video_processor,
        8,
        audio_sample_rate = kw.get("rate", _H3_RATE),
        pixel_mean = pipe.pixel_mean,
        pixel_std = pipe.pixel_std,
    )


def _h3_render(
    pipe,
    session,
    latent_frames = 14,
):
    torch.manual_seed(1)
    z = torch.randn(1, 8, latent_frames, 4, 6)
    with torch.inference_mode(), uint8_video_frames(pipe), session.hooks():
        return pipe(z)


def _audio(frames):
    torch.manual_seed(2)
    return (torch.rand(2, _H3_RATE * frames // 8) * 2 - 1) * 0.5


def _mp4_layout(data):
    """What two encodes of the same frames and soundtrack with the same settings must share. Video packet bytes
    are not in it: libx264 through PyAV is not run-to-run byte-reproducible here (two stock exports of one clip
    can differ in a rate-control decision), so they would make the comparison flaky."""
    import re

    with av.open(io.BytesIO(data)) as c:
        streams = []
        for s in c.streams:
            cc = s.codec_context
            shape = (
                (cc.width, cc.height, str(cc.pix_fmt))
                if s.type == "video"
                else (cc.sample_rate, cc.channels)
            )
            streams.append((s.index, s.type, cc.name, shape, str(s.time_base)))
        video = [p for p in c.demux(video = 0) if p.size]
        c.seek(0)
        audio = [bytes(p) for p in c.demux(audio = 0) if p.size]
        c.seek(0)
        decoded = sum(1 for _ in c.decode(video = 0))
        extradata = bytes(c.streams.video[0].codec_context.extradata or b"")
        settings = re.search(
            rb"x264 - core [^\x00]*", extradata + b"".join(bytes(p) for p in video[:1])
        )
    return {
        "streams": streams,
        "x264_settings": settings.group(0) if settings else None,
        "video_packets": len(video),
        "decoded_frames": decoded,
        "audio_packets": audio,
    }


def test_h3_streamed_mp4_matches_the_x264_export(tmp_path):
    from core.inference.video_encode import encode_x264

    pipe = _H3Pipe(_h3_vae())
    session = _h3_session(pipe)
    frames = _h3_render(pipe, session)
    assert isinstance(frames, torch.Tensor) and frames.dtype == torch.uint8
    streamed = sum(len(host) for host, _ in session.chunks)
    # two clips stream while decoding; the final cross-fade tail comes from the exported clip
    assert 0 < streamed < len(frames)
    audio = _audio(len(frames))
    data = session.take(frames, 8, audio, _H3_RATE)
    assert data is not None, vse.STATS["last"]
    reference = tmp_path / "reference.mp4"
    assert encode_x264(frames, 8, str(reference), audio, _H3_RATE)
    got, want = _mp4_layout(data), _mp4_layout(reference.read_bytes())
    assert want["x264_settings"] is not None and want["decoded_frames"] == len(frames)
    # same streams in the same order, same encoder settings, same frame count, the soundtrack byte for byte
    assert got == want


def test_h3_soundtrack_mismatch_falls_back():
    pipe = _H3Pipe(_h3_vae())
    for audio, rate in ((None, None), ("audio", _H3_RATE + 1)):
        session = _h3_session(pipe)
        frames = _h3_render(pipe, session)
        track = _audio(len(frames)) if audio else None
        assert session.take(frames, 8, track, rate) is None
    assert vse.STATS["last"] in ("audio differs", "audio rate differs")


def test_h3_differing_frames_fall_back():
    pipe = _H3Pipe(_h3_vae())
    session = _h3_session(pipe)
    frames = _h3_render(pipe, session).clone()
    frames[1, 0, 0, 0] ^= 1
    assert session.take(frames, 8, _audio(len(frames)), _H3_RATE) is None
    assert vse.STATS["last"] == "frames differ"


def test_h3_a_second_decode_pass_falls_back():
    pipe = _H3Pipe(_h3_vae())
    session = _h3_session(pipe)
    vae = pipe.vae

    def retried(z, return_dict = True):
        # the OOM scope's retry: a second _decode pass inside one outermost decode
        vae._decode(z)
        return (vae._decode(z),)

    vae.decode = retried
    frames = _h3_render(pipe, session)
    assert session.take(frames, 8, _audio(len(frames)), _H3_RATE) is None
    assert vse.STATS["last"] == "decoder restarted"


def test_h3_hooks_leave_the_decode_output_and_the_vae_untouched():
    pipe = _H3Pipe(_h3_vae())
    before = {name: pipe.vae.__dict__.get(name) for name in ("decode", "_decode", "_decode_clip")}
    torch.manual_seed(1)
    z = torch.randn(1, 8, 14, 4, 6)
    with torch.inference_mode(), uint8_video_frames(pipe):
        reference = pipe(z)
    session = _h3_session(pipe)
    frames = _h3_render(pipe, session)
    session.close()
    assert torch.equal(frames, reference)
    assert {name: pipe.vae.__dict__.get(name) for name in before} == before


def test_h3_engagement_gates(monkeypatch):
    pipe = _H3Pipe(_h3_vae())
    assert vse.open_h3_stream_encode(pipe, 8, engage = False) is None
    monkeypatch.setenv(vse.STREAM_ENCODE_ENV, "0")
    assert vse.open_h3_stream_encode(pipe, 8, engage = True) is None
    monkeypatch.delenv(vse.STREAM_ENCODE_ENV)
    # a Wan VAE never takes the H3 session, and an H3 VAE never takes the Wan one
    assert vse.open_h3_stream_encode(_Pipe(_vae()), 8, engage = True) is None
    assert vse.open_stream_encode(pipe, 8, engage = True) is None


def test_encode_mp4_claims_an_h3_stream_with_its_soundtrack():
    from core.inference.video import VideoBackend

    pipe = _H3Pipe(_h3_vae())
    session = _h3_session(pipe)
    frames = _h3_render(pipe, session)
    audio = _audio(len(frames))
    before = vse.STATS["streamed"]
    with vse.offer_stream(session):
        data = VideoBackend._encode_mp4(frames, 8, audio, pipe, audio_sample_rate = _H3_RATE)
    assert vse.STATS["streamed"] == before + 1
    with av.open(io.BytesIO(data)) as c:
        assert len(c.streams.audio) == 1
    assert vse._mp4_frame_count(data, frames.shape[1:3]) == len(frames)
