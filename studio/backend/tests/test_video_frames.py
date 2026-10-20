# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""video_frames must match diffusers' np / pil export exactly, else fall back."""

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from diffusers.video_processor import VideoProcessor  # noqa: E402

from core.inference.video_frames import device_uint8, uint8_video_frames  # noqa: E402


def _clip(
    dtype,
    *,
    normalized,
    frames = 5,
    height = 16,
    width = 24,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    video = torch.rand((1, 3, frames, height, width), generator = g)
    # Exact rounding boundaries (k + 0.5) / 255 plus range ends.
    edge = (
        (torch.arange(height * width) % 256)
        .float()
        .add(0.5)
        .div(255)
        .clamp(0, 1)
        .view(height, width)
    )
    video[0, 0, 0] = edge
    video[0, 1, 0] = 0.0
    video[0, 2, 0] = 1.0
    if normalized:
        video = video * 2 - 1
    return video.to(dtype)


def _encode_video_uint8(np_frames):
    # diffusers.utils.export_utils.encode_video's np branch.
    return (np_frames * 255).round().astype("uint8")


def _run(vp, video, output_type):
    with uint8_video_frames(types.SimpleNamespace(video_processor = vp)):
        return vp.postprocess_video(video, output_type = output_type)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("normalized", [False, True])
def test_matches_the_np_export(dtype, normalized):
    vp = VideoProcessor(vae_scale_factor = 8, do_normalize = normalized)
    video = _clip(dtype, normalized = normalized)
    expected = _encode_video_uint8(vp.postprocess_video(video, output_type = "np"))
    got = _run(vp, video, "np")
    assert got.dtype == torch.uint8 and got.shape == expected.shape
    assert np.array_equal(got.numpy(), expected)


def test_matches_the_pil_export_for_every_clip_in_a_batch():
    vp = VideoProcessor(vae_scale_factor = 8, do_normalize = True)
    video = torch.cat([_clip(torch.bfloat16, normalized = True, seed = s) for s in (3, 4)])
    expected = np.stack(
        [
            np.stack([np.array(im) for im in clip])
            for clip in vp.postprocess_video(video, output_type = "pil")
        ]
    )
    got = _run(vp, video, "pil")
    assert np.array_equal(got.numpy(), expected)
    assert np.array_equal(got[1].numpy(), expected[1])


def test_every_float32_near_a_rounding_boundary():
    k = torch.arange(256, dtype = torch.float32)
    bits = torch.cat([(k + 0.5) / 255, k / 255]).clamp(0, 1).view(torch.int32)
    values = torch.cat([bits + d for d in range(-512, 513)]).view(torch.float32)
    values = values[(values >= 0) & (values <= 1)]
    n = values.numel() - values.numel() % 3
    frames = values[:n].view(-1, 3, 1, 1)
    got = device_uint8(frames).reshape(-1).numpy()
    expected = _encode_video_uint8(VaeImageProcessor.pt_to_numpy(frames)).reshape(-1)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("bad", [float("nan"), 1.5, -0.25])
@pytest.mark.parametrize("output_type", ["np", "pil"])
def test_out_of_range_takes_the_original_postprocess(bad, output_type):
    vp = VideoProcessor(vae_scale_factor = 8, do_normalize = False)
    video = _clip(torch.float32, normalized = False)
    video[0, 0, 1, 2, 3] = bad
    expected = vp.postprocess_video(video, output_type = output_type)
    got = _run(vp, video, output_type)
    if output_type == "np":
        assert isinstance(got, np.ndarray) and np.array_equal(got, expected, equal_nan = True)
    else:
        assert isinstance(got[0][1], type(expected[0][1]))
        assert all(np.array_equal(np.array(a), np.array(b)) for a, b in zip(got[0], expected[0]))


def test_other_output_types_and_the_processor_are_untouched():
    vp = VideoProcessor(vae_scale_factor = 8, do_normalize = True)
    video = _clip(torch.float32, normalized = True)
    assert torch.equal(_run(vp, video, "pt"), vp.postprocess_video(video, output_type = "pt"))
    assert "postprocess_video" not in vars(vp)
    with uint8_video_frames(types.SimpleNamespace()):
        pass


def test_restored_when_the_pipeline_raises():
    vp = VideoProcessor(vae_scale_factor = 8)
    with pytest.raises(RuntimeError):
        with uint8_video_frames(types.SimpleNamespace(video_processor = vp)):
            assert "postprocess_video" in vars(vp)
            raise RuntimeError("boom")
    assert "postprocess_video" not in vars(vp)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA / ROCm device")
def test_cuda_matches_numpy_and_lands_in_pinned_memory():
    vp = VideoProcessor(vae_scale_factor = 8, do_normalize = False)
    video = _clip(torch.float32, normalized = False, frames = 40, height = 64, width = 96)
    expected = _encode_video_uint8(vp.postprocess_video(video, output_type = "np"))
    got = _run(vp, video.cuda(), "np")
    assert got.device.type == "cpu" and got.is_pinned()
    assert np.array_equal(got.numpy(), expected)


def test_a_card_too_full_for_a_slice_takes_the_original_postprocess(monkeypatch):
    vp = VideoProcessor(vae_scale_factor = 8, do_normalize = False)
    video = _clip(torch.float32, normalized = False)
    expected = vp.postprocess_video(video, output_type = "np")

    def _oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(torch, "aminmax", _oom)
    got = _run(vp, video, "np")
    assert isinstance(got, np.ndarray) and np.array_equal(got, expected)
