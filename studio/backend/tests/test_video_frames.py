# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""video_frames: the device-side uint8 conversion must hand the encoder exactly the frames diffusers' np / pil export
produced, and give back the legacy object whenever it does not apply."""

import inspect
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from diffusers.video_processor import VideoProcessor  # noqa: E402

from core.inference.video_frames import legacy_output_type, to_uint8_frames  # noqa: E402


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
    # Values sitting exactly on the rounding boundaries (k + 0.5) / 255 and the ends of the range.
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("normalized", [False, True])
def test_matches_the_np_export(dtype, normalized):
    vp = VideoProcessor(vae_scale_factor = 8, do_normalize = normalized)
    video = _clip(dtype, normalized = normalized)
    expected = _encode_video_uint8(vp.postprocess_video(video, output_type = "np")[0])
    got = to_uint8_frames(vp.postprocess_video(video, output_type = "pt")[0], "np")
    assert got.dtype == torch.uint8 and got.is_contiguous()
    assert np.array_equal(got.numpy(), expected)


def test_matches_the_pil_export():
    vp = VideoProcessor(vae_scale_factor = 8, do_normalize = True)
    video = _clip(torch.bfloat16, normalized = True, seed = 3)
    expected = np.stack([np.array(im) for im in vp.postprocess_video(video, output_type = "pil")[0]])
    got = to_uint8_frames(vp.postprocess_video(video, output_type = "pt")[0], "pil")
    assert np.array_equal(got.numpy(), expected)


def test_every_float32_near_a_rounding_boundary():
    k = torch.arange(256, dtype = torch.float32)
    bits = torch.cat([(k + 0.5) / 255, k / 255]).clamp(0, 1).view(torch.int32)
    values = torch.cat([bits + d for d in range(-512, 513)]).view(torch.float32)
    values = values[(values >= 0) & (values <= 1)]
    n = values.numel() - values.numel() % 3
    frames = values[:n].view(-1, 3, 1, 1)
    got = to_uint8_frames(frames, "np").reshape(-1).numpy()
    expected = _encode_video_uint8(VaeImageProcessor.pt_to_numpy(frames)).reshape(-1)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("bad", [float("nan"), 1.5, -0.25])
@pytest.mark.parametrize("output_type", ["np", "pil"])
def test_out_of_range_rebuilds_the_legacy_object(bad, output_type):
    frames = _clip(torch.float32, normalized = False)[0].permute(1, 0, 2, 3).contiguous()
    frames[1, 0, 2, 3] = bad
    got = to_uint8_frames(frames, output_type)
    legacy = VaeImageProcessor.pt_to_numpy(frames)
    if output_type == "np":
        assert isinstance(got, np.ndarray)
        assert np.array_equal(got, legacy, equal_nan = True)
    else:
        assert isinstance(got, list) and len(got) == frames.shape[0]
        assert np.array_equal(np.array(got[1]), np.array(VaeImageProcessor.numpy_to_pil(legacy)[1]))


def test_non_tensor_frames_pass_through():
    frames = [object(), object()]
    assert to_uint8_frames(frames, "np") is frames


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA / ROCm device")
def test_cuda_matches_cpu_and_lands_in_pinned_memory():
    frames = _clip(torch.float32, normalized = False, frames = 40, height = 64, width = 96)[0].permute(
        1, 0, 2, 3
    )
    frames = frames.contiguous()
    got = to_uint8_frames(frames.cuda(), "np")
    assert got.device.type == "cpu" and got.is_pinned()
    assert torch.equal(got, to_uint8_frames(frames, "np"))


def test_legacy_output_type_reads_the_pipeline_default():
    def wan(prompt = None, output_type = "np"):
        pass

    def ltx(prompt = None, output_type = "pil"):
        pass

    def bare(prompt = None, **kwargs):
        pass

    def latent(prompt = None, output_type = "latent"):
        pass

    for fn, expected in ((wan, "np"), (ltx, "pil"), (bare, None), (latent, None)):
        assert legacy_output_type(None, inspect.signature(fn).parameters, False) == expected


def test_legacy_output_type_reads_the_modular_decode_input():
    param = types.SimpleNamespace(name = "output_type", default = "pil")
    other = types.SimpleNamespace(name = "prompt", default = None)
    pipe = types.SimpleNamespace(blocks = types.SimpleNamespace(inputs = [other, param]))
    assert legacy_output_type(pipe, {}, True) == "pil"
    assert legacy_output_type(types.SimpleNamespace(), {}, True) is None
