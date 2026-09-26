# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""fp16 decode for a force_upcast SDXL VAE: the rescale is exact, the flag flips, the encode keeps fp32 math, and a
non-finite decode falls back to fp32 for good."""

from __future__ import annotations

import copy
import logging
import types

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from core.inference import diffusion_vae_fp16 as vf  # noqa: E402

CUDA = types.SimpleNamespace(device = "cuda")


def _tiny_vae(force_upcast = True):
    torch.manual_seed(0)
    return diffusers.AutoencoderKL(
        in_channels = 3,
        out_channels = 3,
        down_block_types = ("DownEncoderBlock2D",) * 3,
        up_block_types = ("UpDecoderBlock2D",) * 3,
        block_out_channels = (16, 32, 32),
        layers_per_block = 1,
        latent_channels = 4,
        norm_num_groups = 8,
        force_upcast = force_upcast,
    ).eval()


def _pipe(vae, unet = True):
    return types.SimpleNamespace(vae = vae, unet = object() if unet else None)


def test_rescaled_decoder_is_the_same_function_in_fp32():
    vae = _tiny_vae()
    ref = copy.deepcopy(vae)
    z = torch.randn(1, 4, 8, 8)
    vae.to(torch.float16)
    assert vf.enable_fp16_vae_decode(_pipe(vae), CUDA) is True
    assert vae.config.force_upcast is False and vae._unsloth_fp16_decode is True
    # Rescaled convs really moved (else the equality below is vacuous).
    entry = vae.decoder.up_blocks[0].upsamplers[0].conv
    assert not torch.equal(
        entry.weight.float(), ref.decoder.up_blocks[0].upsamplers[0].conv.weight.half().float()
    )
    vae.to(torch.float32)
    ref.to(torch.float16).to(torch.float32)  # the same fp16-rounded weights, unscaled
    with torch.no_grad():
        got = vae.decode(z, return_dict = False)[0]
        want = ref.decode(z, return_dict = False)[0]
    # Only the fp16 rounding of the smallest rescaled weights remains; a wrong shortcut or eps scale is off by ~0.4.
    torch.testing.assert_close(got, want, rtol = 0, atol = 1e-3)


def test_encode_keeps_fp32_math_and_restores_fp16_weights():
    vae = _tiny_vae()
    ref = copy.deepcopy(vae).to(torch.float16).to(torch.float32)
    vae.to(torch.float16)
    vf.enable_fp16_vae_decode(_pipe(vae), CUDA)
    x = torch.randn(1, 3, 32, 32).to(torch.float16)
    with torch.no_grad():
        got = vae.encode(x).latent_dist.mean
        want = ref.encode(x.float()).latent_dist.mean
    assert got.dtype is torch.float32
    torch.testing.assert_close(got, want)
    assert (
        vae.encoder.conv_in.weight.dtype is torch.float16
        and vae.quant_conv.weight.dtype is torch.float16
    )


def test_non_finite_decode_reruns_fp32_and_stays_there(monkeypatch, caplog):
    vae = _tiny_vae().to(torch.float16)
    calls = []

    def fake_decode(z, return_dict = True):
        calls.append(z.dtype)
        dtype = vae.decoder.conv_in.weight.dtype
        fill = float("nan") if dtype is torch.float16 else 0.5
        return (torch.full((1, 3, 4, 4), fill, dtype = dtype),)

    vae.decode = fake_decode
    logger = logging.getLogger("vae_fp16_test")
    assert vf.enable_fp16_vae_decode(_pipe(vae), CUDA, logger = logger) is True
    with caplog.at_level(logging.WARNING, logger = "vae_fp16_test"):
        out = vae.decode(torch.zeros(1, 4, 2, 2, dtype = torch.float16), return_dict = False)[0]
    assert torch.isfinite(out).all() and out.dtype is torch.float32
    assert calls == [torch.float16, torch.float32]
    assert vae.config.force_upcast is True
    assert vae.decoder.conv_in.weight.dtype is torch.float16
    assert "not finite" in caplog.text
    vae.decode(torch.zeros(1, 4, 2, 2, dtype = torch.float16), return_dict = False)
    assert calls[-1] is torch.float16 and len(calls) == 3


@pytest.mark.parametrize(
    "case",
    [
        "no_unet",
        "not_cuda",
        "bf16",
        "fp32",
        "no_force_upcast",
        "other_class",
        "attention_in_up_block",
    ],
)
def test_declines_and_leaves_the_vae_untouched(case):
    vae = _tiny_vae(force_upcast = case != "no_force_upcast")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}.get(case, torch.float16)
    vae.to(dtype)
    if case == "attention_in_up_block":
        vae.decoder.up_blocks[1].attentions = torch.nn.ModuleList([torch.nn.Identity()])
    if case == "other_class":
        vae.__class__ = type("AutoencoderKLOther", (type(vae),), {})
    before = {k: v.clone() for k, v in vae.state_dict().items()}
    target = types.SimpleNamespace(device = "mps") if case == "not_cuda" else CUDA
    assert vf.enable_fp16_vae_decode(_pipe(vae, unet = case != "no_unet"), target) is False
    assert all(torch.equal(before[k], v) for k, v in vae.state_dict().items())
    assert vae.config.force_upcast is (case != "no_force_upcast")
    assert "decode" not in vae.__dict__


def test_second_call_is_a_no_op():
    vae = _tiny_vae().to(torch.float16)
    pipe = _pipe(vae)
    assert vf.enable_fp16_vae_decode(pipe, CUDA) is True
    after_first = {k: v.clone() for k, v in vae.state_dict().items()}
    assert vf.enable_fp16_vae_decode(pipe, CUDA) is True
    assert all(torch.equal(after_first[k], v) for k, v in vae.state_dict().items())
