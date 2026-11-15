# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The U-Net QKV fuse drops the projections its fused processor no longer calls, bit-identically, and can rebuild them."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from core.inference import diffusion_speed as dspeed  # noqa: E402

ENV = "UNSLOTH_DIFFUSION_DROP_UNFUSED_QKV"


def _unet():
    from diffusers import UNet2DConditionModel
    torch.manual_seed(0)
    return UNet2DConditionModel(
        sample_size = 8,
        in_channels = 4,
        out_channels = 4,
        layers_per_block = 1,
        block_out_channels = (32, 64),
        down_block_types = ("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types = ("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim = 24,
        attention_head_dim = 8,
        norm_num_groups = 8,
    ).eval()


def _inputs():
    g = torch.Generator().manual_seed(1)
    return (
        torch.randn(1, 4, 8, 8, generator = g),
        torch.tensor([10]),
        torch.randn(1, 3, 24, generator = g),
    )


def _bytes(m) -> int:
    return sum(p.numel() * p.element_size() for p in m.parameters())


def _pipe(unet):
    return types.SimpleNamespace(
        unet = unet, vae = None, fuse_qkv_projections = unet.fuse_qkv_projections
    )


def _attn(unet):
    from diffusers.models.attention_processor import Attention
    return [m for m in unet.modules() if isinstance(m, Attention)]


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(ENV, raising = False)


def test_fuse_drops_the_unused_projections_and_output_is_bit_identical(monkeypatch):
    # the claim is about the drop, not the fuse: a fused U-Net that keeps its unfused copies (main) against one that
    # dropped them (the fused GEMM itself may differ from three GEMMs in the last bit on some BLAS builds)
    kept = _unet()
    monkeypatch.setenv(ENV, "0")
    dspeed._fuse_qkv(_pipe(kept), None)
    monkeypatch.delenv(ENV)
    with torch.no_grad():
        before = kept(*_inputs()).sample
    unet = _unet()
    unfused_bytes = _bytes(unet)
    assert dspeed._fuse_qkv(_pipe(unet), None) is True
    with torch.no_grad():
        after = unet(*_inputs()).sample
    assert torch.equal(before, after)
    # the fused weights replace the dropped ones byte for byte: nothing is held twice
    assert _bytes(unet) == unfused_bytes
    for m in _attn(unet):
        if m.is_cross_attention:
            assert hasattr(m, "to_q") and not hasattr(m, "to_k") and not hasattr(m, "to_v")
        else:
            assert not any(hasattr(m, n) for n in ("to_q", "to_k", "to_v"))


def test_restore_rebuilds_the_projections_bit_exactly():
    unet = _unet()
    originals = {
        (i, n): getattr(m, n).weight.detach().clone()
        for i, m in enumerate(_attn(unet))
        for n in ("to_q", "to_k", "to_v")
    }
    dspeed._fuse_qkv(_pipe(unet), None)
    assert dspeed.restore_unfused_projections(unet) == len(_attn(unet))
    for (i, n), w in originals.items():
        assert torch.equal(getattr(_attn(unet)[i], n).weight, w)
    assert dspeed.restore_unfused_projections(unet) == 0  # nothing left to rebuild


def test_kill_switch_keeps_them(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    unet = _unet()
    unfused_bytes = _bytes(unet)
    dspeed._fuse_qkv(_pipe(unet), None)
    assert all(hasattr(m, "to_k") for m in _attn(unet))
    assert _bytes(unet) > unfused_bytes


def test_a_peft_adapter_keeps_them():
    pytest.importorskip("peft")
    from peft import LoraConfig

    unet = _unet()
    unet.add_adapter(
        LoraConfig(r = 2, lora_alpha = 2, target_modules = ["to_q", "to_k", "to_v"]), adapter_name = "a"
    )
    unet.fuse_qkv_projections()
    assert dspeed.drop_unfused_projections(unet) == 0
    assert all(hasattr(m, "to_k") for m in _attn(unet))
