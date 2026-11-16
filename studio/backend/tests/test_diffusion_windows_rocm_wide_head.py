# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows ROCm: the fused SDPA kernels take a head dim of 512 (the VAE's single-head attention) and compute it wrong,
so such calls must go to the exact math path, in inference and in the LoRA trainer's latent encode."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_vae_attn_chunked as C  # noqa: E402


def _cuda_q(head_dim):
    return types.SimpleNamespace(
        device = types.SimpleNamespace(type = "cuda"),
        dtype = torch.float32,
        shape = (1, 1, 4096, head_dim),
        requires_grad = False,
    )


@pytest.fixture
def rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.14.0", raising = False)


@pytest.mark.parametrize("head_dim", [384, 512])
def test_windows_rocm_wide_head_is_not_a_fused_call(monkeypatch, rocm, head_dim):
    monkeypatch.setattr(sys, "platform", "win32")
    q = _cuda_q(head_dim)
    assert C.fused_sdpa_available(q, q, q) is False


@pytest.mark.parametrize("platform, head_dim", [("win32", 128), ("win32", 256), ("linux", 512)])
def test_other_heads_and_platforms_keep_the_kernel_query(monkeypatch, rocm, platform, head_dim):
    # A fake query makes torch's own SDPAParams query raise, which counts as "a fused kernel takes it".
    monkeypatch.setattr(sys, "platform", platform)
    q = _cuda_q(head_dim)
    assert C.fused_sdpa_available(q, q, q) is True


def test_windows_cuda_is_unaffected(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch, "__version__", "2.11.0+cu130")
    monkeypatch.setattr(sys, "platform", "win32")
    q = _cuda_q(512)
    assert C.fused_sdpa_available(q, q, q) is True


def test_dit_trainer_routes_its_vae_attention_on_rocm(rocm):
    diffusers = pytest.importorskip("diffusers")
    from core.training import diffusion_dit_trainer as dit

    torch.manual_seed(0)
    vae = diffusers.AutoencoderKL(
        in_channels = 3,
        out_channels = 3,
        down_block_types = ("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types = ("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels = (32, 64),
        layers_per_block = 1,
        latent_channels = 4,
        norm_num_groups = 8,
    ).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        ref = vae.encode(x).latent_dist.mode()
    assert dit._guard_vae_attention(vae) > 0
    patched = [m for m in vae.modules() if getattr(m, "_unsloth_vae_attn_chunked", False)]
    assert patched
    with torch.no_grad():
        # Exact: the routed path is the same math on CPU.
        assert torch.equal(vae.encode(x).latent_dist.mode(), ref)


class _UNet2DConditionModel:
    pass


@pytest.mark.parametrize("platform, allowed", [("win32", False), ("linux", True)])
def test_vae_decode_compile_stays_eager_on_windows_rocm(monkeypatch, rocm, platform, allowed):
    # A compiled decode bypasses the eager routing; a U-Net pipe otherwise always compiles its VAE decode.
    from core.inference import diffusion_speed as S

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(S, "_fused_vae_planned", lambda pipe: False)
    monkeypatch.setattr(S, "_UNET_WHOLE_COMPILE", frozenset({"_UNet2DConditionModel"}))
    pipe = types.SimpleNamespace(unet = _UNet2DConditionModel(), vae = object())
    assert S._vae_decode_compile_allowed(pipe, "default") is allowed
