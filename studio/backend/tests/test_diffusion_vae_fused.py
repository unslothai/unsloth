# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""diffusion_vae_fused: bit-exact helpers (CPU), gating, and fused-vs-stock decode/encode on tiny random VAEs (CUDA)."""

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_vae_fused as F  # noqa: E402


def _cuda_triton() -> bool:
    return torch.cuda.is_available() and not getattr(torch.version, "hip", None) and F.runtime_ok()


needs_cuda = pytest.mark.skipif(not _cuda_triton(), reason = "needs NVIDIA CUDA + Triton")


# ---------------------------------------------------------------------------------------------------------------- CPU


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("extent", [1, 7, 32])
def test_blend_seam_bit_identical_to_stock_loops(dtype, extent):
    diffusers = pytest.importorskip("diffusers")
    wan = diffusers.AutoencoderKLWan
    g = torch.Generator().manual_seed(0)
    a = torch.randn(1, 3, 2, 40, 44, generator = g).to(dtype)
    b = torch.randn(1, 3, 2, 40, 44, generator = g).to(dtype)
    for stock, dim in ((wan.blend_v, -2), (wan.blend_h, -1)):
        assert torch.equal(
            stock(None, a, b.clone(), extent), F.blend_seam(a, b.clone(), extent, dim)
        )


def test_hv_causal_mask_matches_stock():
    mod = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15")
    stock = mod.HunyuanVideo15AttnBlock.prepare_causal_attention_mask(
        3, 5, torch.float32, "cpu", batch_size = 2
    )
    assert torch.equal(stock, F._hv_causal_mask(3, 5, torch.float32, "cpu", batch_size = 2))


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv(F.VAE_FUSED_ENV, "0")
    assert F.runtime_ok() is False
    assert F.will_install(object()) is False


def test_rocm_and_old_triton_keep_stock(monkeypatch):
    # the #11801 kernels are gated off on ROCm (missing add_rn / rint, wrong fp16 GroupNorm on gfx1151): same here
    monkeypatch.delenv(F.VAE_FUSED_ENV, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.4.0", raising = False)
    assert F.runtime_ok() is False
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    F._triton_version_ok.cache_clear()
    import triton

    monkeypatch.setattr(triton, "__version__", "3.2.0")
    try:
        assert F.runtime_ok() is False
    finally:
        F._triton_version_ok.cache_clear()


def test_windows_without_msvc_keeps_stock(monkeypatch):
    monkeypatch.setattr(F.sys, "platform", "win32")
    F._toolchain_ok.cache_clear()
    import core._msvc_env as msvc

    monkeypatch.setattr(msvc, "crt_headers_reachable", lambda: False)
    try:
        assert F._toolchain_ok() is False
    finally:
        F._toolchain_ok.cache_clear()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_blend_seam_4d_matches_autoencoder_kl(dtype):
    diffusers = pytest.importorskip("diffusers")
    akl = diffusers.AutoencoderKL
    g = torch.Generator().manual_seed(1)
    a = torch.randn(1, 3, 40, 44, generator = g).to(dtype)
    b = torch.randn(1, 3, 40, 44, generator = g).to(dtype)
    assert torch.equal(akl.blend_v(None, a, b.clone(), 16), F.blend_seam(a, b.clone(), 16, -2))
    assert torch.equal(akl.blend_h(None, a, b.clone(), 16), F.blend_seam(a, b.clone(), 16, -1))


def test_install_is_noop_without_runtime(monkeypatch):
    monkeypatch.setattr(F, "runtime_ok", lambda: False)

    class AutoencoderKL(torch.nn.Module):
        pass

    assert F.install(AutoencoderKL()) == 0


def test_uncovered_class_not_planned():
    class AutoencoderKLMiniMaxH3(torch.nn.Module):  # has its own fast path
        pass

    assert F.will_install(AutoencoderKLMiniMaxH3()) is False


def test_speed_layer_skips_vae_compile_when_fused(monkeypatch):
    from core.inference import diffusion_speed as S

    monkeypatch.delenv(S.COMPILE_VAE_ENV, raising = False)
    monkeypatch.setattr(F, "will_install", lambda vae: True)

    class Pipe:
        vae = object()
        unet = None

    assert S._fused_vae_planned(Pipe()) is True
    assert S._vae_decode_compile_allowed(Pipe(), "max") is False
    # forcing the compile keeps the old behaviour and skips the fused path
    monkeypatch.setenv(S.COMPILE_VAE_ENV, "1")
    assert S._fused_vae_planned(Pipe()) is False
    assert S._install_fused_vae(Pipe(), None) is False


# --------------------------------------------------------------------------------------------------------------- CUDA


def _psnr(a, b):
    mse = (a.float() - b.float()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(4.0 / mse)).item()


def _decode(vae, z):
    return vae.decode(z, return_dict = False)[0].float()


def _check(
    vae,
    z,
    *,
    min_psnr,
    encode_x = None,
):
    with torch.inference_mode():
        ref = _decode(vae, z)
        n = F.install(vae)
        assert n > 0
        out = _decode(vae, z)
        assert not any(
            getattr(m, "_unsloth_vae_fused_failed", False) for m in vae.modules()
        ), "fused path fell back"
        assert out.shape == ref.shape
        assert _psnr(out, ref) > min_psnr, _psnr(out, ref)
        if encode_x is not None:
            F.uninstall(vae)
            lat_ref = vae.encode(encode_x).latent_dist.mode().float()
            F.install(vae)
            lat = vae.encode(encode_x).latent_dist.mode().float()
            assert (lat - lat_ref).norm() / lat_ref.norm() < 2e-2
        F.uninstall(vae)
        assert torch.equal(_decode(vae, z), ref) or _psnr(_decode(vae, z), ref) > 80


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_autoencoder_kl_fused_matches_stock(dtype):
    from diffusers import AutoencoderKL

    torch.manual_seed(0)
    vae = AutoencoderKL(
        block_out_channels = (32, 64), down_block_types = ("DownEncoderBlock2D",) * 2,
        up_block_types = ("UpDecoderBlock2D",) * 2, latent_channels = 4, norm_num_groups = 32, layers_per_block = 1,
    ).cuda().to(dtype).eval()  # fmt: skip
    vae.to(memory_format = torch.channels_last)
    z = torch.randn(1, 4, 24, 20, device = "cuda", dtype = dtype)
    x = torch.rand(1, 3, 48, 40, device = "cuda", dtype = dtype) * 2 - 1
    _check(vae, z, min_psnr = 40 if dtype != torch.float32 else 70, encode_x = x)


@needs_cuda
@pytest.mark.parametrize("tiled", [False, True])
def test_wan_fused_matches_stock_across_chunks(tiled, monkeypatch):
    from diffusers import AutoencoderKLWan

    torch.manual_seed(0)
    vae = (
        AutoencoderKLWan(base_dim = 32, z_dim = 4, dim_mult = [1, 2, 2, 2], num_res_blocks = 1)
        .cuda()
        .half()
        .eval()
    )
    if tiled:
        vae.enable_tiling(tile_sample_min_height = 64, tile_sample_min_width = 64, tile_sample_stride_height = 48,
                          tile_sample_stride_width = 48)  # fmt: skip
        monkeypatch.setenv(F.TILE_BATCH_ENV, "3")
    z = torch.randn(
        1, 4, 4, 14, 18, device = "cuda", dtype = torch.float16
    )  # 4 latent frames: cached chunks
    x = torch.rand(1, 3, 5, 48, 64, device = "cuda", dtype = torch.float16) * 2 - 1
    _check(vae, z, min_psnr = 45, encode_x = None if tiled else x)


@needs_cuda
def test_wan22_residual_decoder_fused_matches_stock():
    # Wan-2.2 5B layout: residual up blocks (DupUp3D shortcut), patch_size 2, temporal-upsample resamples
    from diffusers import AutoencoderKLWan

    torch.manual_seed(0)
    vae = AutoencoderKLWan(
        base_dim = 32, decoder_base_dim = 32, z_dim = 8, dim_mult = [1, 2, 2, 2], num_res_blocks = 1, in_channels = 12,
        out_channels = 12, is_residual = True, patch_size = 2, scale_factor_spatial = 16,
        latents_mean = [0.0] * 8, latents_std = [1.0] * 8,
    ).cuda().half().eval()  # fmt: skip
    z = torch.randn(1, 8, 3, 6, 8, device = "cuda", dtype = torch.float16)
    _check(vae, z, min_psnr = 45)


@needs_cuda
def test_qwenimage_single_frame_fused_matches_stock():
    from diffusers import AutoencoderKLQwenImage

    torch.manual_seed(0)
    vae = (
        AutoencoderKLQwenImage(base_dim = 32, z_dim = 4, dim_mult = [1, 2, 2, 2], num_res_blocks = 1)
        .cuda()
        .bfloat16()
        .eval()
    )
    z = torch.randn(1, 4, 1, 16, 12, device = "cuda", dtype = torch.bfloat16)
    x = torch.rand(1, 3, 1, 64, 48, device = "cuda", dtype = torch.bfloat16) * 2 - 1
    _check(vae, z, min_psnr = 40, encode_x = x)


@needs_cuda
def test_hunyuanvideo15_fused_matches_stock():
    from diffusers import AutoencoderKLHunyuanVideo15

    torch.manual_seed(0)
    vae = AutoencoderKLHunyuanVideo15(
        in_channels = 3, out_channels = 3, latent_channels = 8, block_out_channels = (32, 64, 64, 64, 64),
        layers_per_block = 1, spatial_compression_ratio = 16, temporal_compression_ratio = 4,
    ).cuda().bfloat16().eval()  # fmt: skip
    z = torch.randn(1, 8, 3, 4, 5, device = "cuda", dtype = torch.bfloat16)
    _check(vae, z, min_psnr = 40)


@needs_cuda
def test_ltx2_fused_matches_stock():
    from diffusers import AutoencoderKLLTX2Video

    from core.inference.video_ltx2 import _VIDEO_VAE_CONFIG

    torch.manual_seed(0)
    cfg = dict(_VIDEO_VAE_CONFIG)  # the LTX-2.3 layout, narrowed
    cfg.update(
        latent_channels = 16, block_out_channels = (32, 64, 128, 128), decoder_block_out_channels = (32, 64, 64, 128),
        layers_per_block = (1, 1, 1, 1, 1), decoder_layers_per_block = (1, 1, 1, 1, 1),
    )  # fmt: skip
    vae = AutoencoderKLLTX2Video(**cfg).cuda().bfloat16().eval()
    z = torch.randn(1, 16, 3, 3, 4, device = "cuda", dtype = torch.bfloat16)
    _check(vae, z, min_psnr = 40)


@needs_cuda
def test_guard_falls_back_to_stock_for_good(monkeypatch):
    from diffusers import AutoencoderKL

    torch.manual_seed(0)
    vae = AutoencoderKL(
        block_out_channels = (32,), down_block_types = ("DownEncoderBlock2D",), up_block_types = ("UpDecoderBlock2D",),
        latent_channels = 4, norm_num_groups = 32, layers_per_block = 1,
    ).cuda().eval()  # fmt: skip
    z = torch.randn(1, 4, 8, 8, device = "cuda")
    with torch.inference_mode():
        ref = _decode(vae, z)
        F.install(vae)

        def boom(*a, **k):
            raise RuntimeError("kernel exploded")

        monkeypatch.setattr(F, "group_norm_act", boom)
        out = _decode(vae, z)
    assert torch.allclose(
        out, ref, atol = 5e-3
    )  # channels-last weights: another (TF32) cuDNN algorithm
    assert any(getattr(m, "_unsloth_vae_fused_failed", False) for m in vae.modules())


@needs_cuda
def test_guard_restores_causal_cache_state_on_fallback(monkeypatch):
    # a fast residual block that dies AFTER writing its first cache slot must not desync the stock retry
    from diffusers import AutoencoderKLWan

    torch.manual_seed(0)
    vae = (
        AutoencoderKLWan(base_dim = 32, z_dim = 4, dim_mult = [1, 2, 2, 2], num_res_blocks = 1)
        .cuda()
        .half()
        .eval()
    )
    z = torch.randn(1, 4, 3, 8, 8, device = "cuda", dtype = torch.float16)
    with torch.inference_mode():
        ref = _decode(vae, z)
        F.install(vae)
        real = F.causal_conv
        calls = {"n": 0}

        def flaky(
            conv,
            x,
            cache = None,
            **kw,
        ):
            calls["n"] += 1
            if (
                calls["n"] == 2
            ):  # conv2 of the first block: conv1 already advanced feat_idx and wrote a slot
                raise RuntimeError("boom")
            return real(conv, x, cache, **kw)

        monkeypatch.setattr(F, "causal_conv", flaky)
        out = _decode(vae, z)
    assert _psnr(out, ref) > 45
