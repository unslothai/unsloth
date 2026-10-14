# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""diffusion_vae_oom: the halved-tile retry ladder (CPU, fake OOMs) and the structural fused GroupNorm(+SiLU) match."""

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_vae_oom as O  # noqa: E402
from core.inference import diffusion_vae_fused as F  # noqa: E402


class _FakeOOM(RuntimeError):
    def __init__(self):
        super().__init__("CUDA out of memory. Tried to allocate 2.00 GiB")


class _WanLike(torch.nn.Module):
    """Stock-diffusers-shaped tiling attrs; decode OOMs while the tile side exceeds ``fits``."""

    spatial_compression_ratio = 8

    def __init__(self, fits: int):
        super().__init__()
        self.use_tiling = False
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192
        self.fits = fits
        self.seam_below = 0
        self.seen = []

    def enable_tiling(self):
        self.use_tiling = True

    def _decode(self, z, return_dict = True):
        side = z.shape[-1] * 8 if not self.use_tiling else self.tile_sample_min_height
        if z.shape[-1] == 64:  # the full latent; the quality probe decodes a smaller corner crop
            self.seen.append((self.use_tiling, self.tile_sample_min_height, self.tile_sample_stride_height))
        else:  # the probe crop always fits
            out = z.repeat_interleave(8, -1).repeat_interleave(8, -2) + 1
            # tiles under ``seam_below`` px decode with a visible error
            return (out + (1.0 if self.use_tiling and self.tile_sample_min_height < self.seam_below else 0.0),)
        if side > self.fits:
            raise _FakeOOM()
        return (z.repeat_interleave(8, -1).repeat_interleave(8, -2) + 1,)


def test_halved_keeps_overlap_fraction_and_floor():
    attrs = {"tile_sample_min_height": 256, "tile_sample_stride_height": 192, "tile_latent_min_size": 32}
    assert O.halved(attrs, 8, 1) == {
        "tile_sample_min_height": 128, "tile_sample_stride_height": 96, "tile_latent_min_size": 16,
    }  # fmt: skip
    assert O.halved(attrs, 8, 2)["tile_sample_min_height"] == 64  # the 8-latent floor
    assert O.halved(attrs, 8, 3) is None
    assert O.halved({}, 8, 1) is None


def test_halved_overlap_factor_geometry_tiles_exactly():
    """HunyuanImage-2.1: 384 px / 12-latent tiles at f=0.25. A plain halving (6 latents) steps 4 latents but keeps
    144 px per tile, so diffusers' stock tiled decode returns a too-large image; it lands on the 8-latent floor."""
    attrs = {"tile_sample_min_size": 384, "tile_latent_min_size": 12}
    assert O.halved(attrs, 32, 1, 0.25) == {"tile_sample_min_size": 256, "tile_latent_min_size": 8}
    assert O.halved(attrs, 32, 2, 0.25) is None
    kl = {"tile_sample_min_size": 1024, "tile_latent_min_size": 128}
    assert O.halved(kl, 8, 3, 0.25) == {"tile_sample_min_size": 128, "tile_latent_min_size": 16}
    assert O.halved(kl, 8, 4, 0.25)["tile_latent_min_size"] == 8 and O.halved(kl, 8, 5, 0.25) is None
    # stride-style geometry stays a whole number of latents (LTX-2: 32x)
    ltx = {"tile_sample_min_height": 512, "tile_sample_stride_height": 448}
    assert O.halved(ltx, 32, 1) == {"tile_sample_min_height": 256, "tile_sample_stride_height": 224}
    assert O.halved(ltx, 32, 2) is None


def test_untiled_oom_retries_tiled_then_halves_and_restores():
    vae = _WanLike(fits = 128)
    assert O.install(vae)
    z = torch.zeros(1, 4, 1, 64, 64)
    out = vae._decode(z)
    assert out[0].shape[-2:] == (512, 512)
    # untiled -> tiled at stock 256 -> halved 128 (stride 96)
    assert vae.seen == [(False, 256, 192), (True, 256, 192), (True, 128, 96)]
    assert vae.use_tiling is False and vae.tile_sample_min_height == 256
    assert vae.tile_sample_stride_height == 192
    stats = vae._unsloth_vae_oom_stats
    assert stats["tiled_retry"] == 1 and stats["recovered"] == 1


def test_tiled_level_is_memoised_per_shape():
    vae = _WanLike(fits = 64)
    vae.use_tiling = True
    O.install(vae)
    z = torch.zeros(1, 4, 1, 64, 64)
    vae._decode(z)
    assert [s[1] for s in vae.seen] == [256, 128, 64]
    vae.seen.clear()
    vae._decode(z)
    assert [s[1] for s in vae.seen] == [64]  # straight to the level that fit
    assert vae.tile_sample_min_height == 256


def test_halved_tiles_that_seam_are_refused():
    vae = _WanLike(fits = 64)
    vae.seam_below = 128  # 64 px tiles fit but seam; 128 px tiles are clean but OOM
    vae.use_tiling = True
    O.install(vae)
    with pytest.raises(RuntimeError, match = "visible seams"):
        vae._decode(torch.zeros(1, 4, 1, 64, 64))
    stats = vae._unsloth_vae_oom_stats
    assert stats["quality_refused"] == 1 and stats["probe_psnr"][2] < O.MIN_PSNR_DB
    assert stats["probe_psnr"][1] >= O.MIN_PSNR_DB
    assert vae.tile_sample_min_height == 256


def test_floor_raises_and_non_oom_propagates():
    vae = _WanLike(fits = 1)
    vae.use_tiling = True
    O.install(vae)
    with pytest.raises(RuntimeError, match = "every tile size"):
        vae._decode(torch.zeros(1, 4, 1, 64, 64))
    assert vae.tile_sample_min_height == 256

    vae = _WanLike(fits = 10**6)
    O.install(vae)
    vae._decode = types.MethodType(lambda self, z: (_ for _ in ()).throw(ValueError("x")), vae)
    O.install(vae)
    with pytest.raises(ValueError):
        vae._decode(torch.zeros(1))


def test_untiled_gate_keeps_its_own_fallback():
    vae = _WanLike(fits = 128)
    O.install(vae)

    def gate(*a, **k):
        return None

    gate._unsloth_untiled_decode = True
    vae.decode = gate
    with pytest.raises(RuntimeError, match = "out of memory"):
        vae._decode(torch.zeros(1, 4, 1, 64, 64))
    assert vae.seen == [(False, 256, 192)]


def test_kill_switch(monkeypatch):
    monkeypatch.setenv(O.OOM_RETRY_ENV, "0")
    vae = _WanLike(fits = 128)
    assert O.install(vae) is False
    assert "_decode" not in vae.__dict__


def test_uninstall_restores_class_decode():
    vae = _WanLike(fits = 10**6)
    O.install(vae)
    assert "_decode" in vae.__dict__
    O.uninstall(vae)
    assert "_decode" not in vae.__dict__


# -- structural fused GroupNorm(+SiLU) --------------------------------------------------------------------------


class _Block(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.act = torch.nn.SiLU()
        self.norm1 = torch.nn.GroupNorm(4, c)
        self.conv = torch.nn.Conv2d(c, c, 3, padding = 1)
        self.norm2 = torch.nn.GroupNorm(4, c)

    def forward(self, x):
        h = self.act(self.norm1(x))  # paired
        h = self.conv(h)
        h2 = self.norm2(h)  # NOT paired: its output also feeds the residual
        return x + self.act(h2) + h2


class _Dec(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = _Block(c)
        self.norm_out = torch.nn.GroupNorm(4, c)
        self.out_act = torch.nn.SiLU()
        self.attn_norm = torch.nn.GroupNorm(4, c)

    def forward(self, x):
        x = self.block(x)
        x = x + self.attn_norm(x)  # GroupNorm without SiLU
        return self.out_act(self.norm_out(x))


class AutoencoderKLSomethingNew(torch.nn.Module):
    def __init__(self, c = 8):
        super().__init__()
        self.decoder = _Dec(c)

    def _decode(self, z, return_dict = True):
        return (self.decoder(z),)


def test_dataflow_trace_pairs_only_exclusive_group_norm_silu():
    vae = AutoencoderKLSomethingNew()
    norms = F._plain_group_norms(vae.decoder)
    silus = F._plain_silus(vae.decoder)
    calls, consumed, _ = F._trace_group_norm_silu(
        norms, silus, lambda: vae._decode(torch.randn(1, 8, 6, 6))
    )
    names = {id(m): n for n, m in vae.decoder.named_modules()}
    got = {names[id(m)]: (c, k) for m, c, k in zip(norms, calls, consumed)}
    assert got["block.norm1"] == (1, 1)
    assert got["norm_out"] == (1, 1)
    assert got["attn_norm"] == (1, 0)
    # norm2's output reaches the SiLU AND the residual: the trace sees the SiLU, the probe check must catch the rest
    assert got["block.norm2"] == (1, 1)


def _install_on_cpu(monkeypatch, vae):
    monkeypatch.setattr(F, "runtime_ok", lambda: True)
    monkeypatch.delenv(F.STRUCTURAL_ENV, raising = False)
    n = F.install_structural(vae)
    assert n == 4
    state = vae._unsloth_vae_structural
    # calibrate as on the first CUDA decode (CPU tensors take the reference GroupNorm, so it is exact here)
    z = torch.randn(1, 8, 6, 6)
    inner = vae.__dict__["_unsloth_vae_structural_inner"]
    with torch.no_grad():
        calls, consumed, ref = F._trace_group_norm_silu(
            F._plain_group_norms(vae.decoder), F._plain_silus(vae.decoder), lambda: inner(z)[0]
        )
    return state, z, ref


def test_structural_patch_is_exact_when_pairing_is_right(monkeypatch):
    vae = AutoencoderKLSomethingNew()
    state, z, ref = _install_on_cpu(monkeypatch, vae)
    norms = F._plain_group_norms(vae.decoder)
    silus = F._plain_silus(vae.decoder)
    names = {id(m): n for n, m in vae.decoder.named_modules()}
    paired = [names[id(m)] in ("block.norm1", "norm_out") or (False if names[id(m)] else None) for m in norms]
    F._structural_patch(norms, paired, silus, state, None)
    with torch.no_grad():
        out = vae.__dict__["_unsloth_vae_structural_inner"](z)[0]
    assert torch.allclose(out, ref, atol = 1e-6)
    assert state.calls == 4
    F._structural_unpatch(norms, silus)
    assert not any("forward" in m.__dict__ for m in (*norms, *silus))


def test_structural_probe_rejects_a_wrong_pairing(monkeypatch):
    """norm2 looks paired to the trace but its output also feeds a residual: the probe check rejects the patch."""
    vae = AutoencoderKLSomethingNew()
    monkeypatch.setattr(F, "runtime_ok", lambda: True)
    F.install_structural(vae)
    state = vae._unsloth_vae_structural
    z = torch.randn(1, 8, 6, 6) * 3
    # route the first decode through calibration on CPU
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(F, "_gn_fusable", lambda x, norm: False)
    with torch.no_grad():
        vae._decode(z)
    assert state.status == "rejected"
    assert not any("forward" in m.__dict__ for m in vae.decoder.modules())


def test_structural_skips_measured_classes_and_kill_switch(monkeypatch):
    class AutoencoderKL(AutoencoderKLSomethingNew):
        pass

    assert F.structural_candidate(AutoencoderKL()) is False
    assert F.structural_candidate(AutoencoderKLSomethingNew()) is True
    monkeypatch.setenv(F.STRUCTURAL_ENV, "0")
    assert F.structural_candidate(AutoencoderKLSomethingNew()) is False
    monkeypatch.delenv(F.STRUCTURAL_ENV)
    vae = AutoencoderKLSomethingNew()
    vae._unsloth_decode_blocks_bypassed = True
    assert F.structural_candidate(vae) is False


def _cuda_triton() -> bool:
    return torch.cuda.is_available() and not getattr(torch.version, "hip", None) and F.runtime_ok()


needs_cuda = pytest.mark.skipif(not _cuda_triton(), reason = "needs NVIDIA CUDA + Triton")


def _psnr(a, b):
    mse = (a.float() - b.float()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(4.0 / mse)).item()


def _tiny(name):
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    if name == "AutoencoderKLHunyuanImage":
        return diffusers.AutoencoderKLHunyuanImage(
            in_channels = 3, out_channels = 3, latent_channels = 8, block_out_channels = (32, 64, 64),
            layers_per_block = 1, spatial_compression_ratio = 4, sample_size = 64,
        ), (1, 8, 12, 10)  # fmt: skip
    if name == "AutoencoderKLHunyuanVideo":
        return diffusers.AutoencoderKLHunyuanVideo(
            in_channels = 3, out_channels = 3, latent_channels = 4,
            down_block_types = ("HunyuanVideoDownBlock3D",) * 3, up_block_types = ("HunyuanVideoUpBlock3D",) * 3,
            block_out_channels = (32, 32, 32), layers_per_block = 1, norm_num_groups = 8,
            spatial_compression_ratio = 4, temporal_compression_ratio = 4,
        ), (1, 4, 2, 10, 12)  # fmt: skip
    raise AssertionError(name)


@needs_cuda
@pytest.mark.parametrize("name", ["AutoencoderKLHunyuanImage", "AutoencoderKLHunyuanVideo"])
def test_structural_engages_on_uncovered_class(name):
    vae, shape = _tiny(name)
    vae = vae.cuda().to(torch.bfloat16).eval()
    z = torch.randn(*shape, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        ref = vae.decode(z, return_dict = False)[0].float()
        assert F.will_install(vae) and F.install(vae) > 0
        out = vae.decode(z, return_dict = False)[0].float()
        state = vae._unsloth_vae_structural
        assert state.status == "ok", vars(state)
        assert state.gn_silu > 0 and state.calls > 0  # engaged on the real call
        assert _psnr(out, ref) > 35, _psnr(out, ref)
        F.uninstall(vae)
        assert "_decode" not in vae.__dict__
        assert torch.equal(vae.decode(z, return_dict = False)[0].float(), ref)


def test_wide_tiles_defer_to_halved_stock_tiles_on_oom(monkeypatch):
    """Real tiny diffusers QwenImage VAE behind Studio's wide tiles: every tile wider than 16 latents OOMs; the wide
    tiles (32-latent floor) and the stock 32-latent tiles fail, the halved stock tiles (16 latents) decode."""
    diffusers = pytest.importorskip("diffusers")
    from core.inference import diffusion_vae_tiling as vt

    monkeypatch.delenv(vt.WIDE_TILES_ENV, raising = False)
    monkeypatch.setenv(vt.MAX_TILE_ENV, "1")
    torch.manual_seed(0)
    vae = diffusers.AutoencoderKLQwenImage(
        base_dim = 8, z_dim = 4, dim_mult = [1, 2, 4, 4], num_res_blocks = 1, attn_scales = [],
        temperal_downsample = [False, True, True], latents_mean = [0.0] * 4, latents_std = [1.0] * 4,
    ).eval()  # fmt: skip
    z = torch.randn(1, 4, 1, 48, 48, generator = torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = vae.decode(z).sample
    vae.enable_tiling()
    assert vt.install(vae)
    assert O.install(vae)
    stock_tile = vae.tile_sample_min_height
    sides = []
    real_decoder = vae.decoder.forward

    def decoder(x, *a, **k):
        sides.append(int(x.shape[-1]))
        if x.shape[-1] > 16:
            raise _FakeOOM()
        return real_decoder(x, *a, **k)

    monkeypatch.setattr(vae.decoder, "forward", decoder)
    with torch.no_grad():
        out = vae.decode(z).sample
    assert out.shape == ref.shape and torch.isfinite(out).all()
    assert max(sides) > 16 and sides[-1] <= 16  # the wide / stock tiles were tried first
    stats = vae._unsloth_vae_oom_stats
    assert stats["halvings"] == 1 and stats["recovered"] == 1
    assert vae.tile_sample_min_height == stock_tile and not vae.__dict__.get("_unsloth_vae_oom_halvings")
    # the halved tiles still blend: no hard seam against the untiled decode
    assert (out - ref).abs().mean() < 0.1
