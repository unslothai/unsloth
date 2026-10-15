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

    def _decode(
        self,
        z,
        return_dict = True,
    ):
        side = z.shape[-1] * 8 if not self.use_tiling else self.tile_sample_min_height
        if z.shape[-1] == 64:  # the full latent; the quality probe decodes a smaller corner crop
            self.seen.append(
                (self.use_tiling, self.tile_sample_min_height, self.tile_sample_stride_height)
            )
        else:  # the probe crop always fits
            out = z.repeat_interleave(8, -1).repeat_interleave(8, -2) + 1
            # tiles under ``seam_below`` px decode with a visible error
            return (
                out
                + (
                    1.0
                    if self.use_tiling and self.tile_sample_min_height < self.seam_below
                    else 0.0
                ),
            )
        if side > self.fits:
            raise _FakeOOM()
        return (z.repeat_interleave(8, -1).repeat_interleave(8, -2) + 1,)


def test_halved_keeps_overlap_fraction_and_floor():
    attrs = {
        "tile_sample_min_height": 256,
        "tile_sample_stride_height": 192,
        "tile_latent_min_size": 32,
    }
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
    assert (
        O.halved(kl, 8, 4, 0.25)["tile_latent_min_size"] == 8 and O.halved(kl, 8, 5, 0.25) is None
    )
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

    def _decode(
        self,
        z,
        return_dict = True,
    ):
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
    paired = [
        names[id(m)] in ("block.norm1", "norm_out") or (False if names[id(m)] else None)
        for m in norms
    ]
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
    assert vae.tile_sample_min_height == stock_tile and not vae.__dict__.get(
        "_unsloth_vae_oom_halvings"
    )
    # the halved tiles still blend: no hard seam against the untiled decode
    assert (out - ref).abs().mean() < 0.1


# -- PR review round 1 regressions ------------------------------------------------------------------------------


class _StaleCacheWanLike(_WanLike):
    """A failed full-size attempt leaves a stale frame cache; any decode before ``clear_cache`` runs out of memory."""

    def __init__(self, fits):
        super().__init__(fits)
        self.stale = False

    def clear_cache(self):
        self.stale = False

    def _decode(
        self,
        z,
        return_dict = True,
    ):
        if self.stale:
            raise _FakeOOM()
        try:
            return super()._decode(z, return_dict)
        except _FakeOOM:
            self.stale = True
            raise


def test_cache_released_before_the_quality_probe():
    """Item 1: the probe of the next level must not run against the failed attempt's stale cache. Before the fix the
    8-latent level's probe ran out of memory and was refused, though 64 px tiles fit."""
    vae = _StaleCacheWanLike(fits = 64)
    vae.use_tiling = True
    O.install(vae)
    out = vae._decode(torch.zeros(1, 4, 1, 64, 64))
    assert out[0].shape[-2:] == (512, 512)
    stats = vae._unsloth_vae_oom_stats
    assert stats["quality_refused"] == 0 and stats["recovered"] == 1
    assert stats["probe_psnr"][2] == 99.0


class _WrongCropWanLike(_WanLike):
    """At 128 px tiles the tiled probe crop comes back a different size than the untiled crop."""

    def _decode(
        self,
        z,
        return_dict = True,
    ):
        out = super()._decode(z, return_dict)
        if z.shape[-1] != 64 and self.use_tiling and self.tile_sample_min_height == 128:
            return (out[0][..., :-8],)
        return out


def test_probe_shape_mismatch_falls_through_to_the_next_level():
    """Item 2: a mismatched probe crop marks that level unusable and the ladder moves on. Before the fix the
    subtraction raised a non-OOM RuntimeError out of the decode."""
    vae = _WrongCropWanLike(fits = 64)
    vae.use_tiling = True
    O.install(vae)
    out = vae._decode(torch.zeros(1, 4, 1, 64, 64))
    assert out[0].shape[-2:] == (512, 512)
    stats = vae._unsloth_vae_oom_stats
    assert stats["probe_psnr"][1] == "shape" and stats["recovered"] == 1
    assert [s[1] for s in vae.seen] == [
        256,
        64,
    ]  # the unusable 128 px level was never attempted at full size


def test_untiled_gate_found_through_wrapped_chain():
    """Item 3: the ROCm bf16 decode wraps the untiled gate and exposes it only via ``__wrapped__``."""
    vae = _WanLike(fits = 128)
    O.install(vae)

    def gate(*a, **k):
        return None

    gate._unsloth_untiled_decode = True

    def bf16_decode(*a, **k):
        return gate(*a, **k)

    bf16_decode.__wrapped__ = gate
    vae.decode = bf16_decode
    with pytest.raises(RuntimeError, match = "out of memory"):
        vae._decode(torch.zeros(1, 4, 1, 64, 64))
    assert vae.seen == [(False, 256, 192)]  # no tiled retry of its own: the gate owns it


class AutoencoderKLTileableNew(AutoencoderKLSomethingNew):
    spatial_compression_ratio = 8

    def __init__(self, c = 8):
        super().__init__(c)
        self.use_tiling = False
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192
        self.decoder.proj3d = torch.nn.Conv3d(
            c, c, 3
        )  # a 3-D conv the channels_last_3d relayout touches

    def enable_tiling(self):
        self.use_tiling = True


def _layers(vae):
    fn, out = vae.__dict__.get("_decode"), []
    while fn is not None and len(out) < 8:
        out.append(fn)
        fn = getattr(fn, "__wrapped__", None)
    return out


def test_uninstall_unwraps_structural_layer_under_the_oom_wrapper(monkeypatch):
    """Item 4: with the OOM retry on top, uninstall must still remove the structural calibrator underneath (and keep
    the retry). Before the fix the calibrator stayed and its saved state was dropped."""
    monkeypatch.setattr(F, "runtime_ok", lambda: True)
    vae = AutoencoderKLTileableNew()
    assert F.install_structural(vae) > 0
    assert O.install(vae)
    assert any(getattr(f, "_unsloth_vae_fused", False) for f in _layers(vae))
    F.uninstall(vae)
    layers = _layers(vae)
    assert not any(getattr(f, "_unsloth_vae_fused", False) for f in layers), layers
    assert getattr(vae.__dict__.get("_decode"), "_unsloth_vae_oom_retry", False)
    assert "_unsloth_vae_structural_inner" not in vae.__dict__
    with torch.no_grad():
        out = vae._decode(torch.randn(1, 8, 6, 6))
    assert out[0].shape == (1, 8, 6, 6)


def test_rejected_structural_fusion_restores_contiguous_layout(monkeypatch):
    """Item 5: the speed layer relays 3-D decode convs to channels_last_3d because fusion is planned; a rejected
    calibration must put them back. Before the fix the weights stayed channels_last_3d."""
    from core.inference import diffusion_speed as S

    monkeypatch.delenv(S.VAE_CHANNELS_LAST_3D_ENV, raising = False)
    monkeypatch.setattr(F, "runtime_ok", lambda: True)
    vae = AutoencoderKLTileableNew()
    w = vae.decoder.proj3d.weight
    assert S._vae_channels_last_3d(vae, None, fused = True)
    assert w.is_contiguous(memory_format = torch.channels_last_3d) and not w.is_contiguous()
    F.install_structural(vae)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(F, "_gn_fusable", lambda x, norm: False)
    with torch.no_grad():
        vae._decode(torch.randn(1, 8, 6, 6) * 3)
    assert vae._unsloth_vae_structural.status == "rejected"
    assert vae.decoder.proj3d.weight.is_contiguous()


def test_halving_continues_on_the_axis_above_the_floor():
    """An axis already at the 8-latent floor keeps its size while the other keeps halving."""
    attrs = {"tile_sample_min_height": 256, "tile_sample_min_width": 128}
    assert O.halved(attrs, 8, 1) == {"tile_sample_min_height": 128, "tile_sample_min_width": 64}
    assert O.halved(attrs, 8, 2) == {"tile_sample_min_height": 64, "tile_sample_min_width": 64}
    assert O.halved(attrs, 8, 3) is None


class _UntiledProbeOOMWanLike(_WanLike):
    """The untiled reference crop of the quality probe never fits."""

    def __init__(self, fits):
        super().__init__(fits)
        self.untiled_probes = 0

    def _decode(
        self,
        z,
        return_dict = True,
    ):
        if z.shape[-1] != 64 and not self.use_tiling:
            self.untiled_probes += 1
            raise _FakeOOM()
        return super()._decode(z, return_dict)


def test_unprobed_level_is_remembered_with_the_working_level():
    """A level allowed after its probe ran out of memory is not probed again once it is the remembered level."""
    vae = _UntiledProbeOOMWanLike(fits = 128)
    vae.use_tiling = True
    O.install(vae)
    z = torch.zeros(1, 4, 1, 64, 64)
    assert vae._decode(z)[0].shape[-2:] == (512, 512)
    assert vae.untiled_probes == 1 and vae._unsloth_vae_oom_stats["probe_psnr"][1] is None
    vae.seen.clear()
    assert vae._decode(z)[0].shape[-2:] == (512, 512)
    assert vae.untiled_probes == 1
    assert [s[1] for s in vae.seen] == [128]


def test_ratio_from_tile_geometry_when_nothing_else_publishes_it():
    class _Bare:
        tile_sample_min_size = 512
        tile_latent_min_size = 32

    assert O._spatial_ratio(_Bare()) == 16


# -- PR review round 3 regressions ------------------------------------------------------------------------------


class _ShapeLog(_WanLike):
    """Full latents are >= 48 latents wide; narrower calls are probe crops. Logs every call."""

    def __init__(self, fits):
        super().__init__(fits)
        self.calls = []

    def _decode(
        self,
        z,
        return_dict = True,
    ):
        full = z.shape[-1] >= 48
        self.calls.append((full, self.use_tiling, self.tile_sample_min_height, tuple(z.shape[-2:])))
        side = z.shape[-1] * 8 if not self.use_tiling else self.tile_sample_min_height
        if full and side > self.fits:
            raise _FakeOOM()
        return (z.repeat_interleave(8, -1).repeat_interleave(8, -2) + 1,)


def _probes(vae):
    return [
        c for c in vae.calls if not c[0] and not c[1]
    ]  # untiled decodes of a crop = probe references


def test_quality_verdict_is_kept_per_shape():
    """Round 3 item 1: a second latent shape is probed again; a known shape's memoised level stays probe-free."""
    vae = _ShapeLog(fits = 128)
    vae.use_tiling = True
    O.install(vae)
    vae._decode(torch.zeros(1, 4, 1, 64, 64))
    assert len(_probes(vae)) == 1
    vae._decode(torch.zeros(1, 4, 1, 48, 48))
    assert len(_probes(vae)) == 2  # before: the 64x64 verdict was reused for 48x48
    vae.calls.clear()
    vae._decode(torch.zeros(1, 4, 1, 64, 64))
    assert _probes(vae) == [] and [c[2] for c in vae.calls] == [128]


@pytest.mark.parametrize("fits, level_side", [(256, 256), (128, 128)])
def test_untiled_caller_reuses_the_remembered_level(fits, level_side):
    """Round 3 item 3: an untiled caller that recovered through the stock tiles (256) or halved tiles (128) goes
    straight to that size next time instead of retrying the untiled decode that failed."""
    vae = _ShapeLog(fits = fits)
    O.install(vae)
    z = torch.zeros(1, 4, 1, 64, 64)
    vae._decode(z)
    vae.calls.clear()
    out = vae._decode(z)
    assert out[0].shape[-2:] == (512, 512)
    full = [c for c in vae.calls if c[0]]
    assert full == [(True, True, level_side, (64, 64))], full
    assert vae.use_tiling is False and vae.tile_sample_min_height == 256
    # a tiled caller of the same shape keeps its own memo
    vae.use_tiling = True
    vae.calls.clear()
    vae._decode(z)
    assert [c for c in vae.calls if c[0]][0][2] == 256


class _NarrowWidth(_ShapeLog):
    def __init__(self, fits):
        super().__init__(fits)
        self.tile_sample_min_width = 128
        self.tile_sample_stride_width = 96


def test_probe_crop_is_sized_per_axis():
    """Round 3 item 4: level 1 halves height to 16 latents and width to 8; the crop spans two tiles on each axis
    (32 x 16), not two of the smaller one on both (16 x 16)."""
    vae = _NarrowWidth(fits = 128)
    vae.use_tiling = True
    O.install(vae)
    vae._decode(torch.zeros(1, 4, 1, 64, 64))
    assert [c[3] for c in _probes(vae)] == [(32, 16)]


def test_uninstall_restores_contiguous_layout(monkeypatch):
    """Round 3 item 2: uninstall undoes the channels_last_3d relayout recorded for the fused passes."""
    from core.inference import diffusion_speed as S

    monkeypatch.delenv(S.VAE_CHANNELS_LAST_3D_ENV, raising = False)
    monkeypatch.setattr(F, "runtime_ok", lambda: True)
    vae = AutoencoderKLTileableNew()
    assert S._vae_channels_last_3d(vae, None, fused = True)
    F.install_structural(vae)
    assert not vae.decoder.proj3d.weight.is_contiguous()
    F.uninstall(vae)
    assert vae.decoder.proj3d.weight.is_contiguous()
    assert "_unsloth_channels_last_modules" not in vae.__dict__


def test_overlap_factor_searches_the_interval_above_the_floor():
    """With overlap 0.2 the 8-latent floor never tiles exactly; 10 latents does, so the ladder goes 32 -> 15 -> 10."""
    attrs = {"tile_sample_min_size": 256, "tile_latent_min_size": 32}
    assert O.halved(attrs, 8, 1, 0.2) == {"tile_sample_min_size": 120, "tile_latent_min_size": 15}
    assert O.halved(attrs, 8, 2, 0.2) == {"tile_sample_min_size": 80, "tile_latent_min_size": 10}
    assert O.halved(attrs, 8, 3, 0.2) is None


class _TallTileProbeOOM(_UntiledProbeOOMWanLike):
    """16 px-tall (2-latent) inputs: the 8-latent height tiles never seam, only the width tiles do."""

    def _decode(self, z, return_dict = True):
        if z.shape[-1] != 64 and not self.use_tiling:
            self.untiled_probes += 1
            raise _FakeOOM()
        if z.shape[-1] == 64:
            self.seen.append((self.use_tiling, self.tile_sample_min_height, self.tile_sample_min_width))
            if not self.use_tiling or self.tile_sample_min_width > self.fits:
                raise _FakeOOM()
        return (z.repeat_interleave(8, -1).repeat_interleave(8, -2) + 1,)


def test_unprobed_floor_ignores_an_axis_the_input_fits_in():
    vae = _TallTileProbeOOM(fits = 128)
    vae.tile_sample_min_height = 64  # 8 latents: at the floor already
    vae.tile_sample_stride_height = 48
    vae.use_tiling = True
    O.install(vae)
    out = vae._decode(torch.zeros(1, 4, 1, 8, 64))
    assert out[0].shape[-2:] == (64, 512)
    stats = vae._unsloth_vae_oom_stats
    assert stats["recovered"] == 1 and stats["quality_refused"] == 0
    assert vae.seen[-1][2] == 128  # width tiles halved to 16 latents


class _BatchProbeWanLike(_WanLike):
    def __init__(self, fits):
        super().__init__(fits)
        self.probe_batches = []

    def _decode(self, z, return_dict = True):
        if z.shape[-1] != 64:
            self.probe_batches.append(int(z.shape[0]))
        return super()._decode(z, return_dict)


def test_probe_crop_takes_one_batch_element():
    vae = _BatchProbeWanLike(fits = 64)
    vae.use_tiling = True
    O.install(vae)
    out = vae._decode(torch.zeros(4, 4, 1, 64, 64))
    assert out[0].shape[0] == 4 and out[0].shape[-2:] == (512, 512)
    assert vae.probe_batches and set(vae.probe_batches) == {1}


# -- PR review round 5 regressions ------------------------------------------------------------------------------


class _TembWanLike(_WanLike):
    """LTX-2-shaped ``_decode(z, temb, ...)``: ``temb`` must match ``z``'s batch."""

    def _decode(self, z, temb = None, return_dict = True):
        if temb is not None and temb.shape[0] != z.shape[0]:
            raise ValueError(f"temb batch {temb.shape[0]} != latent batch {z.shape[0]}")
        return super()._decode(z, return_dict)


def test_probe_slices_batch_sized_args_with_the_latent():
    """Round 5 item 1: the probe crop is one sample, so a batch-sized positional ``temb`` must follow it."""
    vae = _TembWanLike(fits = 128)
    vae.use_tiling = True
    O.install(vae)
    z = torch.zeros(2, 4, 1, 64, 64)
    out = vae._decode(z, torch.zeros(2, 3), return_dict = False)
    assert out[0].shape[0] == 2 and out[0].shape[-2:] == (512, 512)
    assert vae._unsloth_vae_oom_stats["probe_psnr"][1] == 99.0


class _ActiveLog(_WanLike):
    def __init__(self, fits):
        super().__init__(fits)
        self.active = []

    def _decode(self, z, return_dict = True):
        self.active.append((self.use_tiling, O.oom_halving_active(self)))
        return super()._decode(z, return_dict)


def test_stock_tile_retry_marks_the_retry_active():
    """Round 5 item 2: the level-0 (stock tile) retry must read as an active retry, so the wide-tile wrappers hand
    over to stock tiling instead of rerunning the wide planner that failed."""
    vae = _ActiveLog(fits = 256)
    O.install(vae)
    vae._decode(torch.zeros(1, 4, 1, 64, 64))
    assert vae.active == [(False, False), (True, True)]
    assert O.oom_halving_active(vae) is False


def test_wide_tile_wrappers_read_the_active_marker():
    import inspect

    from core.inference import diffusion_vae_tiling as vt
    from core.inference import video_ltx2_vae_tiles as lt

    for mod in (vt, lt):
        src = inspect.getsource(mod.install)
        assert "oom_halving_active(self)" in src and "_unsloth_vae_oom_halvings" not in src


class _CleanDec(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm_out = torch.nn.GroupNorm(4, c)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        return self.act(self.norm_out(x))


class AutoencoderKLCleanNew(torch.nn.Module):
    def __init__(self, c = 8):
        super().__init__()
        self.decoder = _CleanDec(c)

    def _decode(self, z, return_dict = True):
        return (self.decoder(z),)


def test_structural_rejected_when_every_norm_falls_back(monkeypatch):
    """Round 5 item 3: a correct pairing whose norms all take group_norm_act's stock fallback fused nothing, so the
    calibration must reject it (before: counted as fused and accepted)."""
    monkeypatch.setattr(F, "runtime_ok", lambda: True)
    vae = AutoencoderKLCleanNew()
    F.install_structural(vae)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(F, "_gn_fusable", lambda x, norm: False)
    with torch.no_grad():
        vae._decode(torch.randn(1, 8, 6, 6))
    state = vae._unsloth_vae_structural
    assert state.status == "rejected" and state.kernel_calls == 0
    assert not any("forward" in m.__dict__ for m in vae.decoder.modules())
