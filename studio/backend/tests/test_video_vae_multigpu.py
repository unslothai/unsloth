# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wan VAE tiled decode with tiles shared by two cards: bitwise the stock tiled decode (CPU here, both "cards" the
CPU; test_video_vae_multigpu_cuda-style coverage runs on Kaggle 2x T4), gated off for anything it cannot mirror."""

from __future__ import annotations

import types

import pytest
import torch

from core.inference import video_vae_multigpu as vm

wan = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")


def _vae(patch_size = None):
    kw = dict(
        base_dim = 8,
        z_dim = 4,
        dim_mult = [1, 1, 1, 1],
        num_res_blocks = 1,
        temperal_downsample = [False, True, True],
        latents_mean = [0.0] * 4,
        latents_std = [1.0] * 4,
    )
    if patch_size is not None:
        kw.update(
            patch_size = patch_size,
            is_residual = True,
            in_channels = 12,
            out_channels = 12,
            z_dim = 4,
            scale_factor_spatial = 16,
        )
    torch.manual_seed(0)
    vae = wan.AutoencoderKLWan(**kw).eval()
    with torch.no_grad():
        for p in vae.parameters():
            p.normal_(0, 0.2)
    vae.enable_tiling(
        tile_sample_min_height = 32,
        tile_sample_min_width = 32,
        tile_sample_stride_height = 24,
        tile_sample_stride_width = 24,
    )
    return vae


def _latents(
    vae,
    frames = 3,
    h = 9,
    w = 11,
):
    g = torch.Generator().manual_seed(1)
    return torch.randn(1, vae.config.z_dim, frames, h, w, generator = g)


@pytest.mark.parametrize("patch_size", [None, 2])
def test_split_tiles_match_the_stock_tiled_decode_bitwise(patch_size):
    try:
        vae = _vae(patch_size)
    except TypeError:
        pytest.skip("this diffusers has no patch_size Wan VAE")
    z = _latents(vae)
    with torch.no_grad():
        ref = vae.tiled_decode(z, return_dict = False)[0]
        from core.inference.diffusion_multigpu import build_replica

        replica = build_replica(
            vae, torch.device("cpu")
        )  # the copy the split builds, created on its device
        out = vm.split_tiled_decode(vae, z, torch.device("cpu"), replica = replica)
        built = vm.split_tiled_decode(
            vae, z, torch.device("cpu")
        )  # the copy built by the split itself
    assert out.shape == ref.shape
    assert torch.equal(out, ref) and torch.equal(built, ref)


def _copy(vae):
    import copy
    return copy.deepcopy(vae)


def test_the_mirrored_body_is_the_installed_diffusers_one():
    # A diffusers that changes tiled_decode turns the split off (fingerprint mismatch) instead of drifting from it.
    assert vm.eligible(_vae()) is None
    assert vm.body_fingerprint(wan.AutoencoderKLWan.tiled_decode) in vm._KNOWN_BODIES


def test_a_changed_body_or_another_vae_is_refused():
    vae = _vae()

    class Other(torch.nn.Module):
        pass

    assert "not the Wan VAE" in vm.eligible(Other())

    def tiled_decode(
        self,
        z,
        return_dict = True,
    ):
        return None

    vae.__class__ = type(
        "AutoencoderKLWan", (wan.AutoencoderKLWan,), {"tiled_decode": tiled_decode}
    )
    assert "differs" in vm.eligible(vae)


def test_install_routes_tiled_decode_and_falls_back_off_cuda(monkeypatch):
    vae = _vae()
    pipe = types.SimpleNamespace(vae = vae)
    z = _latents(vae)
    with torch.no_grad():
        ref = vae.tiled_decode(z, return_dict = False)[0]
    assert vm.install(pipe, 1)
    with torch.no_grad():
        out = vae.decode(z, return_dict = False)[0]  # CPU latents: one card, stock tiles
    assert torch.equal(out, ref)
    assert vm.stats(pipe)["one_card"] == 1 and vm.stats(pipe)["split"] == 0
    assert "CUDA" in vm.stats(pipe)["last"]
    monkeypatch.setenv(vm.ENV, "0")
    assert not vm.install(types.SimpleNamespace(vae = _vae()), 1)
    vm.uninstall(pipe)
    assert "tiled_decode" not in vae.__dict__ and vm.stats(pipe) is None


def test_hooks_stay_with_the_source_while_the_copy_is_built():
    vae = _vae()
    calls = []
    handle = vae.decoder.register_forward_hook(lambda m, a, o: calls.append(1))
    with vm._without_hooks(vae):
        assert not vae.decoder._forward_hooks
        replica = _copy(vae)
    assert vae.decoder._forward_hooks and not replica.decoder._forward_hooks
    handle.remove()


def test_partner_free_memory_counts_reusable_cache_and_the_need_scales_with_the_tile(monkeypatch):
    fake = types.SimpleNamespace(
        cuda = types.SimpleNamespace(mem_get_info = lambda o: (1 << 30, 16 << 30)), device = torch.device
    )
    import core.inference.diffusion_memory as dm

    monkeypatch.setattr(dm, "cuda_reusable_cache_bytes", lambda device = None: 3 << 30)
    assert vm._partner_free_mib(fake, 1) == 4096
    vae = _vae()
    vae.tile_sample_min_height = vae.tile_sample_min_width = 256
    base = vm.need_mib(vae)
    vae.tile_sample_min_height = vae.tile_sample_min_width = (
        512  # wide tiles: four times the working set
    )
    assert vm.need_mib(vae) - base == 3 * vm._WORKING_MIB


def test_a_copy_holding_tensors_on_another_card_is_detected():
    m = torch.nn.Linear(2, 2)
    m.cache = torch.zeros(1)
    assert vm._stray_tensors(m, torch.device("cpu")) == []
    m.cache = types.SimpleNamespace()  # not a tensor
    assert vm._stray_tensors(m, torch.device("cuda", 1)) == []


@pytest.mark.parametrize("failing", ["load", "partner"])
def test_a_failing_card_still_releases_the_copy(monkeypatch, failing):
    # Either card raising (an OOM, say) must still free the copy the partner built, or the frames kept alive by the
    # traceback hold it while the stock decode retries.
    import gc
    import weakref

    from core.inference import diffusion_multigpu as dm

    vae = _vae()
    z = _latents(vae)
    built = []

    class Boom(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("card out of memory")

    def fake_build(src, device):
        copy = _vae()
        if failing == "partner":
            copy.post_quant_conv = Boom()
        built.append(weakref.ref(copy))
        return copy

    monkeypatch.setattr(dm, "build_replica", fake_build)
    if failing == "load":
        vae.post_quant_conv = Boom()
    with torch.no_grad(), pytest.raises(RuntimeError, match = "card out of memory") as info:
        vm.split_tiled_decode(vae, z, torch.device("cpu"))
    gc.collect()
    assert info.tb is not None  # the failed decode's frames are still alive here
    assert built and built[0]() is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason = "needs two CUDA cards"
)
def test_split_tiles_keep_the_callers_autocast_on_two_cards():
    # Autocast is thread-local: the partner thread decodes at the caller's precision, so the result is still the
    # stock tiled decode under the same autocast.
    vae = _vae().to("cuda:0")
    z = _latents(vae).to("cuda:0")
    with torch.no_grad(), torch.autocast("cuda", dtype = torch.float16):
        ref = vae.tiled_decode(z, return_dict = False)[0]
        out = vm.split_tiled_decode(vae, z, torch.device("cuda:1"))
    assert out.dtype == ref.dtype
    assert torch.equal(out, ref)


def test_partner_tiles_never_run_the_sources_patched_forwards():
    # Speed paths give blocks instance-level forwards that close over the block (and wrap its stock bound method);
    # a deepcopy keeps those closures, so the partner would run the load card's modules. They must be rebuilt
    # against the copy, and the output must still be the stock tiled decode.
    import threading

    vae = _vae()
    z = _latents(vae)
    with torch.no_grad():
        ref = vae.tiled_decode(z, return_dict = False)[0]
    ran = []

    def patch(block):
        stock = type(block).forward.__get__(block)

        def fast(
            *args,
            _block = block,
            **kwargs,
        ):
            ran.append((threading.current_thread().name, id(_block)))
            return stock(*args, **kwargs)

        def forward(*args, **kwargs):
            return fast(*args, **kwargs)

        block.forward = forward

    blocks = [m for m in vae.decoder.modules() if type(m).__name__.endswith("ResidualBlock")]
    assert blocks
    for block in blocks:
        patch(block)
    source = {id(b) for b in blocks}
    with torch.no_grad():
        out = vm.split_tiled_decode(vae, z, torch.device("cpu"))
    partner = [i for name, i in ran if name == "unsloth-vae-split"]
    assert partner and not (set(partner) & source)
    assert torch.equal(out, ref)


def test_a_copy_whose_forward_reaches_the_source_is_detected():
    from core.inference.diffusion_multigpu import build_replica

    vae = _vae()
    block = next(m for m in vae.decoder.modules() if type(m).__name__.endswith("ResidualBlock"))
    block.forward = lambda x, *a, _b = block, **k: type(_b).forward(_b, x, *a, **k)
    copy = build_replica(vae, torch.device("cpu"))
    assert any("captured" in s for s in vm._stray_tensors(copy, torch.device("cpu")))
    assert vm._rebind_forwards(vae, copy) >= 1
    assert vm._stray_tensors(copy, torch.device("cpu")) == []


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason = "needs two CUDA cards"
)
def test_split_tiles_with_the_fused_wan_forwards_on_two_cards():
    # The default speed tier installs fused Wan forwards (closures over the load card's blocks) before the split:
    # the partner must run its own copies, and the result is the fused one-card tiled decode.
    from core.inference import diffusion_vae_fused as fused

    vae = _vae().to("cuda:0")
    if not fused.install_wan_vae(vae, encoder = False):
        pytest.skip("fused Wan VAE path unavailable here")
    z = _latents(vae).to("cuda:0")

    def engaged():
        return sum(fused.ENGAGED.values())

    with torch.no_grad():
        before = engaged()
        ref = vae.tiled_decode(z, return_dict = False)[0]
        one_card = engaged() - before
        before = engaged()
        out = vm.split_tiled_decode(vae, z, torch.device("cuda:1"))
        split = engaged() - before
    # Every tile ran the fused path on both cards (a partner falling back to stock forwards would count fewer).
    assert one_card > 0 and split == one_card
    # One card repeats bitwise; the partner's cuDNN / Triton picks on its own card can differ by an ulp
    # (Kaggle 2x T4: 4.5e-7 at scale 0.85 in fp32).
    torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 2e-6)
