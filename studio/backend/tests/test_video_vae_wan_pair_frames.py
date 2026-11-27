# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wan's untiled decode with two latent frames per decoder call: same frames as the one-frame loop, taken only when
the second frame's activations fit, off under the kill switch, and back to one frame on OOM."""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers import AutoencoderKLWan

from core.inference import decode_layers
from core.inference import video_vae_untiled as U
from core.inference.diffusion_vae_fused import wan_pair_decode_ok, wan_paired_decode


def _tiny_wan(kind: str) -> AutoencoderKLWan:
    torch.manual_seed(0)
    if kind == "wan2.1":  # Wan 2.1 / 2.2 A14B layout
        vae = AutoencoderKLWan(
            base_dim = 8, z_dim = 4, dim_mult = [1, 2, 2, 2], num_res_blocks = 1,
            latents_mean = [0.0] * 4, latents_std = [1.0] * 4,
        )  # fmt: skip
    else:  # Wan 2.2 TI2V 5B layout: residual up blocks, patchified output
        vae = AutoencoderKLWan(
            base_dim = 8, decoder_base_dim = 16, z_dim = 4, dim_mult = [1, 2, 2, 2], num_res_blocks = 1,
            is_residual = True, in_channels = 12, out_channels = 12, patch_size = 2, scale_factor_temporal = 4,
            scale_factor_spatial = 16, latents_mean = [0.0] * 4, latents_std = [1.0] * 4,
        )  # fmt: skip
    with torch.no_grad():
        for p in vae.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return vae.eval()


@pytest.mark.parametrize("kind", ["wan2.1", "wan2.2-5b"])
@pytest.mark.parametrize("latent_frames", [1, 2, 3, 4, 6])
def test_paired_matches_one_frame_loop(kind, latent_frames):
    vae = _tiny_wan(kind)
    assert wan_pair_decode_ok(vae)
    torch.manual_seed(1)
    z = torch.randn(1, 4, latent_frames, 4, 6)
    with torch.no_grad():
        ref = vae._decode(z).sample
        for step in (2, 3):
            got = wan_paired_decode(vae, z, frames_per_call = step).sample
            assert got.shape == ref.shape and got.dtype == ref.dtype
            assert (got - ref).abs().max().item() <= 1e-5
        assert torch.equal(wan_paired_decode(vae, z).sample, wan_paired_decode(vae, z).sample)
    assert all(c is None for c in vae._feat_map)


def _gated(
    monkeypatch,
    free_gib: float,
    kind: str = "wan2.1",
):
    if U.PAIR_FRAMES_ENV not in os.environ:
        monkeypatch.setenv(U.PAIR_FRAMES_ENV, "force")  # CPU: past the NVIDIA sm80+ default
    vae = _tiny_wan(kind)
    vae.enable_tiling()

    class Pipe:
        pass

    pipe = Pipe()
    pipe.vae = vae
    family = "wan2.1-t2v-1.3b" if kind == "wan2.1" else "wan2.2-ti2v-5b"
    monkeypatch.setattr(U, "_free_bytes", lambda device: int(free_gib * 2**30))
    assert U.install_untiled_decode(pipe, family)
    return vae, family


def _frames(vae, z):
    with torch.no_grad():
        return vae.decode(z).sample


def test_gate_takes_pairs_when_they_fit(monkeypatch):
    vae, _ = _gated(monkeypatch, 1000)
    z = torch.randn(1, 4, 5, 4, 6)
    out = _frames(vae, z)
    assert vae._unsloth_untiled_stats["paired"] == 1
    assert vae._unsloth_untiled_stats["untiled"] == 1
    assert vae.use_tiling and not vae.__dict__.get(U._PAIR_ATTR)
    vae.use_tiling = False
    with torch.no_grad():
        ref = vae._decode(z).sample
    assert (out - ref).abs().max().item() <= 1e-5


def test_gate_keeps_one_frame_when_only_untiled_fits(monkeypatch):
    vae, family = _gated(monkeypatch, 1000)
    big = (1, 4, 5, 60, 104)
    need = U.untiled_decode_bytes(family, big, 4)
    extra = U.pair_frames_extra_bytes(family, big, 4)
    assert extra and extra > 0
    free = (
        need * U._MARGIN + U._MARGIN_BYTES + extra * 0.5
    )  # untiled fits, the second frame does not
    monkeypatch.setattr(U, "_free_bytes", lambda device: int(free))
    monkeypatch.setattr(U, "untiled_decode_bytes", lambda *a, **k: need)
    monkeypatch.setattr(U, "pair_frames_extra_bytes", lambda *a, **k: extra)
    seen = []
    monkeypatch.setattr(
        "core.inference.diffusion_vae_fused.wan_paired_decode", lambda *a, **k: seen.append(1)
    )
    _frames(vae, torch.randn(1, 4, 5, 4, 6))
    assert vae._unsloth_untiled_stats == {
        "untiled": 1, "tiled": 0, "wide": 0, "oom_fallback": 0, "paired": 0,
    }  # fmt: skip
    assert not seen


def test_kill_switch(monkeypatch):
    monkeypatch.setenv(U.PAIR_FRAMES_ENV, "0")
    vae, _ = _gated(monkeypatch, 1000)
    _frames(vae, torch.randn(1, 4, 5, 4, 6))
    assert vae._unsloth_untiled_stats["paired"] == 0
    assert vae._unsloth_untiled_stats["untiled"] == 1


def test_short_latents_stay_one_frame(monkeypatch):
    vae, _ = _gated(monkeypatch, 1000)
    _frames(vae, torch.randn(1, 4, 2, 4, 6))
    assert vae._unsloth_untiled_stats["paired"] == 0


def test_pair_oom_falls_back_to_one_frame_untiled(monkeypatch):
    vae, _ = _gated(monkeypatch, 1000, kind = "wan2.2-5b")
    import core.inference.diffusion_vae_fused as F

    def boom(*a, **k):
        raise torch.cuda.OutOfMemoryError("fake")

    monkeypatch.setattr(F, "wan_paired_decode", boom)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    z = torch.randn(1, 4, 5, 4, 6)
    out = _frames(vae, z)
    stats = vae._unsloth_untiled_stats
    assert stats["paired"] == 0 and stats["untiled"] == 1 and stats["tiled"] == 0
    vae.use_tiling = False
    with torch.no_grad():
        ref = vae._decode(z).sample
    assert torch.equal(out, ref)
    # never retried at this size
    vae.use_tiling = True
    _frames(vae, z)
    assert vae._unsloth_untiled_stats["untiled"] == 2 and vae._unsloth_untiled_stats["paired"] == 0


def test_layer_is_installed_once_and_inert_when_tiled(monkeypatch):
    vae, family = _gated(monkeypatch, 1000)
    pipe = type("P", (), {})()
    pipe.vae = vae
    assert U.install_untiled_decode(pipe, family)
    layers = [l for l in vae.__dict__.get(decode_layers.LAYERS_ATTR, []) if l.name == U._PAIR_LAYER]
    assert len(layers) == 1
    vae.__dict__[U._PAIR_ATTR] = True
    vae.use_tiling = True
    called = []
    monkeypatch.setattr(
        "core.inference.diffusion_vae_fused.wan_paired_decode", lambda *a, **k: called.append(1)
    )
    with torch.no_grad():
        vae._decode(torch.randn(1, 4, 3, 4, 6))
    assert not called


def test_default_is_nvidia_sm80_plus_only(monkeypatch):
    monkeypatch.delenv(U.PAIR_FRAMES_ENV, raising = False)
    assert not U._pair_frames_enabled(torch.device("cpu"))
    cap = [(7, 5)]
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: cap[0])
    monkeypatch.setattr(torch.version, "hip", None)
    cuda = torch.device("cuda", 0)
    assert not U._pair_frames_enabled(cuda)  # T4: measured slower
    for cap[0] in ((8, 0), (8, 9), (10, 0)):
        assert U._pair_frames_enabled(cuda)
    monkeypatch.setattr(torch.version, "hip", "6.4")
    assert not U._pair_frames_enabled(cuda)
    monkeypatch.setenv(U.PAIR_FRAMES_ENV, "force")
    assert U._pair_frames_enabled(torch.device("cpu"))
    monkeypatch.setenv(U.PAIR_FRAMES_ENV, "0")
    cap[0] = (9, 0)
    monkeypatch.setattr(torch.version, "hip", None)
    assert not U._pair_frames_enabled(cuda)


def test_cpu_default_keeps_one_frame(monkeypatch):
    monkeypatch.setenv(U.PAIR_FRAMES_ENV, "")
    vae, _ = _gated(monkeypatch, 1000)
    _frames(vae, torch.randn(1, 4, 5, 4, 6))
    assert vae._unsloth_untiled_stats["paired"] == 0
