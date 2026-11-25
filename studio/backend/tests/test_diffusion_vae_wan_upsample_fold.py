# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wan-lineage VAE upsample fold (diffusion_vae_fused._fast_wan_resample): a single-frame (or first-chunk) Resample runs
nearest 2x + conv3x3 as one ConvTranspose2d(k4, s2, p1) on the cards the FLUX / KL fold is measured on. CPU, with the
switch forced on: equals the stock module to fold rounding, never touches the 2x Triton path, follows the switch, and
leaves the video interleave (time_conv) path alone."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

torch = pytest.importorskip("torch")
from core.inference import diffusion_vae_fused as vf  # noqa: E402
from core.inference import diffusion_vae_strips as st  # noqa: E402


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(st.UPSAMPLE_FOLD_ENV, raising = False)
    st._FOLD_CACHE.clear()  # noqa: SLF001
    vf.ENGAGED.pop("upsample_fold", None)
    yield
    st._FOLD_CACHE.clear()  # noqa: SLF001


def _resample(
    kind: str,
    dim: int = 8,
    dtype = torch.float64,
):
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    if kind in ("qi21", "qi21_3d"):
        mod = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_qwenimage21")
        r = mod.QwenImage21Resample(dim, "upsample3d" if kind == "qi21_3d" else "upsample2d")
    elif kind == "qwen":
        mod = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_qwenimage")
        r = mod.QwenImageResample(dim, "upsample3d")
    else:
        mod = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_wan")
        r = mod.WanResample(dim, "upsample2d")
    del diffusers
    return r.to(dtype).eval()


def _no_triton(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the 2x Triton path ran")

    monkeypatch.setattr(vf, "up_nearest2x", boom)


@pytest.mark.parametrize("kind", ["qi21", "qi21_3d", "qwen", "wan"])
@pytest.mark.parametrize("frames", [1, 3])
def test_first_chunk_upsample_runs_the_fold(kind, frames, monkeypatch):
    r = _resample(kind)
    assert vf._wan_resample_ok(r)  # noqa: SLF001  every Wan-lineage upsampler, QI21's upsample3d included
    monkeypatch.setenv(st.UPSAMPLE_FOLD_ENV, "1")
    _no_triton(monkeypatch)
    x = torch.randn(2, 8, frames, 5, 7, dtype = torch.float64)
    with torch.no_grad():
        want = r(x, feat_cache = [None], feat_idx = [0]) if kind != "wan" else r(x)
        cache = [None]
        got = (
            vf._fast_wan_resample(r)(x, feat_cache = cache, feat_idx = [0])
            if kind != "wan"
            # noqa: SLF001
            else (
                vf._fast_wan_resample(r)(x)  # noqa: SLF001
            )
        )
    assert got.shape == want.shape
    assert (got - want).abs().max().item() < 1e-6  # fp32-summed taps on fp64 data
    assert vf.ENGAGED.get("upsample_fold") == 1


def test_switch_off_keeps_the_2x_path(monkeypatch):
    r = _resample("qi21")
    monkeypatch.setenv(st.UPSAMPLE_FOLD_ENV, "0")
    calls = []

    def fake_up(
        y,
        interleave,
        bias = None,
    ):
        calls.append(interleave)
        b, c, t, h, w = y.shape
        up = torch.nn.functional.interpolate(
            y.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), scale_factor = 2.0
        )
        return up, t

    monkeypatch.setattr(vf, "up_nearest2x", fake_up)
    x = torch.randn(1, 8, 1, 4, 4, dtype = torch.float64)
    with torch.no_grad():
        want = r(x)
        got = vf._fast_wan_resample(r)(x)  # noqa: SLF001
    assert calls == [False] and torch.allclose(got, want)
    assert "upsample_fold" not in vf.ENGAGED


def test_unmeasured_device_keeps_the_2x_path():
    """Unset switch: CPU is not a measured device, so the fold declines (returns None)."""
    r = _resample("qi21")
    x = torch.randn(1, 8, 1, 4, 4, dtype = torch.float64)
    assert vf._wan_folded_upsample(r.resample[1], x) is None  # noqa: SLF001


def test_batched_frames_over_the_whole_map_limit_decline(monkeypatch):
    """The strip fold writes one frame: several frames above the whole-map limit stay on the 2x path."""
    r = _resample("qi21")
    monkeypatch.setenv(st.UPSAMPLE_FOLD_ENV, "1")
    monkeypatch.setattr(st, "FOLD_FULL_MAX_ELEMS", 16)
    conv = r.resample[1]
    assert vf._wan_folded_upsample(conv, torch.randn(2, 8, 1, 4, 4, dtype = torch.float64)) is None  # noqa: SLF001
    one = torch.randn(1, 8, 1, 4, 4, dtype = torch.float64)
    got = vf._wan_folded_upsample(conv, one)  # noqa: SLF001  one frame: folded strip by strip
    with torch.no_grad():
        ref = conv(torch.nn.functional.interpolate(one[:, :, 0], scale_factor = 2.0))
    assert (got - ref).abs().max().item() < 1e-6


def test_video_interleave_step_is_untouched(monkeypatch):
    """A later chunk of an upsample3d block interleaves time_conv frames: never folded."""
    r = _resample("qwen")
    if not vf._wan_resample_ok(r):  # noqa: SLF001
        pytest.skip("qwen resample is not on the fused path")
    monkeypatch.setenv(st.UPSAMPLE_FOLD_ENV, "1")
    seen = []

    def fake_up(
        y,
        interleave,
        bias = None,
    ):
        seen.append(interleave)
        raise RuntimeError("stop")

    monkeypatch.setattr(vf, "up_nearest2x", fake_up)
    monkeypatch.setattr(
        vf, "causal_conv", lambda conv, x, prev, with_bias = True: (x.repeat(1, 2, 1, 1, 1), x)
    )
    x = torch.randn(1, 8, 2, 4, 4, dtype = torch.float64)
    with pytest.raises(RuntimeError, match = "stop"):
        vf._fast_wan_resample(r)(
            x, feat_cache = [torch.zeros(1, 8, 1, 4, 4, dtype = torch.float64)], feat_idx = [0]
        )  # noqa: SLF001
    assert seen == [True] and "upsample_fold" not in vf.ENGAGED


def test_qi21_upsample3d_later_chunk_stays_stock(monkeypatch):
    """Qwen-Image-2.1's one-frame time_conv refuses a feature cache: a later chunk runs (and raises) exactly as stock."""
    r = _resample("qi21_3d")
    monkeypatch.setenv(st.UPSAMPLE_FOLD_ENV, "1")
    _no_triton(monkeypatch)
    x = torch.randn(1, 8, 1, 4, 4, dtype = torch.float64)
    fast = vf._fast_wan_resample(r)  # noqa: SLF001
    cache = [None]
    with torch.no_grad():
        fast(x, feat_cache = cache, feat_idx = [0])
        assert cache == ["Rep"]
        with pytest.raises(ValueError, match = "feature cache"):
            r(x, feat_cache = [torch.zeros(1, 8, 1, 4, 4, dtype = torch.float64)], feat_idx = [0])
        with pytest.raises(ValueError, match = "feature cache"):
            fast(x, feat_cache = [torch.zeros(1, 8, 1, 4, 4, dtype = torch.float64)], feat_idx = [0])
