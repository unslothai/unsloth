# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests for the Wan-lineage VAE upsample fold in diffusion_vae_fused._fast_wan_resample."""

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
    assert vf._wan_resample_ok(r)  # noqa: SLF001
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
    with torch.no_grad():
        assert vf._wan_folded_upsample(r.resample[1], x) is None  # noqa: SLF001


def test_batched_frames_over_the_whole_map_limit_decline(monkeypatch):
    """The strip fold writes one frame: several frames above the whole-map limit stay on the 2x path."""
    r = _resample("qi21")
    monkeypatch.setenv(st.UPSAMPLE_FOLD_ENV, "1")
    monkeypatch.setattr(st, "FOLD_FULL_MAX_ELEMS", 16)
    conv = r.resample[1]
    one = torch.randn(1, 8, 1, 4, 4, dtype = torch.float64)
    with torch.no_grad():
        assert (
            vf._wan_folded_upsample(conv, torch.randn(2, 8, 1, 4, 4, dtype = torch.float64)) is None
        )  # noqa: SLF001
        got = vf._wan_folded_upsample(conv, one)  # noqa: SLF001
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


def _torch_up(monkeypatch):
    def up(
        y,
        interleave,
        bias = None,
    ):
        b, c, t, h, w = y.shape
        return torch.nn.functional.interpolate(
            y.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), scale_factor = 2.0
        ), t

    monkeypatch.setattr(vf, "up_nearest2x", up)


def test_compiled_resample_does_not_guard_on_the_fold_cache(monkeypatch):
    """Under torch.compile the fold declines before its weak-key cache is read: one graph, not one per call."""
    from torch._dynamo.utils import counters

    r = _resample("qi21", dtype = torch.float32)
    monkeypatch.setenv(st.UPSAMPLE_FOLD_ENV, "1")
    _torch_up(monkeypatch)
    for warm in (False, True):
        st._FOLD_CACHE.clear()  # noqa: SLF001
        torch._dynamo.reset()
        counters.clear()
        fast = vf._fast_wan_resample(r)  # noqa: SLF001
        with torch.no_grad():
            if warm:
                fast(torch.randn(1, 8, 1, 5, 7))
            fn = torch.compile(fast, backend = "eager", fullgraph = True, dynamic = True)
            for height in (5, 5, 7):
                fn(torch.randn(1, 8, 1, height, 7))
        assert counters["stats"]["unique_graphs"] == 1, warm
    torch._dynamo.reset()


def test_training_keeps_the_conv_weight_gradient(monkeypatch):
    r = _resample("qi21", dtype = torch.float32)
    monkeypatch.setenv(st.UPSAMPLE_FOLD_ENV, "1")
    _torch_up(monkeypatch)
    x = torch.randn(1, 8, 1, 5, 7, requires_grad = True)
    r(x).square().mean().backward()
    want = r.resample[1].weight.grad.clone()
    r.zero_grad(set_to_none = True)
    vf._fast_wan_resample(r)(x).square().mean().backward()  # noqa: SLF001
    got = r.resample[1].weight.grad
    assert got is not None and torch.allclose(got, want, atol = 1e-5)
    assert "upsample_fold" not in vf.ENGAGED


def test_fp16_fold_overflow_keeps_the_2x_path(monkeypatch):
    """Four fp16 taps of 2e4 sum to 8e4 (inf in fp16) while the 2x conv stays finite: that conv is not folded."""
    r = _resample("qi21", dtype = torch.float16)
    monkeypatch.setenv(st.UPSAMPLE_FOLD_ENV, "1")
    _torch_up(monkeypatch)
    with torch.no_grad():
        r.resample[1].weight.fill_(20000)
        r.resample[1].bias.zero_()
        x = torch.full((1, 8, 1, 3, 3), 0.002, dtype = torch.float16)
        want = r(x)
        got = vf._fast_wan_resample(r)(x)  # noqa: SLF001
    assert bool(want.isfinite().all()) and bool(got.isfinite().all())
    assert torch.equal(got, want)
    assert "upsample_fold" not in vf.ENGAGED
    assert not st.fold_finite(r.resample[1])
