# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Live latent previews: layout handling, the x0 estimate, the fitted maps, and (on CUDA) that a preview
never synchronises the host, never touches the latent, and publishes a JPEG."""

from __future__ import annotations

import base64
import io
import threading
import time

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_preview as DP
from core.inference.diffusion_preview_factors import FACTORS, FAMILY_FACTORS


def _spec(
    layout,
    channels,
    patch = 1,
    down = 8,
):
    out = 3 * patch * patch
    weight = tuple(tuple(0.01 * ((c + o) % 7) for o in range(out)) for c in range(channels))
    return DP.LatentRGB(layout = layout, down = down, patch = patch, weight = weight, bias = (0.5,) * out)


def test_kill_switch_and_request_toggle(monkeypatch):
    monkeypatch.delenv(DP.PREVIEW_ENV, raising = False)
    assert DP.preview_wanted(None) is True
    assert DP.preview_wanted(False) is False
    monkeypatch.setenv(DP.PREVIEW_ENV, "0")
    assert DP.preview_wanted(True) is False


def test_tokens_grid_is_the_first_image_or_frame():
    spec = _spec("tokens", 64, patch = 2, down = 16)
    lat = torch.arange(2 * 4 * 6 * 64, dtype = torch.float32).reshape(2, 24, 64)
    g = DP.latent_grid(lat, spec, height = 64, width = 96)
    assert tuple(g.shape) == (4, 6, 64)
    assert torch.equal(g.reshape(24, 64), lat[0])
    video = torch.randn(1, 48, 64)
    assert torch.equal(DP.latent_grid(video, spec, 64, 96).reshape(24, 64), video[0, :24])
    assert DP.latent_grid(torch.randn(1, 25, 64), spec, 64, 96) is None
    assert DP.latent_grid(torch.randn(1, 24, 63), spec, 64, 96) is None


def test_unpacked_layouts_take_their_own_shape():
    spec = _spec("bchw", 16)
    lat = torch.randn(2, 16, 5, 7)
    assert torch.equal(DP.latent_grid(lat, spec, 0, 0), lat[0].permute(1, 2, 0))
    spec5 = _spec("bcthw", 48, down = 16)
    vid = torch.randn(1, 48, 3, 5, 7)
    assert torch.equal(DP.latent_grid(vid, spec5, 0, 0), vid[0, :, 0].permute(1, 2, 0))
    assert DP.latent_grid(torch.randn(1, 16, 5, 7), spec5, 0, 0) is None


def test_project_places_each_patch_pixel():
    # identity map: the pixel shuffle must land (dy, dx, rgb) where it belongs
    patch, out = 2, 12
    weight = torch.eye(out)
    grid = torch.arange(3 * 2 * out, dtype = torch.float32).reshape(3, 2, out)
    rgb = DP.project(grid, weight, torch.zeros(out), patch)
    assert tuple(rgb.shape) == (6, 4, 3)
    for gy in range(3):
        for gx in range(2):
            cell = grid[gy, gx].reshape(patch, patch, 3)
            assert torch.equal(rgb[gy * 2 : gy * 2 + 2, gx * 2 : gx * 2 + 2], cell)


def test_x0_estimate_recovers_the_prediction_of_an_euler_step():
    x0, noise = torch.rand(4, 4, 3), torch.randn(4, 4, 3)
    s_prev, s_cur = 0.8, 0.6
    prev = (1 - s_prev) * x0 + s_prev * noise
    v = noise - x0
    cur = prev + (s_cur - s_prev) * v
    assert torch.allclose(DP.x0_estimate(prev, cur, (s_prev, s_cur)), x0, atol = 1e-5)
    # affine maps commute with it because the weights sum to 1
    pair = (torch.tensor(s_prev), torch.tensor(s_cur))
    assert torch.allclose(DP.x0_estimate(prev * 2 + 1, cur * 2 + 1, pair), x0 * 2 + 1, atol = 1e-5)
    assert DP.x0_estimate(None, cur, (s_prev, s_cur)) is cur
    assert DP.x0_estimate(prev, cur, None) is cur
    assert DP.x0_estimate(prev, cur, (0.5, 0.5)) is cur


def test_to_uint8_caps_the_long_side():
    img = DP.to_uint8(torch.rand(300, 120, 3), max_side = 128)
    assert img.dtype == torch.uint8 and img.shape[0] <= 128 and img.is_contiguous()


def test_every_mapped_family_has_a_consistent_map():
    assert FACTORS, "no fitted maps"
    for family, key in FAMILY_FACTORS.items():
        spec = DP.factors_for(family)
        if key not in FACTORS:
            assert spec is None
            continue
        assert spec.layout in ("tokens", "bchw", "bcthw")
        outs = 3 * spec.patch * spec.patch
        assert len(spec.bias) == outs
        assert all(len(row) == outs for row in spec.weight)
    assert DP.factors_for("no-such-family") is None


def test_scheduler_step_preview_feeds_and_restores():
    seen = []

    class _Prev:
        def on_step(
            self,
            latents,
            scheduler = None,
            final = False,
        ):
            seen.append(latents)

    class _Sched:
        def step(self, x):
            return (x + 1,)

    class _Pipe:
        scheduler = _Sched()

    pipe = _Pipe()
    with DP.scheduler_step_preview(pipe, _Prev()):
        assert pipe.scheduler.step(1) == (2,)
    assert seen == [2]
    assert "step" not in pipe.scheduler.__dict__
    with DP.scheduler_step_preview(pipe, None):
        assert "step" not in pipe.scheduler.__dict__


def test_create_declines_without_a_map_or_off_cuda(monkeypatch):
    monkeypatch.delenv(DP.PREVIEW_ENV, raising = False)
    pub = lambda url, seq: None
    assert (
        DP.LatentPreviewer.create(
            family = "no-such", requested = None, height = 64, width = 64, device = "cpu", publish = pub
        )
        is None
    )
    assert (
        DP.LatentPreviewer.create(
            family = "flux.1", requested = None, height = 64, width = 64, device = "cpu", publish = pub
        )
        is None
    )
    assert (
        DP.LatentPreviewer.create(
            family = "flux.1", requested = False, height = 64, width = 64, device = "cuda", publish = pub
        )
        is None
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_cuda_preview_never_syncs_never_writes_and_publishes(monkeypatch):
    from PIL import Image

    monkeypatch.delenv(DP.PREVIEW_ENV, raising = False)
    got = []
    done = threading.Event()

    def publish(url, seq):
        got.append((url, seq))
        done.set()

    prev = DP.LatentPreviewer.create(
        family = "flux.1",
        requested = None,
        height = 512,
        width = 768,
        device = "cuda",
        publish = publish,
        min_interval_s = 0.0,
    )
    assert prev is not None
    lat = torch.randn(1, (512 // 16) * (768 // 16), 64, device = "cuda", dtype = torch.bfloat16)
    before = lat.clone()

    class _Sched:
        sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0], device = "cuda")
        _step_index = 0

    sched = _Sched()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError):
            lat.float().sum().item()
        for i in range(4):
            sched._step_index = i + 1
            prev.on_step(lat, sched)
            lat = lat * 0.9  # the preview must have read its own copy in stream order
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert not prev.failed
    assert prev.emitted == 3
    assert done.wait(10.0)
    prev.finish()
    url, _ = got[-1]
    assert url.startswith("data:image/jpeg;base64,")
    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert img.size == (768 // 16 * 2, 512 // 16 * 2)
    del before


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
@pytest.mark.parametrize("side", [1024, 2048, 4096])
def test_cuda_preview_slot_fits_the_picture_at_every_size(monkeypatch, side):
    # at 2048 px a padded slot size picked a coarser divisor than to_uint8, dropping every frame
    monkeypatch.delenv(DP.PREVIEW_ENV, raising = False)
    got = []
    done = threading.Event()

    def publish(url, seq):
        got.append(url)
        done.set()

    prev = DP.LatentPreviewer.create(
        family = "flux.1",
        requested = None,
        height = side,
        width = side,
        device = "cuda",
        publish = publish,
        min_interval_s = 0.0,
    )
    assert prev is not None
    lat = torch.randn(1, (side // 16) ** 2, 64, device = "cuda", dtype = torch.bfloat16)

    class _Sched:
        sigmas = torch.tensor([1.0, 0.5, 0.0], device = "cuda")
        _step_index = 0

    sched = _Sched()
    for i in range(2):
        sched._step_index = i + 1
        prev.on_step(lat, sched)
    assert prev.emitted == 1
    assert done.wait(10.0)
    prev.finish()
    from PIL import Image

    img = Image.open(io.BytesIO(base64.b64decode(got[-1].split(",", 1)[1])))
    assert max(img.size) <= DP.MAX_SIDE and min(img.size) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_cuda_preview_leaves_the_latent_bit_identical(monkeypatch):
    monkeypatch.delenv(DP.PREVIEW_ENV, raising = False)
    prev = DP.LatentPreviewer.create(
        family = "flux.1",
        requested = None,
        height = 256,
        width = 256,
        device = "cuda",
        publish = lambda u, s: None,
        min_interval_s = 0.0,
    )
    lat = torch.randn(1, 256, 64, device = "cuda", dtype = torch.bfloat16)
    snap = lat.clone()
    for _ in range(3):
        prev.on_step(lat, None)
    prev.finish()
    torch.cuda.synchronize()
    assert torch.equal(lat, snap)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_snapshots_are_planned_by_step_and_publishing_keeps_only_the_newest(monkeypatch):
    """With CUDA graphs the host enqueues every step long before the GPU runs them, so a host clock cannot
    pace snapshots: they are spread by step, each in its own slot, and the worker publishes only the
    newest one the GPU has finished."""
    monkeypatch.delenv(DP.PREVIEW_ENV, raising = False)
    got = []
    prev = DP.LatentPreviewer.create(
        family = "flux.1",
        requested = None,
        height = 256,
        width = 256,
        device = "cuda",
        publish = lambda u, s: got.append(s),
        total_steps = 8,
        max_snapshots = 4,
        min_interval_s = 3600.0,
    )
    assert prev.stride == 2
    lat = torch.randn(1, 256, 64, device = "cuda")
    # a long queued kernel keeps every snapshot in flight, like a run-ahead host
    torch.cuda._sleep(int(2e8))
    for _ in range(8):
        prev.on_step(lat, None)
    assert prev.emitted == 4
    torch.cuda.synchronize()
    prev.finish(timeout = 30.0)
    # hour-long rate limit: first snapshot publishes, finish() flushes the newest
    assert got and got[-1] == 4
    assert len(got) <= 2


def test_smoothing_removes_the_period_two_patch_grid():
    # a 2x2 checker (the packed-patch artefact) on a flat picture comes back flat
    h = w = 16
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing = "ij")
    checker = ((yy + xx) % 2).float() * 0.2 - 0.1
    rgb = (0.5 + checker)[..., None].expand(h, w, 3).contiguous()
    out = DP.smooth_patch_grid(rgb)
    assert tuple(out.shape) == (h, w, 3)
    assert float((out[1:-1, 1:-1] - 0.5).abs().max()) < 1e-6
    ramp = torch.linspace(0, 1, w).expand(h, w)[..., None].expand(h, w, 3).contiguous()
    assert torch.allclose(DP.smooth_patch_grid(ramp)[2:-2, 2:-2], ramp[2:-2, 2:-2], atol = 1e-6)
