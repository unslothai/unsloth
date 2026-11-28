# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 live denoise preview: the fitted map is registered and fits H3's (1, 2, 2) packed rows, the modular
loop's video scheduler.step feeds the previewer only when a preview is wanted (never the audio stream, never an API
request, never with the kill switch), and on CUDA a preview of H3 rows never synchronises."""

from __future__ import annotations

import threading
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_preview as DP
from core.inference.diffusion_preview_factors import FACTORS, FAMILY_FACTORS

H3 = "minimax-h3"
CHANNELS = 24


def _patchify(latents):
    """diffusers' patchify_video_latents for H3's (1, 2, 2) patch: rows frame-major, features (c, dy, dx)."""
    b, c, t, h, w = latents.shape
    x = latents.reshape(b, c, t, 1, h // 2, 2, w // 2, 2).permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(-1, c * 4).contiguous()


def test_h3_family_has_a_fitted_token_map():
    assert FAMILY_FACTORS.get(H3) == "minimax_h3"
    spec = DP.factors_for(H3)
    assert spec is not None
    assert (spec.layout, spec.down, spec.patch, spec.channels) == ("tokens", 32, 2, CHANNELS * 4)
    assert all(len(row) == 12 for row in spec.weight) and len(spec.bias) == 12
    # One 24 -> RGB map per latent pixel, placed block-diagonally: sub-pixel s of a token reads only its own channels.
    for c in range(CHANNELS):
        for s in range(4):
            row = spec.weight[c * 4 + s]
            for o in range(4):
                block = row[o * 3 : o * 3 + 3]
                if o == s:
                    assert block == spec.weight[c * 4][0:3]
                else:
                    assert block == (0.0, 0.0, 0.0)
    assert spec.bias == spec.bias[:3] * 4
    # The normalised latent mean (all zeros) lands inside the picture, not at a clamp.
    assert all(0.1 < v < 0.9 for v in spec.bias)


def test_h3_rows_project_like_the_per_pixel_map_on_the_first_frame():
    spec = DP.factors_for(H3)
    gh, gw, frames = 5, 7, 3
    g = torch.Generator().manual_seed(0)
    lat = torch.randn(1, CHANNELS, frames, gh * 2, gw * 2, generator = g)
    rows = _patchify(lat)
    grid = DP.latent_grid(rows, spec, gh * spec.down, gw * spec.down)
    assert grid is not None and grid.shape == (gh, gw, CHANNELS * 4)
    assert grid.data_ptr() == rows.data_ptr()  # a view of the first frame, no copy
    weight = torch.tensor(spec.weight, dtype = torch.float32)
    bias = torch.tensor(spec.bias, dtype = torch.float32)
    rgb = DP.project(grid, weight, bias, spec.patch)
    pixel_w = weight[0::4, 0:3]  # (24, 3)
    expect = lat[0, :, 0].permute(1, 2, 0) @ pixel_w + bias[:3]
    assert rgb.shape == (gh * 2, gw * 2, 3)
    torch.testing.assert_close(rgb, expect, rtol = 1e-5, atol = 1e-5)
    try:
        from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
    except Exception:  # noqa: BLE001 - older diffusers: the local copy above stands in
        return
    assert torch.equal(patchify_video_latents(lat, (1, 2, 2)), rows)


class _Recorder:
    def __init__(self):
        self.seen = []

    def on_step(
        self,
        latents,
        scheduler = None,
        final = False,
    ):
        self.seen.append((latents.clone(), scheduler))


def _h3_pipe(sampler = "res_multistep"):
    sched_mod = pytest.importorskip("diffusers.schedulers.scheduling_minimax_h3")
    from core.inference.video_minimax_h3_sampling import install_h3_schedulers

    pipe = types.SimpleNamespace(
        scheduler = sched_mod.MiniMaxH3Scheduler(shift = 12.0),
        audio_scheduler = sched_mod.MiniMaxH3Scheduler(shift = 3.0),
    )
    assert install_h3_schedulers(pipe, sampler)
    return pipe


def _run_h3_update_blocks(
    pipe,
    steps = 4,
    gh = 3,
    gw = 4,
    frames = 2,
    cond_rows = 0,
):
    """The real diffusers update block (MiniMaxH3LoopSchedulerStep) over a few steps of the packed sequence."""
    denoise = pytest.importorskip("diffusers.modular_pipelines.minimax_h3.denoise")
    pipe.scheduler.set_timesteps(steps)
    pipe.audio_scheduler.set_timesteps(steps)
    n_video = cond_rows + frames * gh * gw
    n_audio = 2 * 6
    g = torch.Generator().manual_seed(1)
    state = types.SimpleNamespace(
        latents = torch.randn(n_video, CHANNELS * 4, generator = g),
        audio_latents = torch.randn(n_audio, 32, generator = g),
        num_condition_video_rows = cond_rows,
        num_condition_audio_rows = 0,
        audio_timesteps = pipe.audio_scheduler.timesteps,
    )
    block = denoise.MiniMaxH3LoopSchedulerStep()
    for i, t in enumerate(pipe.scheduler.timesteps):
        state.noise_pred = torch.randn(1, n_video, CHANNELS * 4, generator = g)
        state.audio_noise_pred = torch.randn(1, n_audio, 32, generator = g)
        block(pipe, state, i, t)
    return state


# res_multistep is the default; euler is the PDD heads' sampler and UNSLOTH_H3_SAMPLER=euler.
@pytest.mark.parametrize("sampler", ["res_multistep", "euler"])
def test_h3_video_steps_feed_the_previewer_and_audio_never(sampler):
    pipe = _h3_pipe(sampler)
    rec = _Recorder()
    gh, gw, frames, cond = 3, 4, 2, 12
    with DP.scheduler_step_preview(pipe, rec):
        state = _run_h3_update_blocks(pipe, steps = 4, gh = gh, gw = gw, frames = frames, cond_rows = cond)
    assert len(rec.seen) == 3
    assert all(s is pipe.scheduler for _, s in rec.seen)
    last, _ = rec.seen[-1]
    # The generated video rows only (conditioning rows excluded), unbatched [N, 96]: the first frame unpacks.
    assert last.shape == (frames * gh * gw, CHANNELS * 4)
    assert torch.equal(last, state.latents[cond:])
    spec = DP.factors_for(H3)
    grid = DP.latent_grid(last, spec, gh * spec.down, gw * spec.down)
    assert grid is not None and torch.equal(grid.reshape(-1, CHANNELS * 4), last[: gh * gw])
    # The Euler x0 pair comes off the video scheduler's own sigmas, as for every other family.
    assert DP._sigma_pair(pipe.scheduler) is not None
    assert "step" not in pipe.scheduler.__dict__ and "step" not in pipe.audio_scheduler.__dict__


def test_h3_preview_never_changes_the_steps():
    a, b = _h3_pipe(), _h3_pipe()
    with DP.scheduler_step_preview(a, _Recorder()):
        with_preview = _run_h3_update_blocks(a)
    without = _run_h3_update_blocks(b)
    assert torch.equal(with_preview.latents, without.latents)
    assert torch.equal(with_preview.audio_latents, without.audio_latents)


@pytest.fixture
def _fake_cuda(monkeypatch):
    """Lets LatentPreviewer.create reach its constructor on a CPU host; the constructor itself only records."""
    made = []

    def _init(self, spec, **kwargs):
        self.spec = spec
        self.kwargs = kwargs
        self.calls = 0
        made.append(self)

    def _on_step(
        self,
        latents,
        scheduler = None,
        final = False,
    ):
        self.calls += 1

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(DP.LatentPreviewer, "__init__", _init)
    monkeypatch.setattr(DP.LatentPreviewer, "on_step", _on_step)
    monkeypatch.delenv(DP.PREVIEW_ENV, raising = False)
    return made


def _create(requested):
    return DP.LatentPreviewer.create(
        family = H3,
        requested = requested,
        height = 544,
        width = 960,
        device = "cuda",
        publish = lambda url, seq: None,
        total_steps = 20,
    )


@pytest.mark.parametrize("requested", [None, True])
def test_h3_render_previews_when_wanted(_fake_cuda, requested):
    prev = _create(requested)
    assert prev is not None and prev.spec is DP.factors_for(H3)
    pipe = _h3_pipe()
    with DP.scheduler_step_preview(pipe, prev):
        _run_h3_update_blocks(pipe, steps = 5)
    assert prev.calls == 4


def test_h3_api_request_never_previews(_fake_cuda):
    # Headless routes (API, chat tool, MCP) pass live_preview=False.
    assert _create(False) is None
    pipe = _h3_pipe()
    original = pipe.scheduler.step
    with DP.scheduler_step_preview(pipe, None):
        assert "step" not in pipe.scheduler.__dict__
        _run_h3_update_blocks(pipe)
    assert pipe.scheduler.step == original
    assert _fake_cuda == []


def test_h3_kill_switch(_fake_cuda, monkeypatch):
    monkeypatch.setenv(DP.PREVIEW_ENV, "0")
    assert _create(None) is None
    assert _create(True) is None
    assert DP.preview_available(H3, "cuda") is False
    monkeypatch.delenv(DP.PREVIEW_ENV)
    assert DP.preview_available(H3, "cuda") is True
    assert _fake_cuda == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_cuda_h3_rows_preview_without_a_sync(monkeypatch):
    monkeypatch.delenv(DP.PREVIEW_ENV, raising = False)
    done = threading.Event()
    got = []

    def publish(url, seq):
        got.append(url)
        done.set()

    prev = DP.LatentPreviewer.create(
        family = H3,
        requested = True,
        height = 544,
        width = 960,
        device = "cuda",
        publish = publish,
        min_interval_s = 0.0,
    )
    assert prev is not None
    gh, gw = 544 // 32, 960 // 32
    rows = torch.randn(4 * gh * gw, CHANNELS * 4, device = "cuda")
    sched = types.SimpleNamespace(sigmas = torch.linspace(1.0, 0.0, 6, device = "cuda"), _step_index = 0)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for i in range(4):
            sched._step_index = i + 1
            prev.on_step(rows, sched)
            rows = rows * 0.9
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert not prev.failed and prev.emitted == 3
    assert done.wait(10.0)
    prev.finish()
    assert got[-1].startswith("data:image/jpeg;base64,")
