# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A latent shape keeps the VAE route of its first decode (Wan2.2-TI2V-5B render 2 differed from render 1)."""

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import video_vae_untiled as U  # noqa: E402

GiB = 2**30
FIVE_B = "wan2.2-ti2v-5b"
LATENT = (1, 48, 5, 44, 80)  # 1280x704, 17 frames


class _FakeVae(torch.nn.Module):
    def __init__(self, oom_sizes = ()):
        super().__init__()
        self.decoder = torch.nn.Linear(2, 2).half()
        self.use_tiling = True
        self.spatial_compression_ratio = 16
        self.tile_sample_min_height = self.tile_sample_min_width = 256
        self.tile_sample_stride_height = self.tile_sample_stride_width = 192
        self.seen = []
        self.oom_sizes = set(oom_sizes)

    def decode(self, z, *args, **kwargs):
        route = (
            ("untiled",)
            if not self.use_tiling
            else (
                self.tile_sample_min_height,
                self.tile_sample_stride_height,
                self.tile_sample_min_width,
                self.tile_sample_stride_width,
            )
        )
        self.seen.append(route)
        if route[0] in self.oom_sizes:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return z


def _install(
    monkeypatch,
    frees,
    vae = None,
):
    vae = vae or _FakeVae()
    it = iter(frees)
    monkeypatch.setattr(U, "_free_bytes", lambda device: next(it))
    monkeypatch.setattr(
        "core.inference.diffusion_vae_oom.clear_partial_decode", lambda vae, z: None
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    assert U.install_untiled_decode(types.SimpleNamespace(vae = vae), FIVE_B)
    return vae


def _render(
    vae,
    n = 1,
    shape = LATENT,
):
    for _ in range(n):
        vae.decode(torch.zeros(*shape))


def _mib(free_mib):
    return int(free_mib * 2**20)


RENDER_FREES = (_mib(10695), _mib(6577), _mib(6577), _mib(10695))


def test_wide_route_of_the_first_decode_is_kept_when_free_memory_drops(monkeypatch):
    vae = _install(monkeypatch, RENDER_FREES)
    _render(vae, 4)
    assert vae.seen[0][0] == 704 and vae.seen[0][2] == 704
    assert vae.seen == [vae.seen[0]] * 4


def test_untiled_route_is_kept_inside_the_proportional_margin(monkeypatch):
    need = U.untiled_decode_bytes(FIVE_B, LATENT, itemsize = 2)
    # Below need * 1.25 + 2 GiB (a first decode would tile), above need * 1.25.
    squeezed = int(need * U._MARGIN + U._MARGIN_BYTES) - GiB
    vae = _install(monkeypatch, (100 * GiB, squeezed, squeezed))
    _render(vae, 3)
    assert vae.seen == [("untiled",)] * 3


def test_route_that_no_longer_fits_repicks_and_the_latch_survives(monkeypatch):
    plan_bytes = None
    vae = _install(monkeypatch, (_mib(10695), int(2.6 * GiB), _mib(10695)))
    _render(vae, 1)
    plan_bytes = U.wide_plan_bytes(FIVE_B, LATENT, 16, vae.seen[0], itemsize = 2)
    assert plan_bytes * U._MARGIN > 2.6 * GiB
    _render(vae, 2)
    assert vae.seen[1] != vae.seen[0]  # forced: the kept plan cannot fit
    assert vae.seen[2] == vae.seen[0]  # memory back: the first route again


def test_oom_on_the_first_route_latches_the_route_that_ran(monkeypatch):
    vae = _install(monkeypatch, (_mib(10695),) * 3, vae = _FakeVae(oom_sizes = (704,)))
    _render(vae, 3)
    assert vae.seen[0][0] == 704
    assert vae.seen[1:] == [(256, 192, 256, 192)] * 3


def test_each_latent_shape_keeps_its_own_route(monkeypatch):
    small = (1, 48, 5, 22, 40)
    vae = _install(monkeypatch, (100 * GiB, _mib(10695), _mib(6577), _mib(6577)))
    _render(vae, 1, shape = small)
    _render(vae, 1)
    _render(vae, 1, shape = small)
    _render(vae, 1)
    assert vae.seen[0] == vae.seen[2] == ("untiled",)
    assert vae.seen[1] == vae.seen[3] and vae.seen[1][0] == 704


def test_kill_switch_picks_from_free_memory_every_call(monkeypatch):
    monkeypatch.setenv(U.ROUTE_LATCH_ENV, "0")
    vae = _install(monkeypatch, RENDER_FREES)
    _render(vae, 2)
    assert vae.seen[0][0] == 704 and vae.seen[1] != vae.seen[0]


def test_wide_plan_bytes_matches_the_planner_price():
    free = _mib(10695)
    plan = U.wide_tile_plan(FIVE_B, LATENT, 16, free, itemsize = 2)
    price = U.wide_plan_bytes(FIVE_B, LATENT, 16, plan, itemsize = 2)
    assert price * U._MARGIN + U._MARGIN_BYTES <= free
    assert U.wide_plan_bytes("unknown-family", LATENT, 16, plan) is None


def test_stock_first_then_room_for_untiled_upgrades_and_stays(monkeypatch):
    # Render 1 short of memory (a compile in flight): stock. Room later: untiled, then kept inside the margin.
    need = U.untiled_decode_bytes(FIVE_B, LATENT, itemsize = 2)
    squeezed = int(need * U._MARGIN + U._MARGIN_BYTES) - GiB
    vae = _install(monkeypatch, (int(2.6 * GiB), 100 * GiB, squeezed, squeezed))
    _render(vae, 4)
    assert vae.seen[0] == (256, 192, 256, 192)
    assert vae.seen[1:] == [("untiled",)] * 3


def test_narrow_wide_first_then_room_for_a_wider_plan_relatches(monkeypatch):
    vae = _install(monkeypatch, (_mib(6577), _mib(10695), _mib(6577), _mib(6577)))
    _render(vae, 4)
    narrow, wider = vae.seen[0], vae.seen[1]
    assert narrow[0] * narrow[2] < wider[0] * wider[2] and wider[0] == 704 and wider[2] == 704
    assert (
        vae.seen[2:] == [wider] * 2
    )  # the wider plan is the latch now: no fall back to the narrow one
