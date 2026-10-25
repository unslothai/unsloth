# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wide Wan VAE tiles: the fewest tiles whose estimate fits free memory, used only when the untiled decode does not."""

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import video_vae_untiled as U  # noqa: E402

GiB = 2**30
FIVE_B = "wan2.2-ti2v-5b"
WAN21 = "wan2.1-t2v-1.3b"


def _tiles(height, width, ratio, plan):
    min_h, stride_h, min_w, stride_w = plan
    return (
        len(range(0, height, stride_h // ratio)),
        len(range(0, width, stride_w // ratio)),
        min_h // ratio,
        min_w // ratio,
    )


def test_plan_covers_the_frame_with_fewer_tiles_than_stock():
    plan = U.wide_tile_plan(FIVE_B, (1, 48, 11, 44, 80), 16, 10 * GiB)
    assert plan is not None
    rows, cols, tile_h, tile_w = _tiles(44, 80, 16, plan)
    assert rows * cols < 4 * 7  # stock 256 / 192 px tiles at 1280x704
    min_h, stride_h, min_w, stride_w = plan
    assert (rows - 1) * stride_h + min_h >= 704 and (cols - 1) * stride_w + min_w >= 1280
    if cols > 1:
        assert min_w - stride_w == 64
    if rows == 1:
        assert stride_h >= 704


def test_plan_shrinks_tiles_with_free_memory_and_gives_up_below_one_fit():
    big = U.wide_tile_plan(FIVE_B, (1, 48, 11, 44, 80), 16, 10 * GiB)
    small = U.wide_tile_plan(FIVE_B, (1, 48, 11, 44, 80), 16, 6 * GiB)
    assert big is not None and small is not None
    assert small[0] * small[2] < big[0] * big[2]
    assert U.wide_tile_plan(FIVE_B, (1, 48, 11, 44, 80), 16, int(2.5 * GiB)) is None
    assert U.wide_tile_plan("unknown-family", (1, 48, 11, 44, 80), 16, 10 * GiB) is None


def test_plan_respects_the_wan21_ratio():
    plan = U.wide_tile_plan(WAN21, (1, 16, 21, 60, 104), 8, 6 * GiB)
    assert plan is not None
    assert all(v % 8 == 0 for v in plan)


class _FakeVae(torch.nn.Module):
    def __init__(self, oom_wide = False):
        super().__init__()
        self.decoder = torch.nn.Linear(2, 2).half()
        self.use_tiling = True
        self.spatial_compression_ratio = 16
        self.tile_sample_min_height = self.tile_sample_min_width = 256
        self.tile_sample_stride_height = self.tile_sample_stride_width = 192
        self.seen = []
        self.oom_wide = oom_wide

    def decode(self, z, *args, **kwargs):
        sizes = (self.tile_sample_min_height, self.tile_sample_min_width, self.use_tiling)
        self.seen.append(sizes)
        if self.oom_wide and self.use_tiling and sizes[0] != 256:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return z


@pytest.mark.parametrize("oom", [False, True])
def test_gate_decodes_wide_when_untiled_does_not_fit(monkeypatch, oom):
    vae = _FakeVae(oom_wide = oom)
    pipe = types.SimpleNamespace(vae = vae)
    monkeypatch.setattr(U, "_free_bytes", lambda device: 10 * GiB)
    monkeypatch.setattr(
        "core.inference.diffusion_vae_oom.clear_partial_decode", lambda vae, z: None
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    assert U.install_untiled_decode(pipe, FIVE_B)
    z = torch.zeros(1, 48, 11, 44, 80)
    vae.decode(z)
    assert vae.seen[0][2] is True and vae.seen[0][0] == 704
    if oom:
        assert vae.seen[-1][:2] == (256, 256)  # the retry runs at the stock size
    assert (vae.tile_sample_min_height, vae.tile_sample_stride_height) == (
        256,
        192,
    )  # restored either way


def test_kill_switch_keeps_stock_tiles(monkeypatch):
    monkeypatch.setenv(U.WIDE_TILES_ENV, "0")
    vae = _FakeVae()
    monkeypatch.setattr(U, "_free_bytes", lambda device: 10 * GiB)
    assert U.install_untiled_decode(types.SimpleNamespace(vae = vae), FIVE_B)
    vae.decode(torch.zeros(1, 48, 11, 44, 80))
    assert vae.seen == [(256, 256, True)]
