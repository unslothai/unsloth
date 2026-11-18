# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""int8 weight-only GEMV tile choice: a grouped layer tries its one-group tile first, then the configs' own K tiles
inside a wider group; the config that launched is remembered per (card, layout), so a grouped layer whose tile does
not fit never demotes the card's per-row Linears. CPU only: the launch is replaced by a recorder."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_int8_gemv as G  # noqa: E402

_REAL_KERNEL = G._kernel


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(G.GROUP_TILES_ENV, raising = False)
    monkeypatch.setattr(G, "_DEVICE_CFG", {})
    monkeypatch.setattr(G, "COUNTS", {"gemv": 0, "fallback": 0})
    monkeypatch.setattr(G, "_kernel", lambda: object())


class _CudaX:
    """A CPU tensor that _run treats as living on cuda:0 (the launch itself is stubbed)."""

    def __init__(self, t):
        self.t, self.shape, self.is_cuda, self.device = t, t.shape, True, torch.device("cuda", 0)

    def stride(self, *dim):
        return self.t.stride(*dim)


def _layer(n, k, group):
    q = torch.zeros(n, k, dtype = torch.int8)
    s = torch.ones(n, k // group if group else 1, dtype = torch.bfloat16)
    return _CudaX(torch.zeros(2, k, dtype = torch.bfloat16)), q, (s if group else s.reshape(-1))


def _smem_limited(monkeypatch, fits):
    tried = []

    def launch(x, q, s2, bias, oidx, ow, plan):
        tried.append(plan)
        if not fits(plan):
            raise RuntimeError("OutOfResources: shared memory")
        return ("y", plan)

    monkeypatch.setattr(G, "_launch", launch)
    return tried


def test_plans():
    assert G._plans(0) == G._CONFIGS
    assert G._plans(512) == ((128, 512, 8, 3), (64, 512, 4, 2), (128, 128, 8, 3), (64, 64, 4, 2))
    assert G._plans(128) == ((128, 128, 8, 3), (64, 128, 4, 2), (64, 64, 4, 2))
    assert G._plans(64) == ((128, 64, 8, 3), (64, 64, 4, 2))


def test_kill_switch_keeps_one_group_tiles(monkeypatch):
    monkeypatch.setenv(G.GROUP_TILES_ENV, "0")
    assert G._plans(512) == ((128, 512, 8, 3), (64, 512, 4, 2))


def test_wide_group_runs_split_tiles_and_leaves_per_row_alone(monkeypatch):
    # a T4-sized card: only tiles of at most 128 K columns fit
    tried = _smem_limited(monkeypatch, lambda plan: plan[1] <= 128)
    y = G._run(*_layer(256, 4096, 1024), None)
    assert y == ("y", (128, 128, 8, 3))
    assert tried == [(128, 1024, 8, 3), (64, 1024, 4, 2), (128, 128, 8, 3)]
    assert G._DEVICE_CFG == {(0, 1024): 2}
    tried.clear()
    assert G._run(*_layer(256, 4096, 0), None) == ("y", (128, 128, 8, 3))
    assert tried == [(128, 128, 8, 3)]
    assert G._DEVICE_CFG == {(0, 1024): 2, (0, 0): 0}
    assert G.COUNTS == {"gemv": 2, "fallback": 0}


def test_a_group_that_fits_keeps_its_one_group_tile(monkeypatch):
    tried = _smem_limited(monkeypatch, lambda plan: True)
    assert G._run(*_layer(256, 4096, 512), None) == ("y", (128, 512, 8, 3))
    assert tried == [(128, 512, 8, 3)]


def test_remembered_plan_is_reused(monkeypatch):
    tried = _smem_limited(monkeypatch, lambda plan: plan[0] == 64)
    G._run(*_layer(256, 4096, 512), None)
    tried.clear()
    G._run(*_layer(256, 4096, 512), None)
    assert tried == [(64, 512, 4, 2)]


def test_nothing_fits_falls_back_for_that_layout_only(monkeypatch):
    _smem_limited(monkeypatch, lambda plan: plan[1] < 128 and plan[0] == 0)
    seen = []
    monkeypatch.setattr(G, "reference", lambda *a, **k: seen.append(1) or "ref")
    assert G._run(*_layer(256, 4096, 512), None) == "ref"
    assert G._DEVICE_CFG == {(0, 512): 4}
    assert G._DEVICE_CFG.get((0, 0), 0) == 0


def test_rows_and_outliers_not_specialized():
    """M is the batch (1..MAX_ROWS) and KO the outlier count: neither may compile a binary of its own."""
    pytest.importorskip("triton")
    fn = _REAL_KERNEL()[1]
    assert [p.name for p in fn.params if p.do_not_specialize] == ["M", "KO"]
