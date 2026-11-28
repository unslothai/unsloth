# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Batch-size graphs on a real card: a size-2 graph recorded next to the size-1 one replays with no host sync
(torch.cuda.set_sync_debug_mode("error") after the warm-ups), in the batched graphs' own pool, matching the eager
step."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA card"),
]

from core.inference import diffusion_cuda_graph as cg  # noqa: E402


class _TinyDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(64, 64)
        self.out = torch.nn.Linear(64, 64)

    def forward(
        self,
        hidden_states,
        timestep = None,
        return_dict = True,
    ):
        h = torch.nn.functional.gelu(self.proj(hidden_states)) + timestep[:, None, None]
        return (self.out(h),)


@pytest.fixture(autouse = True)
def _clean():
    cg._POOL_BOX[0] = None
    cg._BATCH_POOL_BOX[0] = None
    cg._LIVE_WRAPPERS.clear()
    cg._CAPTURE_STREAMS.clear()
    yield
    torch.cuda.set_sync_debug_mode(0)
    cg._POOL_BOX[0] = None
    cg._BATCH_POOL_BOX[0] = None
    cg._LIVE_WRAPPERS.clear()
    cg._CAPTURE_STREAMS.clear()


def _inputs(rows: int, seed: int):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    x = torch.randn(rows, 16, 64, device = "cuda", generator = g)
    t = torch.rand(rows, device = "cuda", generator = g)
    return x, t


@torch.no_grad()
def test_size_two_graph_replays_without_a_host_sync_next_to_size_one(monkeypatch):
    monkeypatch.setenv(cg.REAL_WARMUP_ENV, "0")
    module = _TinyDiT().cuda().eval()
    handle = cg.GraphedForward(module).enable()
    # Warm-ups and recordings (the race's idle warm-up for size 2).
    for seed in range(cg.WARMUP_ITERS + 2):
        x, t = _inputs(1, seed)
        handle(x, timestep = t, return_dict = False)
        with cg.batch_rows(2, capture = True):
            x2, t2 = _inputs(2, 100 + seed)
            handle(x2, timestep = t2, return_dict = False)
    torch.cuda.synchronize()
    assert handle.stats["batch_captures"] == 1 and len(handle.cache) == 2
    assert not handle.poisoned
    single = {e.pool_token for k, e in handle.cache.items() if k not in handle._batched}
    batched = {e.pool_token for k, e in handle.cache.items() if k in handle._batched}
    assert (
        len(single) == 1 and len(batched) == 1 and not single & batched
    ), "batched graphs have their own pool"

    x1, t1 = _inputs(1, 7)
    x2, t2 = _inputs(2, 8)
    want1 = module(x1, timestep = t1, return_dict = False)[0]
    want2 = module(x2, timestep = t2, return_dict = False)[0]
    torch.cuda.synchronize()
    replays = handle.stats["replays"]
    torch.cuda.set_sync_debug_mode("error")
    got1 = handle(x1, timestep = t1, return_dict = False)[0]
    with cg.batch_rows(2):
        got2 = handle(x2, timestep = t2, return_dict = False)[0]
    torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    assert handle.stats["replays"] == replays + 2
    assert torch.allclose(got1, want1, rtol = 1e-4, atol = 1e-5)
    assert torch.allclose(got2, want2, rtol = 1e-4, atol = 1e-5)
