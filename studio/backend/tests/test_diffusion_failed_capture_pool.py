# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A capture that fails must not break the captures after it.

``capture_end`` raises in ``cudaStreamEndCapture`` before either caching allocator (device, and from torch 2.11 the
pinned host one) leaves the capture's pool. Python can only take the device allocator off it, so the host allocator
keeps "recording" to that pool for the rest of the process, with a filter that points at the failed CUDAGraph.
Studio therefore retires the pool, records later captures into a fresh one, and keeps the failed graph alive.
"""

from __future__ import annotations

import pytest
import torch

import core.inference.diffusion_block_graph as bg
import core.inference.diffusion_cuda_graph as cg


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


class _Block(torch.nn.Module):
    def __init__(self, sync: bool = False) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(16, 16)
        self.sync = sync

    def forward(
        self,
        x,
        return_dict = True,
    ):
        y = self.lin(x)
        if self.sync and torch.cuda.is_current_stream_capturing():
            y = y + float(y.sum().item()) * 0  # a host read: invalidates the recording
        return (y,) if return_dict is False else y


def _record_twice(graph, x):
    with torch.inference_mode():
        graph(x)  # first sighting: eager
        return graph(x)  # records, then replays


def test_a_block_records_after_another_block_failed_to(monkeypatch):
    _cuda()
    monkeypatch.setattr(cg, "_FAILED_GRAPHS", [], raising = False)
    shared = bg._Shared(torch.cuda.current_device())
    good_first = _Block().cuda()
    bad = _Block(sync = True).cuda()
    good = _Block().cuda()
    x = torch.randn(4, 16, device = "cuda")
    first = bg.BlockGraph(good_first, good_first.forward, shared)
    _record_twice(first, x)
    assert first.stats["captures"] == 1  # the shared pool now holds a live graph

    failing = bg.BlockGraph(bad, bad.forward, shared)
    _record_twice(failing, x)
    assert failing.poisoned and failing.stats["captures"] == 0

    after = bg.BlockGraph(good, good.forward, shared)
    out = _record_twice(after, x)
    assert after.capture_error is None, after.capture_error
    assert after.stats["captures"] == 1 and not after.poisoned
    with torch.inference_mode():
        assert torch.equal(after(x), good(x))
        assert torch.equal(first(x), good_first(x))
    assert torch.equal(out, good(x))
    assert (
        len(cg._FAILED_GRAPHS) == 1
    )  # the failed recording is kept, never freed under the allocator
    torch.randn(4, device = "cuda")  # the RNG left capture mode


def test_a_step_graph_records_after_another_one_failed_to(monkeypatch):
    _cuda()
    monkeypatch.setattr(cg, "_FAILED_GRAPHS", [], raising = False)
    monkeypatch.setattr(cg, "_POOL_BOX", [None])
    x = torch.randn(4, 16, device = "cuda")
    mods = [_Block().cuda(), _Block(sync = True).cuda(), _Block().cuda()]
    handles = [cg.GraphedForward(m).install().enable() for m in mods]
    with torch.inference_mode():
        mods[0](x, return_dict = False)
        assert handles[0].stats["captures"] == 1  # the shared step pool now holds a live graph
        mods[1](x, return_dict = False)
        assert handles[1].poisoned
        out = mods[2](x, return_dict = False)[0]
        assert handles[2].capture_error is None, handles[2].capture_error
        assert handles[2].stats["captures"] == 1
        assert torch.equal(out, mods[2].lin(x))
        assert torch.equal(mods[0](x, return_dict = False)[0], mods[0].lin(x))
    assert len(cg._FAILED_GRAPHS) == 1
    for h in handles:
        h.uninstall()


def test_captures_stop_once_the_failure_budget_is_spent(monkeypatch):
    _cuda()
    monkeypatch.setattr(cg, "_FAILED_GRAPHS", [object()] * cg.MAX_FAILED_CAPTURES)
    shared = bg._Shared(torch.cuda.current_device())
    good = _Block().cuda()
    graph = bg.BlockGraph(good, good.forward, shared)
    x = torch.randn(4, 16, device = "cuda")
    out = _record_twice(graph, x)
    assert graph.stats["captures"] == 0 and graph.poisoned
    assert "failed" in graph.capture_error["msg"]
    assert torch.equal(out, good(x))


def test_a_collision_with_another_recording_does_not_spend_the_failure_budget(monkeypatch):
    monkeypatch.setattr(cg, "_FAILED_GRAPHS", [])
    monkeypatch.setattr(cg, "_COLLIDED_GRAPHS", [], raising = False)
    healed = []
    monkeypatch.setattr(cg, "_heal_generators", lambda: healed.append(1))
    for _ in range(cg.MAX_FAILED_CAPTURES + 1):
        cg.retire_failed_capture(
            object(), None, RuntimeError("beginAllocateToPool: already recording to mempool_id")
        )
    assert not cg.captures_exhausted()
    assert healed == []  # the generators belong to the other thread's live capture
