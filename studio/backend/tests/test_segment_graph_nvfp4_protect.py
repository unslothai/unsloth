# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU: protected NVFP4 steps leave the segment; the fake capture replays its recorded branch, like a CUDA graph."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from core.inference import diffusion_block_graph as bg
from core.inference.diffusion_nvfp4_protect import NVFP4StepController

STEPS = 6
SPEC = "0,-1"
PROTECTED = (0, STEPS - 1)


class _Block(nn.Module):
    def __init__(self, seed: int, ctl) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.lin = nn.Linear(4, 4)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(4, 4, generator = gen))
            self.lin.bias.copy_(torch.randn(4, generator = gen))
        self.ctl = ctl

    def forward(self, x, temb):
        out = self.lin(x)
        if bg._protect_live(self.ctl):
            out = out * 0.5 - 1.0
        return out + temb


@pytest.fixture(autouse = True)
def _cpu_segment(monkeypatch):
    monkeypatch.delenv(bg.SEGMENT_PROTECT_ENV, raising = False)
    monkeypatch.delenv(bg.SEGMENT_GRAPHS_ENV, raising = False)
    monkeypatch.delenv(bg.BLOCK_GRAPHS_ENV, raising = False)
    # Read CPU tensors as card 0 tensors, so the segment's device gates pass; no CUDA call is reached.
    monkeypatch.setattr(torch.Tensor, "device", property(lambda self: torch.device("cuda", 0)))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(bg, "_lazy_chain", lambda root: True)
    monkeypatch.setattr(bg._WeightView, "placement", lambda self, index: ("w", index))
    monkeypatch.setattr(bg.BlockGraph, "_record", lambda self, full, live: None)
    monkeypatch.setattr(bg.SegmentGraph, "_capture", _fake_capture)
    yield


class _FakeGraph:
    def __init__(self, seg, rec, static_in, outputs, baked) -> None:
        self.seg, self.rec, self.static_in, self.outputs, self.baked = (
            seg,
            rec,
            static_in,
            outputs,
            baked,
        )

    def replay(self) -> None:
        ctls = [g.block.ctl for g in self.seg.graphs if g.block.ctl is not None]
        saved = [(c, c.armed, c.protected) for c in ctls]
        for c in ctls:
            c.armed, c.protected = self.baked
        try:
            fresh = self.seg._chain(self.rec, self.static_in)
        finally:
            for c, armed, protected in saved:
                c.armed, c.protected = armed, protected
        for dsts, srcs in zip(self.outputs, fresh):
            for dst, src in zip(dsts, srcs):
                dst.copy_(src)


def _fake_capture(self, full, live):
    rec = self.seen.pop(full)
    ctl = next((g.block.ctl for g in self.graphs if g.block.ctl is not None), None)
    baked = (ctl.armed, ctl.protected) if ctl is not None else (False, False)
    assert not bg._protect_live(ctl), "a segment must never record a protected step"
    entry = bg._SegEntry()
    entry.static_in = [t.clone() for t in live]
    entry.outputs = self._chain(rec, entry.static_in)
    entry.graph = _FakeGraph(self, rec, entry.static_in, entry.outputs, baked)
    entry.record, entry.slots, entry.replays, entry.pool = rec, (), 0, None
    self.cache[full] = entry
    self.stats["segment_captures"] += 1
    return entry


def _build(ctl, *, protect = True):
    blocks = [_Block(i, ctl) for i in range(3)]
    root = nn.ModuleList(blocks)
    shared = bg._Shared(0, None)
    shared.root = root
    graphs = [bg.BlockGraph(b, b.forward, shared) for b in blocks]
    for g in graphs:
        g.protect = ctl if (protect and ctl is not None and ctl.armed) else False
    seg = bg.SegmentGraph(graphs, shared)
    return blocks, graphs, seg


def _run(
    graphs,
    blocks,
    ctl,
    generations: int = 1,
):
    x = torch.randn(2, 4, generator = torch.Generator().manual_seed(7))
    temb = torch.randn(2, 4, generator = torch.Generator().manual_seed(8))
    got, want = [], []
    for _ in range(generations):
        if ctl is not None:
            ctl.begin(STEPS)
        for _step in range(STEPS):
            with torch.no_grad():
                h = x
                for g in graphs:
                    h = g(h, temb)
                ref = x
                for b in blocks:
                    ref = b(ref, temb)
            got.append(h.clone())
            want.append(ref)
            if ctl is not None:
                ctl.advance()
    return got, want


def test_protected_steps_fall_back_and_the_rest_replay_one_recording():
    ctl = NVFP4StepController(SPEC)
    blocks, graphs, seg = _build(ctl)
    got, want = _run(graphs, blocks, ctl, generations = 2)
    for step, (a, b) in enumerate(zip(got, want)):
        assert torch.equal(a, b), f"step {step % STEPS} differs from eager"
    s = seg.stats
    unprotected = STEPS - len(PROTECTED)
    # Generation 1: record on step 1, capture + replay on 2, replay 3-4. Generation 2: all four replay.
    assert s["segment_records"] == 1 and s["segment_captures"] == 1
    assert s["segment_replays"] == (unprotected - 1) + unprotected
    assert s["segment_protected_steps"] == 2 * len(PROTECTED)
    assert len(seg.cache) == 1
    ((key, placement),) = seg.cache.keys()
    assert placement[1] == (("nvfp4_protect", False),)


def test_a_protected_step_mid_run_keeps_the_recording_for_the_next_step():
    ctl = NVFP4StepController("3")
    blocks, graphs, seg = _build(ctl)
    got, want = _run(graphs, blocks, ctl)
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert seg.stats["segment_captures"] == 1
    assert seg.stats["segment_protected_steps"] == 1
    assert seg.stats["segment_replays"] == 4  # step 0 records; 1, 2, 4 and 5 replay


def test_the_kill_switch_keeps_the_whole_run_per_block(monkeypatch):
    monkeypatch.setenv(bg.SEGMENT_PROTECT_ENV, "0")
    assert not bg.segment_protect_fallback_enabled()
    ctl = NVFP4StepController(SPEC)
    blocks, graphs, seg = _build(ctl)
    got, want = _run(graphs, blocks, ctl, generations = 2)
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    s = seg.stats
    assert s["segment_replays"] == 0 and s["segment_captures"] == 0 and s["segment_records"] == 0
    assert s.get("segment_protected_steps", 0) == 0
    assert not seg.cache


def test_an_unarmed_schedule_keys_and_replays_as_before():
    ctl = NVFP4StepController("")
    assert not ctl.armed
    blocks, graphs, seg = _build(ctl)
    got, want = _run(graphs, blocks, ctl, generations = 2)
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    s = seg.stats
    assert s["segment_captures"] == 1 and s.get("segment_protected_steps", 0) == 0
    assert s["segment_replays"] == 2 * STEPS - 1  # only step 0 of the first generation runs eager
    ((key, placement),) = seg.cache.keys()
    assert placement == (("w", 0),) * 3


def test_a_suspended_schedule_is_not_a_protected_step():
    ctl = NVFP4StepController(SPEC)
    ctl.begin(STEPS)
    assert bg._protect_live(ctl)
    ctl.armed = False
    assert not bg._protect_live(ctl)
    assert not bg._protect_live(False) and not bg._protect_live(None)


def test_handle_stats_surface_the_protected_step_count():
    ctl = NVFP4StepController(SPEC)
    blocks, graphs, seg = _build(ctl)
    _run(graphs, blocks, ctl)
    handle = bg.BlockGraphSet.__new__(bg.BlockGraphSet)
    handle.graphs, handle.segments = graphs, [seg]
    total = handle.stats
    assert total["segment_protected_steps"] == len(PROTECTED)
    assert total["segment_replays"] == STEPS - len(PROTECTED) - 1
