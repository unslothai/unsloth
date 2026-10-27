# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Streamed weights stay normal tensors under a render's inference_mode, else compiled blocks recompile."""

from __future__ import annotations

import contextlib
import copy
import types

import pytest
import torch

import core.inference.diffusion_memory as dm
import core.inference.diffusion_offload_prefetch as op


class _HostTensor:
    def __init__(self, n: int):
        self.n = n

    def is_pinned(self):
        return True

    def numel(self):
        return self.n

    def element_size(self):
        return 4

    def to(
        self,
        device,
        non_blocking = False,
    ):
        return torch.zeros(self.n)


class _Event:
    def record(self, stream = None):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass


def _prefetcher(monkeypatch, groups_n: int = 3):
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(op, "_go", lambda: types.SimpleNamespace())
    groups = []
    for _ in range(groups_n):
        params = [torch.nn.Parameter(torch.zeros(4), requires_grad = False) for _ in range(2)]
        groups.append(
            types.SimpleNamespace(
                modules = [],
                parameters = params,
                buffers = [],
                cpu_param_dict = {p: _HostTensor(4) for p in params},
                stream = object(),
            )
        )
    return op.GroupPrefetcher(None, groups, "cpu", depth = 1), groups


@pytest.mark.parametrize("grad", [False, True])
def test_copies_issued_under_inference_mode_are_normal_tensors(monkeypatch, grad):
    pf, groups = _prefetcher(monkeypatch)
    with torch.inference_mode():
        for g in groups:
            pf._issue(g)
        assert torch.is_inference_mode_enabled()
    with torch.set_grad_enabled(grad):
        for g in groups[:1]:
            pf._issue(g)
        assert torch.is_grad_enabled() is grad
    for g in groups:
        for p in g.parameters:
            assert not p.data.is_inference()
            assert not p.is_inference()
            assert p.data.grad_fn is None


def test_normal_tensors_restores_the_callers_modes():
    with torch.inference_mode():
        with op._normal_tensors():
            assert not torch.is_inference_mode_enabled()
            assert not torch.is_grad_enabled()
            assert not torch.empty(1).is_inference()
        assert torch.is_inference_mode_enabled()
        assert torch.empty(1).is_inference()
    with torch.enable_grad():
        with op._normal_tensors():
            assert not torch.is_grad_enabled()
        assert torch.is_grad_enabled()


class _Block(torch.nn.Linear):
    seen: list = []

    def forward(self, x):
        _Block.seen.append((id(self), bool(self.weight.is_inference()), self.weight.device.type))
        return super().forward(x)


class _Net(torch.nn.Module):
    def __init__(
        self,
        width = 1024,
        blocks = 8,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(64, width)
        self.blocks = torch.nn.ModuleList(_Block(width, width) for _ in range(blocks))
        self.proj_out = torch.nn.Linear(width, 64)

    def forward(self, x):
        x = self.proj_in(x)
        for block in self.blocks:
            x = torch.nn.functional.gelu(block(x))
        return self.proj_out(x)


def test_resident_and_streamed_blocks_carry_the_same_dispatch_keys_under_inference_mode(
    monkeypatch,
):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA: diffusers stream group offload")
    pytest.importorskip("diffusers.hooks")
    for name in (
        op.ASYNC_PREFETCH_ENV,
        op.PREFETCH_DEPTH_ENV,
        "UNSLOTH_DIFFUSION_PARTIAL_RESIDENT",
    ):
        monkeypatch.delenv(name, raising = False)
    torch.manual_seed(0)
    net = _Net()
    ref = copy.deepcopy(net).cuda()
    pipe = types.SimpleNamespace(transformer = net, components = {"transformer": net})
    assert dm._apply_group_offload(pipe, "cuda", None, resident_transformer_mib = 13)
    assert op.module_prefetcher(net) is not None
    x = torch.randn(4, 64, device = "cuda")
    with torch.inference_mode():
        want = ref(x)
        for _ in range(3):
            _Block.seen.clear()
            got = net(x)
            torch.cuda.synchronize()
            assert torch.equal(got, want)
    resident = sum(1 for b in net.blocks if getattr(b.weight, "device", None).type == "cuda")
    streamed = len(net.blocks) - resident
    assert resident > 0 and streamed > 0, (resident, streamed)
    assert {dev for _, _, dev in _Block.seen} == {"cuda"}
    assert {inf for _, inf, _ in _Block.seen} == {False}
