# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A lazily streamed H3 denoiser (pin budget refused the up-front pin) gets its block groups pinned in place, group by
group, while the budget holds; the rest keep diffusers' per-onload pin."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers.hooks")

import core.inference.diffusion_memory as dm  # noqa: E402
import core.inference.video_minimax_h3_residency as res  # noqa: E402
import core.inference.video_minimax_h3_te as te  # noqa: E402


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (
        res.H3_BLOCK_PIN_ENV,
        res.H3_STREAM_PREFETCH_ENV,
        "UNSLOTH_DIFFUSION_PIN_ARENA",
        "UNSLOTH_DIFFUSION_GROUP_OFFLOAD_PIN",
        "UNSLOTH_H3_TOP_GROUP_PIN",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(dm, "_pinned_memory_capped", lambda: False)


class _FakeGroup:
    def __init__(self, fail = False):
        self.low_cpu_mem_usage = True
        self.stream = object()
        self.offload_to_disk_path = None
        self.modules = [torch.nn.Linear(4, 4)]
        self.parameters = []
        self.buffers = []
        self.cpu_param_dict = {"lazy": True}
        self.fail = fail
        self.inits = 0

    def _init_cpu_param_dict(self):
        self.inits += 1
        if self.fail:
            raise RuntimeError("boom")
        return {"pinned": True}


def _fake(monkeypatch, groups, allow):
    net = torch.nn.Module()
    monkeypatch.setattr(res, "h3_offload_groups", lambda _t: (None, groups))
    calls = iter(allow)
    monkeypatch.setattr(te, "h3_te_pin_allowed", lambda *_a, **_k: next(calls, False))
    return net


def test_pins_every_group_the_budget_allows_and_marks_one_host_copy(monkeypatch):
    groups = [_FakeGroup() for _ in range(4)]
    net = _fake(monkeypatch, groups, [True] * 4)
    assert res.pin_streamed_block_groups(net) == (4, 4)
    assert all(not g.low_cpu_mem_usage and g.cpu_param_dict == {"pinned": True} for g in groups)
    assert getattr(net, "_unsloth_pin_arena_bytes", None) is not None


def test_budget_reread_per_group_stops_partway(monkeypatch):
    groups = [_FakeGroup() for _ in range(4)]
    net = _fake(monkeypatch, groups, [True, True, False])
    assert res.pin_streamed_block_groups(net) == (2, 4)
    assert [g.low_cpu_mem_usage for g in groups] == [False, False, True, True]
    assert groups[2].inits == 0 and groups[2].cpu_param_dict == {"lazy": True}
    # partly pinned: still lazy, so the caller keeps stream_lazy and the prefetch stays off
    assert getattr(net, "_unsloth_pin_arena_bytes", None) is None


def test_a_failing_group_is_restored_and_stops_the_pin(monkeypatch):
    groups = [_FakeGroup(), _FakeGroup(fail = True), _FakeGroup()]
    net = _fake(monkeypatch, groups, [True] * 3)
    assert res.pin_streamed_block_groups(net) == (1, 3)
    assert groups[1].low_cpu_mem_usage is True and groups[1].cpu_param_dict == {"lazy": True}
    assert groups[2].inits == 0


@pytest.mark.parametrize(
    "env, capped",
    [
        ({"UNSLOTH_H3_STREAM_BLOCK_PIN": "0"}, False),
        ({"UNSLOTH_DIFFUSION_PIN_ARENA": "0"}, False),
        ({}, True),
    ],
)
def test_kill_switch_no_arena_and_capped_hosts_leave_it_lazy(monkeypatch, env, capped):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(dm, "_pinned_memory_capped", lambda: capped)
    groups = [_FakeGroup() for _ in range(3)]
    net = _fake(monkeypatch, groups, [True] * 3)
    assert res.pin_streamed_block_groups(net)[0] == 0
    assert all(g.low_cpu_mem_usage and g.inits == 0 for g in groups)


def test_pin_refused_by_the_host_gate_pins_nothing(monkeypatch):
    """h3_te_pin_allowed is the gate: integrated GPUs (host_pinning_allowed) and the group-offload pin switch say no."""
    import core.inference.host_pinning as hp

    monkeypatch.setattr(hp, "host_pinning_allowed", lambda *_a, **_k: False)
    groups = [_FakeGroup() for _ in range(2)]
    net = torch.nn.Module()
    monkeypatch.setattr(res, "h3_offload_groups", lambda _t: (None, groups))
    assert res.pin_streamed_block_groups(net) == (0, 2)
    assert all(g.low_cpu_mem_usage for g in groups)


def test_already_pinned_groups_are_left_alone(monkeypatch):
    groups = [_FakeGroup() for _ in range(2)]
    for g in groups:
        g.low_cpu_mem_usage = False
    net = _fake(monkeypatch, groups, [True] * 2)
    assert res.pin_streamed_block_groups(net) == (0, 2)
    assert all(g.inits == 0 for g in groups)


class _Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.lin = torch.nn.Linear(width, width)

    def forward(self, x):
        return torch.nn.functional.gelu(self.lin(x))


class _DiT(torch.nn.Module):
    def __init__(
        self,
        width = 512,
        blocks = 6,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(64, width)
        self.transformer_blocks = torch.nn.ModuleList(_Block(width) for _ in range(blocks))
        self.proj_out = torch.nn.Linear(width, 64)

    def forward(self, x):
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.proj_out(x)


def test_lazy_stream_pinned_in_place_prefetches_and_is_bit_identical(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA: diffusers stream group offload")
    import inspect

    from diffusers.hooks import apply_group_offloading

    from core.inference.diffusion_offload_prefetch import module_prefetcher
    from core.inference.diffusion_prequant import _move_groups_outside_inference_mode

    monkeypatch.setattr(te, "h3_te_pin_allowed", lambda *_a, **_k: True)
    params = inspect.signature(apply_group_offloading).parameters
    if "record_stream" not in params or "low_cpu_mem_usage" not in params:
        pytest.skip("diffusers without record_stream / low_cpu_mem_usage")
    torch.manual_seed(0)
    net = _DiT().eval()
    ref = copy.deepcopy(net).cuda()
    apply_group_offloading(
        net,
        onload_device = torch.device("cuda"),
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
        use_stream = True,
        record_stream = True,
        non_blocking = True,
        low_cpu_mem_usage = True,
    )
    _move_groups_outside_inference_mode(net)
    assert res.pin_streamed_top_level_group(net)
    assert res.pin_streamed_block_groups(net) == (6, 6)
    # one host copy: the weights read the pinned slab views themselves
    for block in net.transformer_blocks:
        assert block.lin.weight.device.type == "cpu" and block.lin.weight.is_pinned()
    assert res.install_h3_stream_prefetch(net, "cuda") == 1 + len(net.transformer_blocks)
    assert module_prefetcher(net) is not None
    x = torch.randn(8, 64, device = "cuda")
    with torch.inference_mode():
        want = ref(x)
        for _ in range(3):
            assert torch.equal(net(x), want)
