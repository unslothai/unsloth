# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Background in-place block pin for a lazily streamed H3 denoiser: per-group waits, budget, bit-identical output."""

from __future__ import annotations

import copy
import threading

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
        res.H3_BLOCK_BACKGROUND_PIN_ENV,
        res.H3_STREAM_PREFETCH_ENV,
        dm.BACKGROUND_PIN_ENV,
        dm.FAST_PIN_ENV,
        "UNSLOTH_DIFFUSION_PIN_ARENA",
        "UNSLOTH_DIFFUSION_GROUP_OFFLOAD_PIN",
        "UNSLOTH_H3_TOP_GROUP_PIN",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(dm, "_pinned_memory_capped", lambda: False)


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA: registered pins and diffusers stream group offload")


class _Pipe:
    pass


class _Block(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.lin = torch.nn.Linear(width, width)

    def forward(self, x):
        return torch.nn.functional.gelu(self.lin(x))


class _DiT(torch.nn.Module):
    def __init__(
        self,
        width = 256,
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


def _lazy_stream(net):
    import inspect

    from diffusers.hooks import apply_group_offloading

    from core.inference.diffusion_prequant import _move_groups_outside_inference_mode

    params = inspect.signature(apply_group_offloading).parameters
    if "record_stream" not in params or "low_cpu_mem_usage" not in params:
        pytest.skip("diffusers without record_stream / low_cpu_mem_usage")
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
    return net


def _host_ptrs(net):
    return [b.lin.weight.data_ptr() for b in net.transformer_blocks]


def test_pin_host_tensor_keeps_bits_and_rebuilds_wrapper_subclasses():
    _cuda()
    src = torch.randn(33, 17).to(torch.bfloat16)
    out = res._pin_host_tensor(src)
    assert out.is_pinned() and torch.equal(out, src) and out.data_ptr() != src.data_ptr()
    torchao = pytest.importorskip("torchao")
    from torchao.quantization import quantize_

    try:
        from torchao.quantization import Int8WeightOnlyConfig as Cfg
    except ImportError:
        pytest.skip(f"torchao {torchao.__version__} has no Int8WeightOnlyConfig")
    lin = torch.nn.Linear(64, 64).to(torch.bfloat16)
    quantize_(lin, Cfg())
    weight = lin.weight
    pinned = res._pin_host_tensor(weight)
    assert type(pinned) is type(weight)
    leaves, want = res._host_leaves(pinned), res._host_leaves(weight)
    assert leaves and len(leaves) == len(want)
    for a, b in zip(leaves, want):
        assert a.is_pinned() and torch.equal(a, b)


@pytest.mark.parametrize("prefetch", [True, False])
def test_background_pin_streams_bit_identical_and_frees_the_pageable_copy(monkeypatch, prefetch):
    _cuda()
    from core.inference.diffusion_offload_prefetch import module_prefetcher

    monkeypatch.setattr(te, "h3_te_pin_allowed", lambda *_a, **_k: True)
    if not prefetch:
        monkeypatch.setenv(res.H3_STREAM_PREFETCH_ENV, "0")
    torch.manual_seed(0)
    net = _DiT().eval()
    ref = copy.deepcopy(net).cuda()
    _lazy_stream(net)
    pageable = _host_ptrs(net)
    assert res.pin_streamed_top_level_group(net)
    pipe = _Pipe()
    gate = threading.Event()
    real = res._pin_host_tensor
    # hold the first copy so the install below runs while the pin is still in flight
    monkeypatch.setattr(res, "_pin_host_tensor", lambda src: (gate.wait(5), real(src))[1])
    pinner = res.defer_streamed_block_pins(pipe, net, torch.device("cuda"), logger = None)
    assert pinner is not None and pinner in pipe._unsloth_background_pins
    _top, blocks = res.h3_offload_groups(net)
    assert all(not g.low_cpu_mem_usage for g in blocks)
    covered = res.install_h3_stream_prefetch(net, "cuda")
    assert (covered == 1 + len(blocks)) is prefetch
    assert (module_prefetcher(net) is not None) is prefetch
    gate.set()
    if not covered:
        pinner.join()
    dm.finish_background_pins(pipe)
    assert pinner.groups_pinned == len(blocks) and pinner.stopped_at is None
    for block, old in zip(net.transformer_blocks, pageable):
        weight = block.lin.weight
        assert weight.device.type == "cpu" and weight.is_pinned() and weight.data_ptr() != old
    x = torch.randn(8, 64, device = "cuda")
    with torch.inference_mode():
        want = ref(x)
        for _ in range(3):
            assert torch.equal(net(x), want)
    dm.stop_background_pins(pipe)


def test_an_onload_waits_for_its_groups_pin(monkeypatch):
    _cuda()
    monkeypatch.setattr(te, "h3_te_pin_allowed", lambda *_a, **_k: True)
    torch.manual_seed(0)
    net = _DiT(blocks = 4).eval()
    ref = copy.deepcopy(net).cuda()
    _lazy_stream(net)
    assert res.pin_streamed_top_level_group(net)
    gate = threading.Event()
    real = res._pin_host_tensor
    monkeypatch.setattr(res, "_pin_host_tensor", lambda src: (gate.wait(5), real(src))[1])
    pipe = _Pipe()
    pinner = res.defer_streamed_block_pins(pipe, net, torch.device("cuda"))
    assert res.install_h3_stream_prefetch(net, "cuda") > 0
    threading.Timer(0.3, gate.set).start()
    x = torch.randn(8, 64, device = "cuda")
    # no finish_background_pins: the forward itself must wait per group and still read the right bits
    with torch.inference_mode():
        assert torch.equal(net(x), ref(x))
    assert pinner.waited_s > 0
    pinner.join()
    assert pinner.groups_pinned == 4


def test_budget_reread_per_group_stops_and_leaves_the_rest_pageable(monkeypatch):
    _cuda()
    torch.manual_seed(0)
    net = _DiT(blocks = 5).eval()
    ref = copy.deepcopy(net).cuda()
    _lazy_stream(net)
    monkeypatch.setattr(te, "h3_te_pin_allowed", lambda *_a, **_k: True)
    assert res.pin_streamed_top_level_group(net)
    answers = iter([True, True, True, False])
    monkeypatch.setattr(te, "h3_te_pin_allowed", lambda *_a, **_k: next(answers, False))
    pipe = _Pipe()
    pinner = res.defer_streamed_block_pins(pipe, net, torch.device("cuda"))
    assert res.install_h3_stream_prefetch(net, "cuda") > 0
    dm.finish_background_pins(pipe)
    assert pinner.groups_pinned == 2 and pinner.stopped_at == 2
    pinned = [b.lin.weight.is_pinned() for b in net.transformer_blocks]
    assert pinned == [True, True, False, False, False]
    x = torch.randn(8, 64, device = "cuda")
    with torch.inference_mode():
        want = ref(x)
        for _ in range(2):
            assert torch.equal(net(x), want)


class _FakeGroup:
    def __init__(self, lazy = True):
        self.low_cpu_mem_usage = lazy
        self.stream = object()
        self.offload_to_disk_path = None
        lin = torch.nn.Linear(4, 4)
        self.modules = [lin]
        self.parameters = []
        self.buffers = []
        self.cpu_param_dict = {p: p.data for p in lin.parameters()}


@pytest.mark.parametrize(
    "env, value",
    [
        (res.H3_BLOCK_PIN_ENV, "0"),
        (res.H3_BLOCK_BACKGROUND_PIN_ENV, "0"),
        (dm.BACKGROUND_PIN_ENV, "0"),
        (dm.FAST_PIN_ENV, "0"),
        ("UNSLOTH_DIFFUSION_PIN_ARENA", "0"),
    ],
)
def test_kill_switches_fall_back_to_the_up_front_pin(monkeypatch, env, value):
    groups = [_FakeGroup() for _ in range(3)]
    monkeypatch.setattr(res, "h3_offload_groups", lambda _t: (None, groups))
    monkeypatch.setattr(te, "h3_te_pin_allowed", lambda *_a, **_k: True)
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda *_a, **_k: env != dm.FAST_PIN_ENV)
    monkeypatch.setattr(dm, "host_pinning_allowed", lambda *_a, **_k: True)
    monkeypatch.setenv(env, value)
    pipe = _Pipe()
    assert res.defer_streamed_block_pins(pipe, torch.nn.Module(), "cuda") is None
    assert all(g.low_cpu_mem_usage for g in groups)
    assert not getattr(pipe, "_unsloth_background_pins", None)


@pytest.mark.parametrize("case", ["capped", "refused", "mixed", "none"])
def test_hosts_and_layouts_the_up_front_pin_owns(monkeypatch, case):
    groups = [_FakeGroup(lazy = case != "mixed" or i != 1) for i in range(3)]
    if case == "none":
        groups = []
    monkeypatch.setattr(res, "h3_offload_groups", lambda _t: (None, groups))
    monkeypatch.setattr(te, "h3_te_pin_allowed", lambda *_a, **_k: case != "refused")
    monkeypatch.setattr(dm, "_pinned_memory_capped", lambda: case == "capped")
    monkeypatch.setattr(dm, "_fast_pin_supported", lambda *_a, **_k: True)
    monkeypatch.setattr(dm, "host_pinning_allowed", lambda *_a, **_k: True)
    pipe = _Pipe()
    assert res.defer_streamed_block_pins(pipe, torch.nn.Module(), "cuda") is None
    assert [g.low_cpu_mem_usage for g in groups] == [
        case != "mixed" or i != 1 for i in range(len(groups))
    ]
