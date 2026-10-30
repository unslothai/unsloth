# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the unlisted-arch dispatch of ``rotq_h16`` (``diffusion_convrot_quant.unlisted_device_ok`` and
``diffusion_int8_fused._rotated_act_quant``): every CUDA sm80+ arch without a dense ``rotq_i8`` tile gets the fused
rotation + act quant when it passes the probe and is not slower than the rotation GEMM + act quant. The gate tests mock
the capability and run anywhere; the kernel tests need NVIDIA CUDA, Triton and torchao."""

from __future__ import annotations

import pytest

from core.inference import diffusion_convrot_quant as cq
from core.inference import diffusion_int8_fused as f8

torch = pytest.importorskip("torch")

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or getattr(torch.version, "hip", None) is not None,
    reason = "NVIDIA CUDA required",
)

_ENVS = (
    cq.INT8_ROTQUANT_UNLISTED_ENV,
    cq.INT8_ROTQUANT_H16_ENV,
    cq.INT8_ROTQUANT_ENV,
    "UNSLOTH_DIFFUSION_INT8_GEMM",
)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    for name in _ENVS:
        monkeypatch.delenv(name, raising = False)
    cq._UNLISTED_DEVICE.clear()
    cq._UNLISTED_TIMING.clear()
    cq._UNLISTED_PROBE.clear()
    saved = dict(cq._H16_DEVICE)
    cq._H16_DEVICE.clear()  # a real-card verdict from an earlier test file must not leak in
    yield
    cq._UNLISTED_DEVICE.clear()
    cq._UNLISTED_TIMING.clear()
    cq._UNLISTED_PROBE.clear()
    cq._H16_DEVICE.clear()
    cq._H16_DEVICE.update(saved)


@pytest.mark.parametrize(
    "cap, on",
    [
        ((7, 5), False),  # T4: no bf16 tensor cores for rotq_h16's first dot
        ((8, 0), False),  # A100: dense rotq_i8 tile (and rotq_h16 on top of it)
        ((8, 6), True),  # RTX 30xx / A6000 / A10
        ((8, 7), True),  # Orin
        ((8, 9), False),  # L4 / RTX 40xx: tiled
        ((9, 0), True),  # H100
        ((10, 0), True),  # B200
        ((10, 3), True),  # B300
        ((12, 0), False),  # RTX PRO 6000 / RTX 50xx: tiled
        ((12, 1), True),  # DGX Spark
    ],
)
def test_every_sm80_plus_arch_without_a_tile_is_covered(cap, on):
    assert cq.unlisted_wanted(cap) is on
    if cap >= (8, 0):  # exactly one route to the fused rotation per sm80+ arch
        assert (cq.rotquant_config(cap) is not None) is not on


@pytest.mark.parametrize(
    "env",
    [
        "UNSLOTH_DIFFUSION_INT8_ROTQUANT_UNLISTED",
        "UNSLOTH_DIFFUSION_INT8_ROTQUANT_H16",
        "UNSLOTH_DIFFUSION_INT8_ROTQUANT",
        "UNSLOTH_DIFFUSION_INT8_GEMM",
    ],
)
def test_kill_switches(monkeypatch, env):
    monkeypatch.setenv(env, "0")
    for cap in ((8, 6), (9, 0), (10, 0), (12, 1)):
        assert not cq.unlisted_wanted(cap)


def test_forcing_the_dense_tile_takes_the_listed_route(monkeypatch):
    monkeypatch.setenv(
        cq.INT8_ROTQUANT_ENV, "1"
    )  # dense fallback tile on every unmeasured sm80+ part
    assert not cq.unlisted_wanted((8, 6)) and not cq.unlisted_wanted((10, 0))


def _fake_device(monkeypatch, cap):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: cap)
    monkeypatch.setattr(cq, "_triton_ok", lambda: True)
    monkeypatch.setattr(cq, "_kernels", lambda: object())
    monkeypatch.setattr(cq, "_rotq_op", lambda: "op")
    monkeypatch.setattr(cq, "is_rocm_torch", lambda t: False)
    monkeypatch.setattr(cq, "_ROTQ_HANDLE", None)


@pytest.mark.parametrize(
    "timing, mode, ok",
    [
        ({"h16": 10.0, "stock": 20.0}, None, True),
        ({"h16": 20.0, "stock": 20.0}, None, True),  # a tie is not slower
        (
            {"h16": 30.0, "stock": 20.0},
            None,
            False,
        ),  # slower than the two kernels it replaces: main's path
        ({"h16": 30.0, "stock": 20.0}, "1", True),  # forced: no latency check
    ],
)
def test_latency_gate(monkeypatch, timing, mode, ok):
    _fake_device(monkeypatch, (8, 6))
    if mode is not None:
        monkeypatch.setenv(cq.INT8_ROTQUANT_UNLISTED_ENV, mode)
    monkeypatch.setattr(cq, "_h16_probe", lambda index: "ieee")
    monkeypatch.setattr(cq, "_unlisted_latency", lambda index, tf32x3, stock = None: timing)
    assert cq.unlisted_device_ok(0) is ok
    assert (cq._H16_DEVICE.get(0) == "ieee") is ok
    assert (cq._ROTQ_HANDLE == "op") is ok


def test_latency_check_uses_the_probed_mode_and_the_callers_quant(monkeypatch):
    _fake_device(monkeypatch, (9, 0))
    monkeypatch.setattr(cq, "_h16_probe", lambda index: "tf32x3")
    seen = []
    monkeypatch.setattr(
        cq,
        "_unlisted_latency",
        lambda index, tf32x3, stock = None: seen.append((tf32x3, stock))
        or {"h16": 1.0, "stock": 2.0},
    )
    quant = object()
    assert (
        cq.unlisted_device_ok(0, quant)
        and seen == [(True, quant)]
        and cq._H16_DEVICE[0] == "tf32x3"
    )


def test_failed_probe_keeps_main(monkeypatch):
    _fake_device(monkeypatch, (10, 0))
    monkeypatch.setattr(cq, "_h16_probe", lambda index: False)
    monkeypatch.setattr(
        cq, "_unlisted_latency", lambda *a, **k: pytest.fail("timed after a failed probe")
    )
    assert cq.unlisted_device_ok(0) is False and 0 not in cq._H16_DEVICE


@pytest.mark.parametrize("cap", [(7, 5), (8, 0), (8, 9), (12, 0)])
def test_listed_archs_and_sm75_never_probe(monkeypatch, cap):
    _fake_device(monkeypatch, cap)
    monkeypatch.setattr(cq, "_h16_probe", lambda index: pytest.fail("probed"))
    assert cq.unlisted_device_ok(0) is False


def test_kill_switch_acts_on_a_cached_device(monkeypatch):
    _fake_device(monkeypatch, (8, 6))
    monkeypatch.setattr(cq, "_h16_probe", lambda index: "ieee")
    monkeypatch.setattr(
        cq, "_unlisted_latency", lambda index, tf32x3, stock = None: {"h16": 1.0, "stock": 2.0}
    )
    assert cq.unlisted_device_ok(0)
    monkeypatch.setenv(cq.INT8_ROTQUANT_UNLISTED_ENV, "0")
    assert not cq.unlisted_device_ok(0)
    monkeypatch.delenv(cq.INT8_ROTQUANT_UNLISTED_ENV)
    assert cq.unlisted_device_ok(0)


def _counting_slow_device(monkeypatch):
    _fake_device(monkeypatch, (8, 6))
    calls = {"probe": 0, "time": 0}

    def probe(index):
        calls["probe"] += 1
        return "ieee"

    def latency(
        index,
        tf32x3,
        stock = None,
    ):
        calls["time"] += 1
        return {"h16": 30.0, "stock": 20.0}  # slower: auto rejects

    monkeypatch.setattr(cq, "_h16_probe", probe)
    monkeypatch.setattr(cq, "_unlisted_latency", latency)
    return calls


def test_force_after_an_auto_rejection_enables_the_card(monkeypatch):
    calls = _counting_slow_device(monkeypatch)
    assert not cq.unlisted_device_ok(0) and 0 not in cq._H16_DEVICE
    monkeypatch.setenv(cq.INT8_ROTQUANT_UNLISTED_ENV, "1")
    assert cq.unlisted_device_ok(0) and cq._H16_DEVICE[0] == "ieee"
    assert calls == {"probe": 1, "time": 1}  # probed once per device, never timed under force


def test_auto_after_a_forced_load_runs_the_latency_check(monkeypatch):
    calls = _counting_slow_device(monkeypatch)
    monkeypatch.setenv(cq.INT8_ROTQUANT_UNLISTED_ENV, "1")
    assert cq.unlisted_device_ok(0) and cq._H16_DEVICE[0] == "ieee"
    monkeypatch.delenv(cq.INT8_ROTQUANT_UNLISTED_ENV)
    assert not cq.unlisted_device_ok(0) and 0 not in cq._H16_DEVICE
    assert not cq.unlisted_device_ok(0)  # repeat auto load: cached, not re-timed
    monkeypatch.setenv(cq.INT8_ROTQUANT_UNLISTED_ENV, "1")
    assert cq.unlisted_device_ok(0) and cq._H16_DEVICE[0] == "ieee"
    assert calls == {"probe": 1, "time": 1}


def test_arm_rotated_follows_the_gate(monkeypatch):
    monkeypatch.setattr(f8, "_RQ_DEVICES", frozenset())
    monkeypatch.setattr(f8, "_RQ_HANDLE", None)
    monkeypatch.setattr(f8, "_act_quant_device_ok", lambda index: False)
    monkeypatch.setattr(cq, "unlisted_device_ok", lambda index, quant = None: True)
    monkeypatch.setattr(cq, "_rotq_op", lambda: "op")
    assert f8.arm_rotated(3) and f8._RQ_DEVICES == frozenset({3}) and f8._RQ_HANDLE == "op"
    monkeypatch.setattr(cq, "unlisted_device_ok", lambda index, quant = None: False)
    assert not f8.arm_rotated(3) and f8._RQ_DEVICES == frozenset()


@pytest.mark.parametrize(
    "timing, armed_after",
    [({"h16": 10.0, "stock": 20.0}, "ieee"), ({"h16": 30.0, "stock": 20.0}, False)],
)
def test_dense_route_init_keeps_the_unlisted_verdict(monkeypatch, timing, armed_after):
    # Z-Image order: install_int8_fused arms rotq_h16, then install_int8_gemm's _swap_on runs rotquant_device_config,
    # which finds no dense tile on an unlisted arch and must not drop the verdict (no later arm_rotated restores it).
    _fake_device(monkeypatch, (8, 6))
    monkeypatch.setattr(cq, "_ROTQ_DEVICE", {})
    monkeypatch.setattr(cq, "_ROTQ_DEVICE_K", {})
    monkeypatch.setattr(cq, "_SWIGLU_DEVICE", {})
    monkeypatch.setattr(cq, "_h16_probe", lambda index: "ieee")
    monkeypatch.setattr(cq, "_unlisted_latency", lambda index, tf32x3, stock = None: timing)
    monkeypatch.setattr(cq, "_rotq_probe", lambda index, cfg: pytest.fail("no dense tile to probe"))
    monkeypatch.setattr(f8, "_RQ_DEVICES", frozenset())
    monkeypatch.setattr(f8, "_RQ_HANDLE", None)
    monkeypatch.setattr(f8, "_act_quant_device_ok", lambda index: False)
    assert f8.arm_rotated(0) is bool(armed_after)
    assert cq.rotquant_device_config(0) is None
    assert cq._H16_DEVICE.get(0, False) == armed_after
    assert (f8._RQ_DEVICES == frozenset({0})) is bool(armed_after)
    assert not cq._SWIGLU_DEVICE.get(0)


# CUDA


def _act(m, k, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator = g) * (torch.rand(1, k, generator = g) * 4)
    x[:, :5] *= 80
    x[min(1, m - 1)] = 0
    return x.to(torch.bfloat16).cuda()


@pytest.fixture
def armed(monkeypatch):
    """The unlisted route forced on whatever card runs the tests (the gate's arch check mocked)."""
    monkeypatch.setenv(cq.INT8_ROTQUANT_UNLISTED_ENV, "1")
    monkeypatch.setattr(cq, "unlisted_wanted", lambda cap, mode = None: True)
    monkeypatch.setattr(f8, "_RQ_DEVICES", frozenset())
    monkeypatch.setattr(f8, "_RQ_HANDLE", None)
    index = torch.cuda.current_device()
    if not f8.arm_rotated(index):
        pytest.skip("rotq_h16 probe refused this device")
    return index


@needs_cuda
@pytest.mark.parametrize("m, k", [(300, 256 * 3), (4352, 4096), (4352, 12288), (19, 256 * 128)])
def test_rotated_act_quant_runs_rotq_h16_once_armed(armed, m, k):
    x = _act(m, k, m + k)
    before = cq.h16_call_count()
    q, s = f8._rotated_act_quant(x, 256, None)
    assert cq.h16_call_count() == before + 1
    rq, rs = cq.rotquant_reference(x, 256, "v2")
    assert q.shape == rq.shape and s.shape == (m, 1)
    assert cq.codes_close(q, s.reshape(-1), rq, rs)


@needs_cuda
def test_rotated_act_quant_keeps_main_on_an_unarmed_card(monkeypatch):
    monkeypatch.setattr(f8, "_RQ_DEVICES", frozenset())
    called = []
    monkeypatch.setattr(f8, "_rotate", lambda x, g: called.append(g) or x)
    monkeypatch.setattr(f8, "_fast_act_quant", lambda x, w: ("q", "s"))
    assert f8._rotated_act_quant(_act(64, 512, 1), 256, None) == ("q", "s") and called == [256]


@needs_cuda
def test_latency_check_is_graph_replayed_and_repeatable():
    """Kernel time, not host launch time: two runs agree within 25% per arm."""
    index = torch.cuda.current_device()
    runs = [cq._unlisted_latency(index, False) for _ in range(2)]
    for name in ("h16", "stock"):
        a, b = runs[0][name], runs[1][name]
        assert 0 < a < float("inf") and abs(a - b) <= 0.25 * max(a, b)


@needs_cuda
def test_compiled_no_graph_break_no_recompile_and_cuda_graph(armed):
    """One graph, no recompile across token counts; the CUDA graph replay must equal eager."""
    import torch._dynamo
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()

    def fn(x):
        q, s = f8._rotated_act_quant(x, 256, None)
        return q.float() * s

    cfn = torch.compile(fn, fullgraph = True, dynamic = True)
    for m in (300, 517, 4352):
        x = _act(m, 4096, m)
        assert torch.equal(cfn(x), fn(x))
    assert counters["stats"].get("unique_graphs", 0) <= 1
    assert not counters["graph_break"]

    static = _act(4352, 4096, 9)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn(static)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(static)
    static.copy_(_act(4352, 4096, 10))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, fn(static))
