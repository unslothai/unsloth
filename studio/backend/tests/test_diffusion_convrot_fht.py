# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``rotq_fht_i8`` (``diffusion_convrot_quant``): the fused fast-Hadamard ConvRot rotation + int8 act quant
that serves every arch the tiled ``rotq_i8`` does not cover. The gate tests mock the capability and run anywhere; the
kernel tests need NVIDIA CUDA, Triton and torchao."""

from __future__ import annotations

import pytest

from core.inference import diffusion_convrot_quant as cq
from core.inference import diffusion_int8_fused as f8

torch = pytest.importorskip("torch")

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or getattr(torch.version, "hip", None) is not None,
    reason = "NVIDIA CUDA required",
)

_ENVS = (cq.INT8_ROTQUANT_FHT_ENV, cq.INT8_ROTQUANT_ENV, "UNSLOTH_DIFFUSION_INT8_GEMM")


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    for name in _ENVS:
        monkeypatch.delenv(name, raising = False)
    cq._FHT_DEVICE.clear()
    cq._FHT_TIMING.clear()
    yield
    cq._FHT_DEVICE.clear()
    cq._FHT_TIMING.clear()


@pytest.mark.parametrize(
    "cap, on",
    [
        ((7, 0), False),  # V100: no int8 act-quant path worth a kernel
        ((7, 5), True),  # T4
        ((8, 0), False),  # A100: tiled rotq_i8 (bit-exact) stays
        ((8, 6), True),  # RTX 30xx / A6000 / A10
        ((8, 7), True),  # Orin
        ((8, 9), False),  # L4 / RTX 40xx: tiled rotq_i8
        ((9, 0), True),  # H100
        ((10, 0), True),  # B200
        ((10, 3), True),  # B300
        ((12, 0), False),  # RTX PRO 6000 / RTX 50xx: tiled rotq_i8
        ((12, 1), True),  # DGX Spark
    ],
)
def test_fht_covers_every_arch_the_tile_table_does_not(cap, on):
    assert cq.fht_wanted(cap) is on
    # exactly one fused rotation kernel per sm75+ arch
    if cap >= (7, 5):
        assert (cq.rotquant_config(cap) is not None) is not on


@pytest.mark.parametrize(
    "env, value",
    [
        (cq.INT8_ROTQUANT_FHT_ENV, "0"),
        (cq.INT8_ROTQUANT_ENV, "0"),
        ("UNSLOTH_DIFFUSION_INT8_GEMM", "0"),
    ],
)
def test_fht_kill_switches(monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    for cap in ((7, 5), (8, 6), (9, 0), (10, 0)):
        assert not cq.fht_wanted(cap)


def test_forcing_the_tiled_kernel_leaves_fht_to_sm75_only(monkeypatch):
    monkeypatch.setenv(cq.INT8_ROTQUANT_ENV, "1")  # tiled fallback on every unmeasured sm80+ part
    assert not cq.fht_wanted((8, 6)) and not cq.fht_wanted((10, 0))
    assert cq.fht_wanted((7, 5))


def _fake_device(monkeypatch, cap):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: cap)
    monkeypatch.setattr(cq, "_triton_ok", lambda: True)
    monkeypatch.setattr(cq, "_kernels", lambda: object())
    monkeypatch.setattr(cq, "_fht_op", lambda: "op")
    monkeypatch.setattr(cq, "is_rocm_torch", lambda t: False)


@pytest.mark.parametrize(
    "timing, mode, want",
    [
        (
            {True: 10.0, False: 12.0, "gemm": 20.0},
            None,
            True,
        ),  # Kronecker faster than the butterflies
        ({True: 15.0, False: 12.0, "gemm": 20.0}, None, False),  # butterflies faster
        (
            {True: 20.0, False: 25.0, "gemm": 20.0},
            None,
            True,
        ),  # tie with the rotation GEMM still passes
        (
            {True: 30.0, False: 31.0, "gemm": 20.0},
            None,
            None,
        ),  # both slower than the GEMM alone: stock
        ({True: 30.0, False: 31.0, "gemm": 20.0}, "1", True),  # forced: no latency check
    ],
)
def test_latency_gate_picks_the_faster_variant_or_stock(monkeypatch, timing, mode, want):
    _fake_device(monkeypatch, (8, 6))
    if mode is not None:
        monkeypatch.setenv(cq.INT8_ROTQUANT_FHT_ENV, mode)
    monkeypatch.setattr(cq, "_fht_probe", lambda index, kron: True)
    monkeypatch.setattr(
        cq,
        "_fht_latency",
        lambda index, variants, stock = None: {v: timing[v] for v in (*variants, "gemm")},
    )
    assert cq.fht_device_ok(0) is (want is not None)
    assert cq._FHT_DEVICE[0] is want


def test_sm75_only_tries_the_butterflies(monkeypatch):
    _fake_device(monkeypatch, (7, 5))
    tried = []
    monkeypatch.setattr(cq, "_fht_probe", lambda index, kron: tried.append(kron) or True)
    monkeypatch.setattr(
        cq, "_fht_latency", lambda index, variants, stock = None: {False: 1.0, "gemm": 50.0}
    )
    assert cq.fht_device_ok(0) and tried == [False] and cq._FHT_DEVICE[0] is False


def test_accuracy_probe_failure_keeps_stock(monkeypatch):
    _fake_device(monkeypatch, (9, 0))
    monkeypatch.setattr(cq, "_fht_probe", lambda index, kron: False)
    monkeypatch.setattr(
        cq,
        "_fht_latency",
        lambda index, variants, stock = None: pytest.fail("timed after a failed probe"),
    )
    assert cq.fht_device_ok(0) is False and cq._FHT_DEVICE[0] is None


def test_a_failed_probe_drops_only_that_variant(monkeypatch):
    _fake_device(monkeypatch, (9, 0))
    monkeypatch.setattr(cq, "_fht_probe", lambda index, kron: not kron)
    seen = []
    monkeypatch.setattr(
        cq,
        "_fht_latency",
        lambda index, variants, stock = None: seen.append(variants) or {False: 5.0, "gemm": 9.0},
    )
    assert cq.fht_device_ok(0) and seen == [(False,)] and cq._FHT_DEVICE[0] is False


def test_latency_gate_times_against_the_callers_act_quant(monkeypatch):
    _fake_device(monkeypatch, (8, 6))
    monkeypatch.setattr(cq, "_fht_probe", lambda index, kron: True)
    got = []
    monkeypatch.setattr(
        cq,
        "_fht_latency",
        lambda index, variants, stock = None: got.append(stock)
        or {True: 1.0, False: 2.0, "gemm": 3.0},
    )
    quant = object()
    assert cq.fht_device_ok(0, quant) and got == [quant]


@pytest.mark.parametrize("cap", [(8, 0), (8, 9), (12, 0)])
def test_listed_archs_never_probe_fht(monkeypatch, cap):
    """The archs with a tiled kernel keep their bit-exact path: no FHT probe, no FHT arming."""
    _fake_device(monkeypatch, cap)
    monkeypatch.setattr(
        cq, "_fht_probe", lambda index, kron: pytest.fail("probed on a listed arch")
    )
    assert cq.fht_device_ok(0) is False


def test_close_tolerates_one_code_step_only():
    q = torch.zeros(64, 256, dtype = torch.int8)
    s = torch.ones(64, dtype = torch.bfloat16)
    r = q.clone()
    r[0, 0] = 1
    assert cq._fht_close(q, s, r, s)
    r[0, 0] = 2
    assert not cq._fht_close(q, s, r, s)
    r = q.clone()
    r[:, :8] = 1  # 3% of codes off by one: more than the sum-order ties explain
    assert not cq._fht_close(q, s, r, s)
    assert not cq._fht_close(q, s, q, s.float())  # scale dtype must match torchao's
    assert not cq._fht_close(q, s, q, s * 1.1)


# ---- kernel (CUDA) ----


def _act(m, k, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator = g) * (torch.rand(1, k, generator = g) * 4)
    x[:, :5] *= 80
    x[min(1, m - 1)] = 0
    return x.to(torch.bfloat16).cuda()


def _variants():
    if not torch.cuda.is_available():
        return [False]
    return [True, False] if torch.cuda.get_device_capability() >= (8, 0) else [False]


@needs_cuda
@pytest.mark.parametrize(
    "m, k",
    [(17, 256), (300, 256 * 3), (4352, 4096), (257, 256 * 21), (4352, 12288), (19, 256 * 128)],
)
@pytest.mark.parametrize("kind", ["v1", "v2"])
@pytest.mark.parametrize("kron", _variants())
def test_fht_matches_rotation_then_torchao_quant(m, k, kind, kron):
    x = _act(m, k, m + k)
    q, s = cq._fht_launch(x, kind, kron = kron)
    rq, rs = cq.rotquant_reference(x, 256, kind)
    assert q.dtype == torch.int8 and s.dtype == rs.dtype and s.shape == (m,)
    assert cq._fht_close(q, s, rq, rs)


@needs_cuda
@pytest.mark.parametrize("k", [4096, 12288])
@pytest.mark.parametrize("kron", _variants())
def test_fht_is_as_close_to_the_exact_rotation_as_the_gemm(k, kron):
    """Against an fp64 rotation rounded once to bf16, the FHT's codes are no further off than the bf16 GEMM's."""
    from core.inference.diffusion_convrot import build_convrot_hadamard

    m = 1024
    x = _act(m, k, 7)
    h = build_convrot_hadamard(256, device = "cuda", dtype = torch.float64)
    exact = (x.double().reshape(m, -1, 256) @ h).reshape(m, k).to(torch.bfloat16)
    from torchao.quantization.granularity import PerRow
    from torchao.quantization.quantize_.workflows.int8.int8_tensor import Int8Tensor

    oq = Int8Tensor.from_hp(exact, PerRow()).qdata
    q, _ = cq._fht_launch(x, "v2", kron = kron)
    rq, _ = cq.rotquant_reference(x, 256, "v2")
    fht_off = int((q != oq).sum())
    gemm_off = int((rq != oq).sum())
    assert fht_off <= max(gemm_off, 16)


@needs_cuda
@pytest.mark.parametrize("kron", _variants())
def test_fht_gemm_output_accuracy_vs_bf16(kron):
    """End product: the int8 GEMM on FHT codes is as accurate against the bf16 GEMM as the stock rotation's codes."""
    from core.inference.diffusion_convrot import build_convrot_hadamard

    m, k, n = 1024, 4096, 512
    x = _act(m, k, 11)
    g = torch.Generator().manual_seed(12)
    w = (torch.randn(n, k, generator = g) * 0.02).cuda()
    h = build_convrot_hadamard(256, device = "cuda", dtype = torch.float32)
    wr = (w.reshape(n, -1, 256) @ h).reshape(n, k)
    ws = wr.abs().amax(1, keepdim = True) / 127
    wq = torch.clamp(torch.round(wr / ws), -128, 127).to(torch.int8)
    ref = x.float() @ w.t()

    def out(q, s):
        return torch._int_mm(q, wq.t()).float() * s.float().reshape(-1, 1) * ws.reshape(1, -1)

    y_fht = out(*cq._fht_launch(x, "v2", kron = kron))
    y_gemm = out(*cq.rotquant_reference(x, 256, "v2"))
    cos = torch.nn.functional.cosine_similarity
    c_fht = float(cos(y_fht.flatten(), ref.flatten(), dim = 0))
    c_gemm = float(cos(y_gemm.flatten(), ref.flatten(), dim = 0))
    assert c_fht >= c_gemm - 1e-6
    assert float((y_fht - y_gemm).abs().max()) <= float((y_gemm - ref).abs().max())


@needs_cuda
def test_fht_handles_an_empty_and_a_one_row_batch():
    for m in (0, 1):
        q, s = cq._fht_launch(torch.zeros(m, 512, dtype = torch.bfloat16, device = "cuda"), "v2")
        assert q.shape == (m, 512) and s.shape == (m,)
        if m:
            assert not q.any() and float(s[0]) > 0  # zero row: zero codes, eps scale


@needs_cuda
def test_fht_probe_accepts_this_device_when_forced(monkeypatch):
    monkeypatch.setenv(cq.INT8_ROTQUANT_FHT_ENV, "1")
    monkeypatch.setattr(cq, "fht_wanted", lambda cap, mode = None: True)
    assert cq.fht_device_ok(torch.cuda.current_device()) is True


@pytest.fixture
def armed(monkeypatch):
    monkeypatch.setenv(cq.INT8_ROTQUANT_FHT_ENV, "1")
    monkeypatch.setattr(cq, "fht_wanted", lambda cap, mode = None: True)
    index = torch.cuda.current_device()
    monkeypatch.setattr(f8, "_FHT_DEVICES", frozenset())
    monkeypatch.setattr(f8, "_FHT_HANDLE", None)
    assert f8.arm_rotated(index)
    return index


@needs_cuda
def test_rotated_act_quant_runs_the_kernel_once_armed(armed):
    x = _act(4352, 4096, 3)
    before = cq.fht_call_count()
    q, s = f8._rotated_act_quant(x, 256, None)
    assert cq.fht_call_count() == before + 1
    rq, rs = cq.rotquant_reference(x, 256, "v2")
    assert q.shape == rq.shape and s.shape == (4352, 1)
    assert cq._fht_close(q, s.reshape(-1), rq, rs)


@needs_cuda
def test_rotated_act_quant_keeps_stock_on_an_unarmed_card(monkeypatch):
    monkeypatch.setattr(f8, "_FHT_DEVICES", frozenset())
    called = []
    monkeypatch.setattr(f8, "_rotate", lambda x, g: called.append(g) or x)
    monkeypatch.setattr(f8, "_fast_act_quant", lambda x, w: ("q", "s"))
    assert f8._rotated_act_quant(_act(64, 512, 1), 256, None) == ("q", "s") and called == [256]


@needs_cuda
def test_compiled_no_graph_break_no_recompile_and_cuda_graph(armed):
    """One graph, no recompile when the token count changes (a new prompt length), and the op captures in a CUDA
    graph with replay equal to eager."""
    import torch._dynamo

    torch._dynamo.reset()
    from torch._dynamo.utils import counters

    counters.clear()

    def fn(x):
        q, s = f8._rotated_act_quant(x, 256, None)
        return q.float() * s

    cfn = torch.compile(fn, fullgraph = True, dynamic = True)
    for m in (300, 517, 4352):
        x = _act(m, 4096, m)
        assert torch.equal(cfn(x), fn(x))
    assert sum(counters["stats"].get(k, 0) for k in ("unique_graphs",)) <= 1
    assert not counters["graph_break"]

    x = _act(4352, 4096, 9)
    static = x.clone()
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
