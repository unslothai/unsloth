# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests for the NVIDIA native W8A8 fused rotation + act quant (diffusion_native_w8a8_cuda): the reference equals
the Linear's own rotation + quant, the routed forward equals the stock traced forward, routing gates (compiled only,
bf16, CUDA, probed card, no input grad), kill switches, and the install point ahead of the compile."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import inspect
import types

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason = "needs torch")


@pytest.fixture
def cr(monkeypatch):
    from core.inference import diffusion_native_w8a8_cuda as mod

    monkeypatch.delenv("UNSLOTH_NATIVE_INT8_ROTQ", raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_ROTQUANT", raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_GEMM", raising = False)
    saved = {
        k: (dict(v) if isinstance(v, dict) else (set(v) if isinstance(v, set) else v))
        for k, v in mod._STATE.items()
    }
    yield mod
    mod._STATE.clear()
    mod._STATE.update(saved)


def _linear(
    i = 512,
    o = 96,
    bias = True,
    rot_group = 256,
    act_int8 = True,
):
    import torch

    from core.inference.diffusion_native_quant import native_linear_class

    torch.manual_seed(0)
    lin = torch.nn.Linear(i, o, bias = bias).to(torch.bfloat16)
    return native_linear_class()(lin, "int8", act_int8 = act_int8, rot_group = rot_group)


def _x(rows = 40, k = 512):
    import torch

    g = torch.Generator().manual_seed(1)
    x = (torch.randn(rows, k, generator = g) * 3).to(torch.bfloat16)
    x[3] = 0
    x[5, 7] = 2e3
    return x


def test_reference_equals_the_linears_own_rotation_and_quant(cr):
    m = _linear()
    x = _x()
    got = {}
    orig = m._int_mm_from_q

    def grab(xq, xs, dtype):
        got["q"], got["s"] = xq, xs
        return orig(xq, xs, dtype)

    m._int_mm_from_q = grab
    m._int_mm_rows(m._rotate(x))
    rq, rs = cr.reference(x, 256)
    import torch

    assert torch.equal(got["q"], rq) and torch.equal(got["s"], rs)
    assert rq.dtype == torch.int8 and rs.dtype == torch.float32 and rs.shape == (x.shape[0], 1)


@pytest.mark.parametrize("bias", [True, False])
def test_routed_forward_equals_stock_traced_maths(cr, bias):
    import torch

    m = _linear(bias = bias)
    x = _x().reshape(2, 20, 512)
    cr._STATE["op"] = lambda a, g: cr.reference(a, g)
    stock = m._forward_int_mm(m._rotate(x))
    routed = m._forward_cuda_rotq(x)
    assert routed.dtype == x.dtype and routed.shape == (2, 20, 96)
    assert torch.equal(routed, stock)


def test_forward_dispatches_to_the_route_only_when_the_gate_opens(cr, monkeypatch):
    import torch

    m = _linear()
    x = _x()
    calls = []
    cr._STATE["op"] = lambda a, g: calls.append(a.shape) or cr.reference(a, g)
    eager = m(x)
    assert not calls  # eager (and CPU) never takes it
    monkeypatch.setattr(type(m), "_cuda_rotq_on", lambda self, x: True)
    routed = m(x)
    assert calls == [torch.Size([40, 512])]
    assert torch.equal(routed, m._forward_int_mm(m._rotate(x)))
    assert eager.shape == routed.shape


def _fake_x(**kw):
    import torch

    base = dict(
        is_cuda = True,
        dtype = torch.bfloat16,
        requires_grad = False,
        device = types.SimpleNamespace(index = 0),
    )
    base.update(kw)
    ns = types.SimpleNamespace(**base)
    ns.numel = lambda: 64 * 512
    return ns


def test_gate(cr, monkeypatch):
    import torch

    m = _linear()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    cr._STATE["op"] = object()
    cr._STATE["on"] = {0: True}
    assert m._cuda_rotq_on(_fake_x())
    # card not probed, other card, no op handle
    assert not m._cuda_rotq_on(_fake_x(device = types.SimpleNamespace(index = 1)))
    cr._STATE["op"] = None
    assert not m._cuda_rotq_on(_fake_x())
    cr._STATE["op"] = object()
    # fp16 / fp32 activations, CPU, an input that needs a gradient
    assert not m._cuda_rotq_on(_fake_x(dtype = torch.float16))
    assert not m._cuda_rotq_on(_fake_x(dtype = torch.float32))
    assert not m._cuda_rotq_on(_fake_x(is_cuda = False))
    with torch.enable_grad():
        assert not m._cuda_rotq_on(_fake_x(requires_grad = True))
    # too few rows for _int_mm
    small = _fake_x()
    small.numel = lambda: 8 * 512
    assert not m._cuda_rotq_on(small)
    # eager: never
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    assert not m._cuda_rotq_on(_fake_x())


def test_gate_refuses_unrotated_and_off_grid_layers(cr, monkeypatch):
    import torch

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    cr._STATE["op"] = object()
    cr._STATE["on"] = {0: True}
    assert not _linear(rot_group = 0)._cuda_rotq_on(_fake_x())
    assert cr.shape_ok(4096, 256) and cr.shape_ok(16384, 256) and cr.shape_ok(256 * 128, 256)
    assert (
        not cr.shape_ok(256 * 129, 256) and not cr.shape_ok(300, 256) and not cr.shape_ok(4096, 64)
    )
    assert not cr.shape_ok(4096, 0)


@pytest.mark.parametrize(
    "env",
    ["UNSLOTH_NATIVE_INT8_ROTQ", "UNSLOTH_DIFFUSION_INT8_ROTQUANT", "UNSLOTH_DIFFUSION_INT8_GEMM"],
)
def test_kill_switches(cr, monkeypatch, env):
    monkeypatch.setenv(env, "0")
    assert cr.disabled()
    assert cr.prepare() is False
    assert cr.install(_wrap(_linear())) == 0


def _wrap(*mods):
    import torch
    return torch.nn.Sequential(*mods)


def test_candidates_and_install_without_cuda(cr, monkeypatch):
    import torch

    t = _wrap(_linear(), _linear(rot_group = 0), _linear(act_int8 = False), torch.nn.Linear(8, 8))
    assert cr.candidates(t) == 1
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cr.install(t) == 0
    assert cr.install(None) == 0


def test_install_probes_the_compute_card_and_reports_the_count(cr, monkeypatch):
    seen = []
    monkeypatch.setattr(cr, "prepare", lambda device = None: seen.append(device) or True)
    monkeypatch.setattr(cr, "_install_share_pass", lambda logger = None: seen.append("share") or True)
    t = _wrap(_linear(), _linear())
    assert cr.install(t, device = "cuda:1") == 2
    assert seen == ["cuda:1", "share"]


def test_launch_falls_back_to_the_reference_off_card(cr):
    import torch

    x = _x()
    q, s = cr._launch(x, 256)  # CPU tensor, no probed tile: the eager maths
    rq, rs = cr.reference(x, 256)
    assert torch.equal(q, rq) and torch.equal(s, rs)


def test_native_kernel_mode_is_passed_to_the_launch(cr, monkeypatch):
    import torch

    from core.inference import diffusion_convrot_quant as cq

    qmin, qmax, rcp, eps = cq._ROTQ_QPARAMS["native"]
    assert (qmin, qmax, eps) == (-127.0, 127.0, 1e-12)
    # the fp32 reciprocal ATen's CUDA div by a scalar multiplies with
    assert rcp == (torch.tensor(1.0) / torch.tensor(127.0)).item()
    assert (
        "NATIVE" in inspect.signature(cq._kernels().rotq_i8.fn).parameters
        if cq._kernels()
        else True
    )
    seen = {}

    class _K:
        def __getitem__(self, grid):
            def run(*args, **kw):
                seen.update(kw)

            return run

    monkeypatch.setattr(
        cq,
        "_kernels",
        lambda: types.SimpleNamespace(
            rotq_i8 = _K(),
            triton = types.SimpleNamespace(
                cdiv = lambda a, b: -(-a // b), next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
            ),
        ),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda d: contextlib.nullcontext())
    q, s = cq._rotq_launch(_x(), 256, "native", (128, 32, 8, 3))
    assert seen["NATIVE"] is True and seen["FP32_SCALE"] is True and seen["EPS"] == 1e-12
    assert s.dtype == torch.float32
    seen.clear()
    cq._rotq_launch(_x(), 256, "v1", (128, 32, 8, 3))
    assert seen["NATIVE"] is False


def test_speed_installs_the_route_before_the_compile():
    from core.inference import diffusion_speed

    src = inspect.getsource(diffusion_speed._compile_repeated_blocks)
    tree = ast.parse(src.lstrip())
    order = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name in ("install_native_rotq", "compile_repeated_blocks"):
                order.append((node.lineno, name))
    order.sort()
    names = [n for _, n in order]
    assert "install_native_rotq" in names and "compile_repeated_blocks" in names
    assert names.index("install_native_rotq") < names.index("compile_repeated_blocks")


def _rotq_graph(cr):
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx

    op = cr._op()

    def f(h):
        a = op(h.view(-1, 512), 256)[0]  # Q
        b = op(h.view(-1, 512), 256)[0]  # K: same input
        c = op(h.reshape(-1, 512), 256)[1]  # V: same input, reshape
        d = op(h.view(-1, 512) * 2, 256)[0]  # another tensor
        e = op(h.view(-1, 256), 256)[0]  # another view shape
        return a, b, c, d, e

    h = _x(40, 512).reshape(2, 20, 512)
    return f, make_fx(f)(h), h


def _rotq_calls(g):
    return sum(1 for n in g.graph.nodes if "native_cuda_rotq" in str(n.target))


def test_share_pass_merges_calls_on_one_input_only(cr):
    import torch

    f, g, h = _rotq_graph(cr)
    assert _rotq_calls(g) == 5
    assert cr.share_rotq(g.graph) == 2
    g.recompile()
    assert _rotq_calls(g) == 3
    for got, ref in zip(g(h), f(h)):
        assert torch.equal(got, ref)


def test_share_pass_install_respects_switch_and_an_existing_pass(cr, monkeypatch):
    from core.inference import diffusion_compile_config as compile_config

    knobs = {}
    monkeypatch.setattr(compile_config, "get_knob", lambda m, a, d = None: knobs.get((m, a), d))
    monkeypatch.setattr(
        compile_config, "set_knob", lambda m, a, v: knobs.__setitem__((m, a), v) or True
    )
    monkeypatch.setenv("UNSLOTH_NATIVE_INT8_ROTQ_SHARE", "0")
    assert cr._install_share_pass() is False and not knobs
    monkeypatch.delenv("UNSLOTH_NATIVE_INT8_ROTQ_SHARE")
    assert cr._install_share_pass() is True
    assert isinstance(
        knobs[("torch._inductor.config", "post_grad_custom_pre_pass")], cr.ShareRotqPass
    )
    assert cr._install_share_pass() is True  # idempotent
    other = object()
    knobs[("torch._inductor.config", "post_grad_custom_pre_pass")] = other
    assert cr._install_share_pass() is False  # never replaces another pass
    assert knobs[("torch._inductor.config", "post_grad_custom_pre_pass")] is other
    assert isinstance(cr.ShareRotqPass().uuid(), str)


def test_share_pass_pickles():
    import pickle

    from core.inference import diffusion_native_w8a8_cuda as mod

    p = pickle.loads(pickle.dumps(mod.ShareRotqPass()))
    assert p.uuid() == mod.ShareRotqPass().uuid()
