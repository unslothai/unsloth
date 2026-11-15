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
    assert not m._cuda_rotq_on(_fake_x(device = types.SimpleNamespace(index = 1)))
    cr._STATE["op"] = None
    assert not m._cuda_rotq_on(_fake_x())
    cr._STATE["op"] = object()
    assert not m._cuda_rotq_on(_fake_x(dtype = torch.float16))
    assert not m._cuda_rotq_on(_fake_x(dtype = torch.float32))
    assert not m._cuda_rotq_on(_fake_x(is_cuda = False))
    with torch.enable_grad():
        assert not m._cuda_rotq_on(_fake_x(requires_grad = True))
    small = _fake_x()
    small.numel = lambda: 8 * 512
    assert not m._cuda_rotq_on(small)
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
    # built first: on a live CUDA context the Linears probe the current card themselves at construction
    t = _wrap(_linear(), _linear())
    seen = []
    monkeypatch.setattr(cr, "prepare", lambda device = None: seen.append(device) or True)
    monkeypatch.setattr(cr, "_install_share_pass", lambda logger = None: seen.append("share") or True)
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
    for env in cr.FOLD_ENVS.values():
        monkeypatch.setenv(env, "0")
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


@pytest.mark.parametrize("bias", [True, False])
def test_gemm_reference_equals_the_linears_epilogue(cr, bias):
    import torch

    m = _linear(bias = bias)
    x = _x()
    xq, xs = cr.reference(x, 256)
    ref = cr.reference_gemm(xq, m.weight_q, xs, m.weight_scale.view(torch.float32), m.bias)
    assert ref.dtype == torch.bfloat16
    assert torch.equal(ref, m._int_mm_from_q(xq, xs, x.dtype).to(x.dtype))


def test_routed_forward_takes_the_fused_gemm_when_armed(cr):
    import torch

    m = _linear()
    x = _x()
    cr._STATE["op"] = lambda a, g: cr.reference(a, g)
    seen = []
    cr._STATE["gemm_op"] = lambda *a: seen.append(a[0].shape) or cr.reference_gemm(*a)
    cr._STATE["gemm"] = {None: True}  # CPU tensors: device index None
    out = m._forward_cuda_rotq(x)
    assert seen == [torch.Size([40, 512])]
    assert torch.equal(out, m._forward_int_mm(m._rotate(x)))
    cr._STATE["gemm"] = {}
    seen.clear()
    assert torch.equal(m._forward_cuda_rotq(x), out) and not seen


def test_gemm_switch_and_shape_gate(cr, monkeypatch):
    assert cr.gemm_shape_ok(4096, 4096) and cr.gemm_shape_ok(16384, 4096)
    assert not cr.gemm_shape_ok(4096 + 32, 4096) and not cr.gemm_shape_ok(4096, 4096 + 8)
    monkeypatch.setenv("UNSLOTH_NATIVE_INT8_GEMM", "0")
    assert cr.gemm_disabled() and cr.prepare_gemm(0) is False
    monkeypatch.delenv("UNSLOTH_NATIVE_INT8_GEMM")
    monkeypatch.setenv("UNSLOTH_NATIVE_INT8_ROTQ", "0")
    assert cr.gemm_disabled()


def test_gemm_launch_keeps_cublas_maths_off_card(cr):
    import torch

    m = _linear()
    xq, xs = cr.reference(_x(), 256)
    ws = m.weight_scale.view(torch.float32)
    assert torch.equal(
        cr._gemm_launch(xq, m.weight_q, xs, ws, m.bias),
        cr.reference_gemm(xq, m.weight_q, xs, ws, m.bias),
    )


def test_gemm_kernel_native_flag(monkeypatch):
    import torch

    from core.inference import diffusion_int8_gemm as g8

    seen = {}

    class _K:
        def __getitem__(self, grid):
            return lambda *a, **kw: seen.update(kw)

    monkeypatch.setattr(g8, "_kernels", lambda: types.SimpleNamespace(i8mm_dq = _K()))
    monkeypatch.setattr(
        g8, "triton", types.SimpleNamespace(cdiv = lambda a, b: -(-a // b)), raising = False
    )
    monkeypatch.setattr(torch.cuda, "device", lambda d: contextlib.nullcontext())
    a = torch.zeros(32, 64, dtype = torch.int8)
    w = torch.zeros(16, 64, dtype = torch.int8)
    g8._launch(a, w, torch.ones(32), torch.ones(16), None, (128, 128, 64, 8, 4, 3), native = True)
    assert seen["NATIVE"] is True
    g8._launch(a, w, torch.ones(32), torch.ones(16), None, (128, 128, 64, 8, 4, 3))
    assert seen["NATIVE"] is False


def _fold_block(cr, tracing_mode = "fake"):
    """A post-grad-like aten graph (make_fx with Inductor's decompositions) of an LTX-2.5 style sublayer pair: RMSNorm +
    AdaLN into a quant, the fused GEMM, tanh GELU into a quant, the GEMM, gated residual."""
    import torch
    from torch._inductor.decomposition import select_decomp_table
    from torch.fx.experimental.proxy_tensor import make_fx

    cr._op(), cr._gemm_op(), cr._norm_op(), cr._gemm_res_op(), cr._gemm_gelu_op()
    rotq = torch.ops.unsloth_studio.native_cuda_rotq
    gemm = torch.ops.unsloth_studio.native_cuda_i8mm

    def block(x, table, temb, wq1, ws1, b1, wq2, ws2):
        p = table[None, None] + temb.reshape(temb.shape[0], temb.shape[1], 3, -1)
        sh, sc, gt = p.unbind(2)
        n = (x * torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim = True) + 1e-6)).to(x.dtype)
        h = n * (1 + sc) + sh
        q, s = rotq(h.reshape(-1, x.shape[-1]), 256)
        y = gemm(q, wq1, s, ws1, b1)
        a = torch.nn.functional.gelu(y.view(x.shape[0], x.shape[1], -1), approximate = "tanh")
        q2, s2 = rotq(a.reshape(-1, a.shape[-1]), 256)
        y2 = gemm(q2, wq2, s2, ws2, None)
        return x + y2.view(x.shape) * gt

    g = torch.Generator().manual_seed(0)
    b, t, k, f = 2, 24, 512, 1024
    args = (
        (torch.randn(b, t, k, generator = g) * 2).bfloat16(),
        (torch.randn(3, k, generator = g) * 0.1).bfloat16(),
        (torch.randn(b, t, 3 * k, generator = g) * 0.1).bfloat16(),
        torch.randint(-127, 128, (f, k), generator = g, dtype = torch.int8),
        torch.rand(f, generator = g) * 1e-3,
        torch.randn(f, generator = g).bfloat16(),
        torch.randint(-127, 128, (k, f), generator = g, dtype = torch.int8),
        torch.rand(k, generator = g) * 1e-3,
    )
    decomp = select_decomp_table()
    gm = make_fx(block, tracing_mode = tracing_mode, decomposition_table = decomp)(*args)
    ref = make_fx(block, decomposition_table = decomp)(*args)(*args)
    return gm, args, ref


def _targets(gm):
    return [str(n.target) for n in gm.graph.nodes if "unsloth_studio" in str(n.target)]


def _arm_all(cr):
    cr._STATE["pro"][None] = {"norm": True}
    cr._STATE["res"][None] = True
    cr._STATE["act"][None] = True


def test_folds_match_and_keep_the_maths(cr):
    """All three folds match the decomposed graph (per-token modulation, batch 2): the GELU and the gated residual land
    in the GEMMs' epilogues, the norm in front of the first quant, the replaced GEMMs and quant leave the graph, and
    the output stays within int8 code-flip noise of the unfolded graph (the folds read the materialised bf16
    modulation and gate, as eager does)."""
    _arm_all(cr)
    gm, args, ref = _fold_block(cr)
    assert cr.fold_gelus(gm.graph) == 1
    assert cr.fold_norms(gm.graph) == 1
    assert cr.fold_residuals(gm.graph) == 1
    gm.graph.eliminate_dead_code()
    gm.recompile()
    names = _targets(gm)
    assert sum(n.endswith("native_cuda_rotq_norm.default") for n in names) == 1
    assert (
        sum(n.endswith("native_cuda_rotq.default") for n in names) == 1
    )  # the GELU output's quant
    assert sum(n.endswith("native_cuda_i8mm_gelu.default") for n in names) == 1
    assert sum(n.endswith("native_cuda_i8mm_res.default") for n in names) == 1
    assert not any(n.endswith("native_cuda_i8mm.default") for n in names)
    assert not any("tanh" in str(n.target) for n in gm.graph.nodes)
    out = gm(*args)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    # a bf16 step in the modulation / gate can flip an int8 code in the next quant: compare the block, not values
    rel = float((out.float() - ref.float()).norm() / ref.float().norm())
    assert rel < 5e-3, rel


def test_folds_need_their_probe_and_respect_other_users(cr):
    import torch

    cr._STATE["pro"][None] = {"norm": False}
    cr._STATE["res"][None] = False
    cr._STATE["act"][None] = False
    gm, _args, _ref = _fold_block(cr)
    assert cr.fold_gelus(gm.graph) == 0
    assert cr.fold_norms(gm.graph) == 0
    assert cr.fold_residuals(gm.graph) == 0
    # a second user of the modulated input (a bf16 Linear such as LTX-2.5's attention gate logits) keeps the norm
    _arm_all(cr)
    gm, _args, _ref = _fold_block(cr)
    rotq = [n for n in gm.graph.nodes if str(n.target).endswith("native_cuda_rotq.default")][0]
    root, _ = cr._view_root(rotq.args[0])
    with gm.graph.inserting_after(root):
        gm.graph.call_function(torch.ops.aten.sum.default, (root,))
    assert cr.fold_norms(gm.graph) == 0
    # a second user of the GEMM output keeps the GELU outside the GEMM
    gm, _args, _ref = _fold_block(cr)
    gemm = [n for n in gm.graph.nodes if str(n.target).endswith("native_cuda_i8mm.default")][0]
    with gm.graph.inserting_after(gemm):
        gm.graph.call_function(torch.ops.aten.sum.default, (gemm,))
    assert cr.fold_gelus(gm.graph) == 0


def test_fold_switches_and_defaults(cr, monkeypatch):
    for env in cr.FOLD_ENVS.values():
        monkeypatch.delenv(env, raising = False)
    # norm prologue opt-in (measured no faster on an L4), GELU and residual epilogues on
    assert set(cr.ShareRotqPass().folds) == {"gelu", "residual"}
    monkeypatch.setenv(cr.FOLD_ENVS["norm"], "1")
    assert cr.fold_enabled("norm") and "norm" in cr.ShareRotqPass().folds
    for kind in ("gelu", "residual"):
        monkeypatch.setenv(cr.FOLD_ENVS[kind], "0")
        assert not cr.fold_enabled(kind) and kind not in cr.ShareRotqPass().folds
    on, off = cr.ShareRotqPass(folds = ("gelu",)), cr.ShareRotqPass(folds = ())
    assert on.uuid() != off.uuid()  # a switch flip misses the FX graph cache


def test_pass_runs_the_folds_it_was_built_with(cr):
    _arm_all(cr)
    gm, _args, _ref = _fold_block(cr)
    cr.ShareRotqPass(share = True, folds = ("residual",))(gm.graph)
    names = _targets(gm)
    assert sum(n.endswith("native_cuda_i8mm_res.default") for n in names) == 1
    assert not any("i8mm_gelu" in n or "rotq_norm" in n for n in names)


@pytest.mark.parametrize(
    "fold, expected", [("fold_gelus", 1), ("fold_norms", 0), ("fold_residuals", 0)]
)
def test_folds_keep_a_dynamic_graph_dynamic(cr, fold, expected):
    """A dynamic=True compile traces symbolic token counts: a fold that bakes the count in adds a shape guard, so the
    graph recompiles per token count where the stock graph compiles once. The norm and residual folds keep the traced
    maths there; the GELU fold needs no count and still folds."""
    _arm_all(cr)
    gm, _args, _ref = _fold_block(cr, tracing_mode = "symbolic")
    x = next(n for n in gm.graph.nodes if n.op == "placeholder").meta["val"]
    env = x.fake_mode.shape_env
    assert not all(isinstance(d, int) for d in x.shape)  # the graph really is symbolic
    before = [str(g.expr) for g in env.guards]
    assert getattr(cr, fold)(gm.graph) == expected
    assert [str(g.expr) for g in env.guards] == before


def test_epilogue_and_prologue_references_on_cpu(cr):
    """The ops' CPU fallbacks equal their references: GELU = the tanh decomposition on the bf16 Linear out, RMSNorm +
    AdaLN per batch and per token, and the residual = bf16 Linear out, fp32 ``r + y * g``, one round."""
    import torch

    x = _x(rows = 48, k = 512)
    xf = x.float()
    assert torch.equal(
        cr.gelu_reference(x),
        (
            xf * 0.5 * (1 + torch.tanh((xf + 0.044715 * xf * xf * xf) * 0.7978845608028654))
        ).bfloat16(),
    )
    g = torch.Generator().manual_seed(2)
    for tt in (1, 24):
        sc = (torch.randn(2, tt, 512, generator = g) * 0.2).bfloat16()
        sh = (torch.randn(2, tt, 512, generator = g) * 0.2).bfloat16()
        q, s = cr._launch_pro(x, 256, 2, sc, sh, 24, 1e-6)
        n = (x * torch.rsqrt(xf.pow(2).mean(-1, keepdim = True) + 1e-6)).bfloat16()
        want = (n.float().view(2, 24, 512) * (1 + sc.float()) + sh.float()).bfloat16().view(48, 512)
        assert cr.bf16_close(cr.norm_reference(x, sc, sh, 24, 1e-6), want, 0.01)
        rq, rs = cr.reference(cr.norm_reference(x, sc, sh, 24, 1e-6), 256)
        assert torch.equal(q, rq) and torch.equal(s, rs)
    m = _linear(i = 512, o = 96)
    xq, xs = cr.reference(x, 256)
    ws = m.weight_scale.view(torch.float32)
    y = cr.reference_gemm(xq, m.weight_q, xs, ws, m.bias)
    assert torch.equal(cr._gemm_launch(xq, m.weight_q, xs, ws, m.bias, act = 1), cr.gelu_reference(y))
    r = (torch.randn(48, 96, generator = g) * 3).bfloat16()
    gate = torch.randn(2, 1, 96, generator = g).bfloat16()
    out = cr._gemm_launch(xq, m.weight_q, xs, ws, m.bias, r, gate, 24)
    want = (
        (r.float().view(2, 24, 96) + y.float().view(2, 24, 96) * gate.float())
        .bfloat16()
        .view(48, 96)
    )
    assert cr.bf16_close(out, want, 0.01)
    assert torch.equal(out, cr.reference_gemm_res(xq, m.weight_q, xs, ws, m.bias, r, gate, 24))


def test_gemm_probe_does_not_need_the_torchao_route(cr, monkeypatch):
    """A card the torchao route's own probe turns down still takes the native epilogue on the arch tile, probed here;
    no arch tile (gated / switched off) keeps cuBLAS."""
    import torch

    from core.inference import diffusion_int8_gemm as g8

    src = inspect.getsource(cr._gemm_probe)
    assert "device_config(index)" in src and "arch_config(" in src
    monkeypatch.setattr(g8, "_kernels", lambda: object())
    monkeypatch.setattr(g8, "_triton_version_ok", lambda: True)
    monkeypatch.setattr(g8, "device_config", lambda index: None)
    monkeypatch.setattr(g8, "arch_config", lambda cap, mode = None: None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (8, 9))
    assert "no fused-dequant GEMM tile" in cr._gemm_probe(0)


def test_native_epilogue_kernel_source():
    """addcmul is one FMA on CUDA, so the native bias epilogue is too; the residual epilogue rounds the Linear out to
    bf16 before ``fma(y, gate, residual)``; the GELU epilogue runs on the bf16 Linear out."""
    from core.inference import diffusion_int8_gemm as g8

    src = inspect.getsource(g8._kernels)
    assert "fma_rn(\n                    y, ws[None, :]" in src
    assert "fma_rn(_rbf16(y), g, r)" in src
    assert "RES: tl.constexpr = False" in src and "ACT: tl.constexpr = 0" in src
    assert {"residual", "gate", "tokens", "act"} <= set(inspect.signature(g8._launch).parameters)


def test_rotq_kernel_norm_prologue_flags():
    from core.inference import diffusion_convrot_quant as cq

    src = inspect.getsource(cq._kernels)
    assert "PRO: tl.constexpr = 0" in src and "MOD: tl.constexpr = False" in src
    assert "libdevice.rsqrt(rss / (NG * G) + NORM_EPS)" in src
    sig = inspect.signature(cq._rotq_launch)
    assert {"prologue", "scale", "shift", "tokens", "norm_eps"} <= set(sig.parameters)


@pytest.mark.parametrize("env", ["UNSLOTH_NATIVE_INT8_ROTQ", "UNSLOTH_NATIVE_INT8_GEMM"])
def test_a_later_load_with_the_switch_off_does_not_inherit_the_route(cr, monkeypatch, env):
    import torch

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    cr._STATE["op"] = object()
    cr._STATE["on"] = {0: True}  # an earlier load in this process armed the route on this card
    cr._STATE["gemm"] = {0: True}
    cr._STATE["gemm_op"] = object()
    first = _linear()
    monkeypatch.setenv(env, "0")
    later = _linear()
    t = _wrap(later)
    monkeypatch.setattr(cr, "prepare", lambda device = None: True)
    cr.install(t)
    assert first._cuda_rotq_on(_fake_x()) and first._cuda_gemm_load
    if env == "UNSLOTH_NATIVE_INT8_ROTQ":
        assert not later._cuda_rotq_on(_fake_x())
    else:
        assert later._cuda_rotq_on(_fake_x()) and not later._cuda_gemm_load
    monkeypatch.delenv(env)
    cr.install(t)  # a reload with the switch back on re-arms the same denoiser
    assert later._cuda_rotq_on(_fake_x()) and later._cuda_gemm_load
