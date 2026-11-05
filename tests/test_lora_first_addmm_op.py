# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Traced fused LoRA order runs as one opaque op (unsloth::lora_first_addmm): same bits as the eager sequence,
no graph break, and the kill switch keeps the old order."""

import pytest
import torch
import unsloth  # noqa: F401
from unsloth.kernels import utils as KU

# Gated on the API, not on the registration result, so a failed registration on a capable torch is a failure.
pytestmark = pytest.mark.skipif(
    not hasattr(torch.library, "custom_op"), reason = "needs torch.library.custom_op"
)
# Compiled code uses the op only when torch.export's flag tells strict export apart (torch 2.7+).
_OP_COMPILED = hasattr(torch.compiler, "_is_exporting_flag")


def _mk(
    M = 64,
    K = 96,
    N = 80,
    r = 8,
    dtype = torch.bfloat16,
):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(M, K, generator = g).to(dtype)
    W = (torch.randn(N, K, generator = g) * 0.02).to(dtype)
    A = (torch.randn(r, K, generator = g) * 0.1).to(dtype)
    B = (torch.randn(N, r, generator = g) * 2e-3).to(dtype)
    return X, W, A, B


def test_op_matches_eager_sequence_bitwise():
    X, W, A, B = _mk()
    XA = X @ A.t()
    ref = torch.addmm(XA.new_zeros(1), XA, B.t(), beta = 0, alpha = 0.5)
    ref.addmm_(X, W.t())
    got = torch.ops.unsloth.lora_first_addmm(X, W.t(), XA, B.t(), 0.5)
    assert torch.equal(got, ref)


def test_opcheck():
    X, W, A, B = _mk(dtype = torch.float32)
    XA = X @ A.t()
    torch.library.opcheck(
        torch.ops.unsloth.lora_first_addmm.default,
        (X, W.t(), XA, B.t(), 0.5),
        test_utils = ("test_schema", "test_faketensor", "test_aot_dispatch_static"),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_compiled_matmul_lora_uses_op_fullgraph(dtype):
    X, W, A, B = _mk(dtype = dtype)
    eager = KU.matmul_lora(X, W, None, A, B, 0.5)
    import torch._dynamo

    torch._dynamo.reset()
    seen = []
    orig = torch.ops.unsloth.lora_first_addmm

    def backend(gm, inputs):
        seen.extend(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        return gm.forward

    f = torch.compile(
        lambda X, W, A, B: KU.matmul_lora(X, W, None, A, B, 0.5), backend = backend, fullgraph = True
    )
    got = f(X, W, A, B)
    assert any("lora_first_addmm" in s for s in seen) == _OP_COMPILED, seen
    assert torch.equal(got, eager)
    f2 = torch.compile(
        lambda X, W, A, B: KU.matmul_lora(X, W, None, A, B, 0.5),
        backend = "aot_eager",
        fullgraph = True,
    )
    assert torch.equal(f2(X, W, A, B), eager)


def test_kill_switch_keeps_old_order_compiled(monkeypatch):
    X, W, A, B = _mk()
    monkeypatch.setattr(KU, "LORA_FUSED_PASSES", False)
    seen = []

    def backend(gm, inputs):
        seen.extend(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        return gm.forward

    import torch._dynamo

    torch._dynamo.reset()
    torch.compile(
        lambda X, W, A, B: KU.matmul_lora(X, W, None, A, B, 0.5), backend = backend, fullgraph = True
    )(X, W, A, B)
    assert not any("lora_first_addmm" in s for s in seen)


def test_without_the_op_traced_falls_back_to_the_functional_order(monkeypatch):
    # Registration failed (or an older torch.library): the traced fused branch keeps the two functional addmms.
    X, W, A, B = _mk(dtype = torch.float32)
    eager = KU.matmul_lora(X, W, None, A, B, 0.5)
    monkeypatch.setattr(KU, "_HAS_LORA_FIRST_OP", False)
    seen = []

    def backend(gm, inputs):
        seen.extend(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        return gm.forward

    torch._dynamo.reset()
    got = torch.compile(
        lambda X, W, A, B: KU.matmul_lora(X, W, None, A, B, 0.5), backend = backend, fullgraph = True
    )(X, W, A, B)
    assert not any("lora_first_addmm" in s for s in seen)
    torch.testing.assert_close(got, eager, rtol = 1e-6, atol = 1e-6)


def test_opcheck_autograd():
    X, W, A, B = _mk(dtype = torch.float32)
    XA = X @ A.t()
    args = [t.clone().requires_grad_(True) for t in (X, W.t(), XA, B.t())] + [0.5]
    torch.library.opcheck(
        torch.ops.unsloth.lora_first_addmm.default,
        tuple(args),
        test_utils = ("test_schema", "test_autograd_registration", "test_aot_dispatch_dynamic"),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_compiled_differentiable_matmul_lora_matches_eager_grads(dtype):
    # A compiled caller that differentiates through matmul_lora gets the addmm gradients.
    import torch._dynamo
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd

    X, W, A, B = _mk(dtype = dtype)

    def run(fn):
        x, a, b = (t.clone().requires_grad_(True) for t in (X, A, B))
        out = fn(x, W, a, b)
        out.float().square().sum().backward()
        return out.detach(), x.grad, a.grad, b.grad

    f = lambda X, W, A, B: KU.matmul_lora(X, W, None, A, B, 0.5)  # noqa: E731
    eager = run(f)
    torch._dynamo.reset()
    seen = []
    boxed = lambda g, i: make_boxed_func(g.forward)  # noqa: E731

    def backend(gm, inputs):
        seen.extend(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        return aot_autograd(fw_compiler = boxed, bw_compiler = boxed)(gm, inputs)

    compiled = run(torch.compile(f, backend = backend, fullgraph = True))
    assert any("lora_first_addmm" in s for s in seen) == _OP_COMPILED, seen
    tol = dict(rtol = 1e-5, atol = 1e-5) if dtype == torch.float32 else dict(rtol = 2e-2, atol = 2e-2)
    assert torch.equal(compiled[0], eager[0])
    for c, e in zip(compiled[1:], eager[1:]):
        torch.testing.assert_close(c, e, **tol)


@pytest.mark.parametrize("mapped", ["X", "W"])
def test_eager_helper_under_vmap(mapped):
    X, W, A, B = _mk(dtype = torch.float32)
    Xs, Ws = torch.stack([X, X * 2]), torch.stack([W, W * 2])

    def ref(X, W):
        return X @ W.t() + 0.5 * (X @ A.t()) @ B.t()

    if mapped == "X":
        got = torch.vmap(lambda x: KU.matmul_lora(x, W, None, A, B, 0.5))(Xs)
        exp = torch.stack([ref(x, W) for x in Xs])
    else:
        got = torch.vmap(lambda w: KU.matmul_lora(X, w, None, A, B, 0.5))(Ws)
        exp = torch.stack([ref(X, w) for w in Ws])
    torch.testing.assert_close(got, exp, rtol = 1e-5, atol = 1e-5)


def _transform_cases():
    def f(x, W, A, B):
        return KU.matmul_lora(x, W, None, A, B, 0.5)

    def jvp(X, W, A, B):
        return torch.func.jvp(lambda x: f(x, W, A, B), (X,), (torch.ones_like(X),))[1]

    def grad_(X, W, A, B):
        return torch.func.grad(lambda x: f(x, W, A, B).float().square().sum())(X)

    def jacrev(X, W, A, B):
        return torch.func.jacrev(lambda x: f(x, W, A, B).float().sum(0))(X)

    def jacfwd(X, W, A, B):
        return torch.func.jacfwd(lambda x: f(x, W, A, B).float().sum(0))(X)

    def vmap_(X, W, A, B):
        return torch.func.vmap(lambda x: f(x, W, A, B))(torch.stack([X, X * 2]))

    return {"jvp": jvp, "grad": grad_, "jacrev": jacrev, "jacfwd": jacfwd, "vmap": vmap_}


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("name", ["jvp", "grad", "jacrev", "jacfwd", "vmap"])
def test_compiled_function_transforms_match_eager(name, dtype, backend):
    # Compiled inside torch.func transforms, matmul_lora keeps the base-then-LoRA order: the op has no
    # forward-mode or batching rule, so compiled jvp / jacfwd returned zeros or failed to trace.
    import torch._dynamo

    X, W, A, B = _mk(M = 8, K = 16, N = 12, r = 4, dtype = dtype)
    fn = _transform_cases()[name]
    eager = fn(X, W, A, B)
    assert eager.abs().max() > 0
    torch._dynamo.reset()
    got = torch.compile(fn, backend = backend, fullgraph = True)(X, W, A, B)
    tol = dict(rtol = 1e-5, atol = 1e-5) if dtype == torch.float32 else dict(rtol = 2e-2, atol = 2e-2)
    torch.testing.assert_close(got, eager, **tol)


def test_custom_op_saves_only_what_the_requested_grads_need():
    # Frozen X and a base created under inference_mode (cannot be saved for backward), trainable adapters.
    X, W, A, B = _mk(dtype = torch.float32)
    with torch.inference_mode():
        Wi = W.clone()
    a, b = A.clone().requires_grad_(True), B.clone().requires_grad_(True)
    XA = X @ a.t()
    out = torch.ops.unsloth.lora_first_addmm(X, Wi.t(), XA, b.t(), 0.5)
    out.square().sum().backward()
    a2, b2 = A.clone().requires_grad_(True), B.clone().requires_grad_(True)
    (X @ W.t() + 0.5 * (X @ a2.t()) @ b2.t()).square().sum().backward()
    torch.testing.assert_close(a.grad, a2.grad, rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(b.grad, b2.grad, rtol = 1e-5, atol = 1e-5)


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_compiled_native_forward_ad_keeps_the_tangent(dtype, backend):
    # Native forward AD (dual_level / make_dual) inside a compiled function: the op has no forward-mode
    # rule, so matmul_lora keeps the base-then-LoRA order and the tangent survives.
    import torch._dynamo
    import torch.autograd.forward_ad as fwAD

    X, W, A, B = _mk(M = 8, K = 16, N = 12, r = 4, dtype = dtype)

    def tangent(x):
        with fwAD.dual_level():
            y = KU.matmul_lora(fwAD.make_dual(x, torch.ones_like(x)), W, None, A, B, 0.5)
            return fwAD.unpack_dual(y).tangent

    eager = tangent(X)
    assert eager is not None and eager.abs().max() > 0
    torch._dynamo.reset()
    got = torch.compile(tangent, backend = backend, fullgraph = True)(X)
    assert got is not None
    tol = dict(rtol = 1e-5, atol = 1e-5) if dtype == torch.float32 else dict(rtol = 2e-2, atol = 2e-2)
    torch.testing.assert_close(got, eager, **tol)


@pytest.mark.parametrize("strict", [False, True])
def test_exported_program_stays_transformable(strict):
    # Export (non-strict outside Dynamo, strict through it) keeps the functional form: torch.func on the
    # exported module has no rule for the opaque op (grad raised, jvp returned zeros).
    if not hasattr(torch, "export"):
        pytest.skip("torch.export is not available")

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Buffers, not closed-over tensors: strict export on older torch rejects free tensors.
            self.register_buffer("W", torch.eye(3))
            self.register_buffer("A", torch.ones(1, 3))
            self.register_buffer("B", torch.ones(3, 1))

        def forward(self, x):
            return KU.matmul_lora(x, self.W, None, self.A, self.B, 0.5)

    x = torch.ones(2, 3)
    ep = torch.export.export(M(), (x,), strict = strict)
    assert not any("lora_first_addmm" in str(n.target) for n in ep.graph.nodes)
    m = ep.module()
    _, tangent = torch.func.jvp(m, (x,), (torch.ones_like(x),))
    torch.testing.assert_close(tangent, torch.full((2, 3), 2.5))
    g = torch.func.grad(lambda x: m(x).sum())(x)
    torch.testing.assert_close(g, torch.full((2, 3), 2.5))


def test_compiled_lora_forward_and_backward_have_no_graph_break():
    # A compiled LoRA layer (forward + its backward) traces as one graph and still uses the op. On torch
    # 2.10 / 2.11 Dynamo folds is_exporting() to True, which once reached a banned `layout` read.
    import torch._dynamo
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd
    from unsloth.kernels import fast_lora

    X, W, A, B = _mk(M = 8, K = 16, N = 12, r = 4, dtype = torch.float32)
    a, b = torch.nn.Parameter(A.clone()), torch.nn.Parameter(B.clone())
    seen = []

    def boxed(g, i):
        # The op sits in the flattened forward / backward graphs AOTAutograd hands to the compilers.
        seen.extend(str(n.target) for n in g.graph.nodes if n.op == "call_function")
        return make_boxed_func(g.forward)

    def backend(gm, inputs):
        return aot_autograd(fw_compiler = boxed, bw_compiler = boxed)(gm, inputs)

    def f(x):
        return fast_lora.LoRA_W.apply(x.view(1, 8, 16), W, None, a, b, 0.5)

    torch._dynamo.reset()
    x = X.clone().requires_grad_(True)
    out = torch.compile(f, backend = backend, fullgraph = True)(x)
    out.square().sum().backward()
    assert any("lora_first_addmm" in s for s in seen) == _OP_COMPILED, seen
    assert x.grad is not None and a.grad is not None and b.grad is not None


def test_exporting_reads_the_export_flag_not_the_folded_query(monkeypatch):
    # torch 2.10 / 2.11 Dynamo folds is_exporting() to True in every compiled region, so plain compiles
    # would lose the op; the flag torch.export sets is exact. Without the flag, assume export.
    if hasattr(torch.compiler, "_is_exporting_flag"):
        monkeypatch.setattr(torch.compiler, "_is_exporting_flag", False)
        assert KU._exporting() is False
        monkeypatch.setattr(torch.compiler, "_is_exporting_flag", True)
        assert KU._exporting() is True
        monkeypatch.delattr(torch.compiler, "_is_exporting_flag")
    assert KU._exporting() is True


_PARTIAL_REGISTRATION_CASE = """
import torch
from torch._library.custom_ops import CustomOpDef
orig = CustomOpDef.register_autograd
def boom(self, *a, **k):
    if self._qualname == "unsloth::lora_first_addmm":
        raise RuntimeError("injected registration failure")
    return orig(self, *a, **k)
CustomOpDef.register_autograd = boom
import unsloth  # noqa: F401
from unsloth.kernels import utils as KU
assert hasattr(torch.ops.unsloth, "lora_first_addmm"), "the schema should still be registered"
print("HAS_OP", KU._HAS_LORA_FIRST_OP)
"""


def _skip_without_triton():
    # A fresh interpreter imports unsloth.kernels without this process's test stubs, so it needs triton.
    import importlib.util

    try:
        spec = importlib.util.find_spec("triton")
    except ValueError:
        # A stub module in sys.modules with no spec.
        spec = None
    # unsloth_zoo's stand-in, while its finder is installed, is not a real triton either.
    found = spec is not None and spec.origin != "triton_stub"
    if not found:
        pytest.skip("triton is not installed")


def test_a_partly_registered_op_is_not_used():
    # Registration fails after the schema exists (autograd here): the op must stay disabled rather than be
    # enabled because its name resolves. A fresh interpreter, since this process registered it already.
    _skip_without_triton()
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        PYTHONPATH = os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]),
        UNSLOTH_DISABLE_CPU_FALLBACK = "0",
    )
    p = subprocess.run(
        [sys.executable, "-c", _PARTIAL_REGISTRATION_CASE],
        cwd = root,
        env = env,
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert p.returncode == 0 and "HAS_OP False" in p.stdout, (p.stdout + p.stderr)[-3000:]


_REEXECUTED_MODULE_CASE = """
import importlib
import torch
import unsloth  # noqa: F401
from unsloth.kernels import utils as KU
assert KU._HAS_LORA_FIRST_OP is True
calls = []
orig = torch.library.custom_op
def counting(name, *a, **k):
    if name == "unsloth::lora_first_addmm":
        calls.append(name)
    return orig(name, *a, **k)
torch.library.custom_op = counting
KU = importlib.reload(KU)
print("REREGISTERED", len(calls), "HAS_OP", KU._HAS_LORA_FIRST_OP)
"""


def test_a_reexecuted_module_reuses_the_completed_registration():
    # Re-registering over a working op could leave it half replaced if a later step failed; a re-executed
    # module must reuse the ready op instead of defining it again.
    _skip_without_triton()
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        PYTHONPATH = os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]),
        UNSLOTH_DISABLE_CPU_FALLBACK = "0",
    )
    p = subprocess.run(
        [sys.executable, "-c", _REEXECUTED_MODULE_CASE],
        cwd = root,
        env = env,
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert p.returncode == 0 and "REREGISTERED 0 HAS_OP True" in p.stdout, (p.stdout + p.stderr)[
        -3000:
    ]


def test_a_completely_registered_op_is_marked_ready():
    # The marker a re-executed module reads is set only once every registration step succeeded.
    assert torch.ops.unsloth.__dict__.get("_lora_first_addmm_ready") is True
    assert KU._HAS_LORA_FIRST_OP is True
