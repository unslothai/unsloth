# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DoRA factored row norm (unsloth/kernels/dora.py) against PEFT's dense forward and an fp64 oracle (CPU)."""

import ast
import copy
import importlib.util
import os
import sys

import pytest
import torch

peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model  # noqa: E402
from peft.tuners.lora import dora as peft_dora  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_ROOT, "unsloth", "kernels", "dora.py")


def _load():
    spec = importlib.util.spec_from_file_location("_unsloth_kernels_dora", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def dora_mod(monkeypatch):
    original = peft_dora.DoraLinearLayer.forward
    original = getattr(original, "_unsloth_original", original)
    monkeypatch.setattr(peft_dora.DoraLinearLayer, "forward", original)
    monkeypatch.delenv("UNSLOTH_DORA_FACTORED_NORM", raising = False)
    mod = _load()
    assert mod.patch_peft_dora_factored_norm()
    yield mod, original


class _Net(torch.nn.Module):
    def __init__(
        self,
        bias,
        conv1d = False,
    ):
        super().__init__()
        self.up = torch.nn.Linear(160, 96, bias = bias)
        self.down = torch.nn.Linear(96, 160, bias = bias)
        if conv1d:
            from transformers.pytorch_utils import Conv1D
            self.up = Conv1D(96, 160)
            self.down = Conv1D(160, 96)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.up(x)))


def _make(
    bias = False,
    dropout = 0.0,
    conv1d = False,
    r = 8,
):
    torch.manual_seed(0)
    net = _Net(bias, conv1d).double()
    cfg = LoraConfig(
        r = r,
        lora_alpha = 2 * r,
        lora_dropout = dropout,
        use_dora = True,
        target_modules = ["up", "down"],
        fan_in_fan_out = conv1d,
        init_lora_weights = False,  # random lora_B, so the norm sees a real update
    )
    model = get_peft_model(net, cfg)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.mul_(5.0)
    return model


def _run(model, x, forward_impl, dtype):
    m = copy.deepcopy(model).to(dtype)
    old = peft_dora.DoraLinearLayer.forward
    peft_dora.DoraLinearLayer.forward = forward_impl
    try:
        xx = x.to(dtype).requires_grad_(True)
        out = m(xx)
        out.float().pow(2).mean().backward()
    finally:
        peft_dora.DoraLinearLayer.forward = old
    grads = {n: p.grad.double() for n, p in m.named_parameters() if p.grad is not None}
    return out.double(), grads


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("conv1d", [False, True])
def test_matches_fp64_oracle(dora_mod, bias, conv1d):
    mod, original = dora_mod
    patched = peft_dora.DoraLinearLayer.forward
    model = _make(bias = bias, conv1d = conv1d)
    model.train()
    x = torch.randn(4, 7, 160, dtype = torch.float64)
    ref_out, ref_grads = _run(model, x, original, torch.float64)  # PEFT dense in fp64 = oracle
    for dtype in (torch.float64, torch.float32, torch.bfloat16):
        before = mod.STATS["factored"]
        out, grads = _run(model, x, patched, dtype)
        peft_out, peft_grads = _run(model, x, original, dtype)
        if dtype != torch.bfloat16:
            # float32 / float64 activations keep PEFT's path (its dense sum is already exact there).
            assert mod.STATS["factored"] == before
            torch.testing.assert_close(out, peft_out, rtol = 0, atol = 0)
            continue
        assert mod.STATS["factored"] - before == 2  # both DoRA modules took the factored path
        err, peft_err = _rel(out, ref_out), _rel(peft_out, ref_out)
        assert err <= peft_err * 1.05 + 1e-3, (dtype, err, peft_err)
        for name, g in ref_grads.items():
            e, pe = _rel(grads[name], g), _rel(peft_grads[name], g)
            assert e <= pe * 1.05 + 2e-2, (dtype, name, e, pe)


@pytest.mark.parametrize("base_dtype", [torch.bfloat16, torch.float16])
def test_16bit_base_with_float32_adapters(dora_mod, base_dtype):
    # Unsloth's layout: 16-bit base weights, float32 LoRA weights, so PEFT casts x to float32 before DoRA.
    mod, original = dora_mod
    patched = peft_dora.DoraLinearLayer.forward
    model = _make(bias = True).float()
    model.train()
    for m in model.modules():
        if hasattr(m, "base_layer"):
            m.base_layer.to(base_dtype)
    x = torch.randn(4, 7, 160).to(base_dtype)
    old = peft_dora.DoraLinearLayer.forward
    outs = {}
    for name, impl in (("fact", patched), ("peft", original)):
        before = mod.STATS["factored"]
        peft_dora.DoraLinearLayer.forward = impl
        try:
            outs[name] = model(x).double()
        finally:
            peft_dora.DoraLinearLayer.forward = old
        assert mod.STATS["factored"] - before == (2 if name == "fact" else 0)
    # PEFT's dense norm is float32 here; the factored one agrees to float32 rounding.
    assert _rel(outs["fact"], outs["peft"]) < 1e-5


def _devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _oracle_norm(W, A, B, s):
    return torch.linalg.norm(W.double() + s * (B.double() @ A.double()), dim = 1)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("a_dtype", [torch.float32, torch.bfloat16])
def test_norm_identity(dora_mod, device, a_dtype):
    mod, _ = dora_mod
    torch.manual_seed(1)
    W = torch.randn(64, 48, device = device).bfloat16()
    A = torch.randn(4, 48, device = device).to(a_dtype)
    B = torch.randn(64, 4, device = device).to(a_dtype)
    fact = mod.factored_dora_weight_norm(W, A, B, 0.7, torch.float32).double()
    torch.testing.assert_close(fact, _oracle_norm(W, A, B, 0.7), rtol = 1e-6, atol = 0)


@pytest.mark.parametrize("device", _devices())
def test_cancellation_counterexample(dora_mod, device):
    # W + s B A = [-0.0078125, -0.015625]: a bfloat16 cross term (2.03125) made the squared norm negative, so the
    # norm clamped to 0 and magnitude / norm became inf, where PEFT's dense path stays finite.
    mod, _ = dora_mod
    W = torch.tensor([[1.0, 1.0]], dtype = torch.bfloat16, device = device)
    A = torch.tensor([[1.0078125, 1.015625]], dtype = torch.bfloat16, device = device)
    B = torch.tensor([[-1.0]], dtype = torch.bfloat16, device = device)
    oracle = _oracle_norm(W, A, B, 1.0)
    for dtype in (torch.float32, torch.bfloat16):
        fact = mod.factored_dora_weight_norm(W, A, B, 1.0, dtype)
        assert torch.isfinite(fact).all() and (fact > 0).all()
        torch.testing.assert_close(fact.double(), oracle, rtol = 4e-3, atol = 0)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("a_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("w_dtype", [torch.bfloat16, torch.float16])
def test_near_cancellation(dora_mod, device, a_dtype, w_dtype):
    # s B A nearly cancels W, so every row norm is a few % of ||W_i||: the squared norm is a difference of terms ~1000x
    # larger than itself. The factored norm must stay finite and close to fp64.
    mod, _ = dora_mod
    torch.manual_seed(2)
    d_out, d_in, r, s = 96, 256, 8, 2.0
    A = (torch.randn(r, d_in, device = device) * 0.3).to(a_dtype)
    B = (torch.randn(d_out, r, device = device) * 0.3).to(a_dtype)
    delta = torch.randn(d_out, d_in, device = device) * 0.01
    W = (delta - s * (B.float() @ A.float())).to(w_dtype)
    oracle = _oracle_norm(W, A, B, s)
    assert (oracle / torch.linalg.norm(W.double(), dim = 1)).max() < 0.1
    fact = mod.factored_dora_weight_norm(W, A, B, s, torch.float32).double()
    lw = B.to(w_dtype) @ A.to(w_dtype)  # PEFT: B @ A and W + s * B @ A in the weight dtype
    peft = torch.linalg.norm(W + s * lw, dim = 1).double()
    assert torch.isfinite(fact).all() and (fact > 0).all()
    err = ((fact - oracle).abs() / oracle).max().item()
    peft_err = ((peft - oracle).abs() / oracle).max().item()
    assert err < 5e-3, (err, peft_err)
    if w_dtype == torch.bfloat16:
        assert err <= peft_err, (err, peft_err)  # bf16 rounds W + s B A to 8 bits; fp16 keeps 11


@pytest.mark.parametrize("device", _devices())
def test_cross_term_transposed_weight_and_partial_tiles(dora_mod, device):
    mod, _ = dora_mod
    torch.manual_seed(5)
    A = torch.randn(9, 1025, device = device)
    # PEFT transposes the stored Conv1D weight before computing the norm.
    W = torch.randn(1025, 97, device = device, dtype = torch.float16).t()
    expected = A.double() @ W.double().t()
    assert _rel(mod._cross_term_fp32(W, A).double(), expected) < 2e-6
    # Inductor elides a float16 -> float32 cast pair, which dropped the low half of a cast-based split.
    torch._dynamo.reset()
    compiled = torch.compile(mod._cross_term_fp32, fullgraph = True)
    assert _rel(compiled(W, A).double(), expected) < 2e-6
    torch._dynamo.reset()


def _variants(kernel):
    # Same fallback as tests/test_triton_variant_budget.py: Triton 3.2 exposes `cache`, not `device_caches`.
    import triton

    while not isinstance(kernel, triton.runtime.jit.JITFunction):
        kernel = kernel.fn
    caches = getattr(kernel, "device_caches", None)
    if caches is not None:
        return sum(len(v[0]) for v in caches.values())
    return sum(len(v) for v in kernel.cache.values())


def _uses_kernel(mod, device):
    return (
        device == "cuda" and mod.triton is not None and torch.cuda.get_device_capability()[0] >= 8
    )


@pytest.mark.parametrize("device", _devices())
def test_one_kernel_variant_across_projection_shapes(dora_mod, device):
    mod, _ = dora_mod
    if not _uses_kernel(mod, device):
        pytest.skip("Triton cross term needs CUDA with compute capability 8+")
    torch.manual_seed(6)
    A = torch.randn(16, 2048, device = device)
    for d_out in (512, 2048, 8192, 1000):  # k / v, q / o, gate / up, and an odd width
        W = torch.randn(d_out, 2048, device = device, dtype = torch.bfloat16)
        # A bfloat16 hi + lo split keeps about 16 bits of the float32 adapter.
        assert _rel(mod._cross_term_fp32(W, A).double(), A.double() @ W.double().t()) < 1e-5
    # 16-aligned widths share one, the odd width adds one
    assert _variants(mod._dora_cross_kernel) <= 2


def test_triton_off_switch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DORA_TRITON", "0")
    assert _load().triton is None


def test_autocast_does_not_lower_precision(dora_mod):
    mod, _ = dora_mod
    torch.manual_seed(3)
    W = torch.randn(32, 64).bfloat16()
    A, B = torch.randn(4, 64), torch.randn(32, 4)
    ref = mod.factored_dora_weight_norm(W, A, B, 0.5, torch.float32)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        out = mod.factored_dora_weight_norm(W, A, B, 0.5, torch.float32)
    torch.testing.assert_close(out, ref, rtol = 0, atol = 0)


def test_dropout_path_matches_peft(dora_mod):
    # lora_dropout > 0 in training: PEFT recomputes the base output from the dropped x (base_result is None).
    mod, original = dora_mod
    patched = peft_dora.DoraLinearLayer.forward
    model = _make(dropout = 0.1)
    model.train()
    x = torch.randn(3, 5, 160, dtype = torch.float64)
    before = mod.STATS["factored"]
    torch.manual_seed(5)
    out_p, g_p = _run(model, x, patched, torch.bfloat16)
    assert mod.STATS["factored"] - before == 2
    torch.manual_seed(5)
    out_o, g_o = _run(model, x, original, torch.bfloat16)
    torch.manual_seed(5)
    out_ref, _ = _run(model, x, original, torch.float64)
    assert _rel(out_p, out_ref) <= _rel(out_o, out_ref) * 1.05 + 1e-3
    for n in g_o:
        assert _rel(g_p[n], g_o[n]) < 5e-2, n


def test_eval_cache_and_off_switch_fall_back(dora_mod, monkeypatch):
    mod, original = dora_mod
    model = _make().bfloat16()
    x = torch.randn(2, 3, 160, dtype = torch.bfloat16)
    model.eval()
    monkeypatch.setattr(peft_dora, "ENABLE_DORA_CACHING", True, raising = False)
    before = dict(mod.STATS)
    with torch.no_grad():
        model(x)
    assert mod.STATS["factored"] == before["factored"]
    assert mod.STATS["fallback"] == before["fallback"] + 2
    monkeypatch.setattr(peft_dora, "ENABLE_DORA_CACHING", False, raising = False)
    with torch.no_grad():
        model(x)
    assert mod.STATS["factored"] == before["factored"] + 2

    # Off switch: nothing is installed.
    monkeypatch.setattr(peft_dora.DoraLinearLayer, "forward", original)
    monkeypatch.setenv("UNSLOTH_DORA_FACTORED_NORM", "0")
    assert not mod.patch_peft_dora_factored_norm()
    assert peft_dora.DoraLinearLayer.forward is original


def test_patch_is_idempotent(dora_mod):
    mod, _ = dora_mod
    fwd = peft_dora.DoraLinearLayer.forward
    assert mod.patch_peft_dora_factored_norm()
    assert peft_dora.DoraLinearLayer.forward is fwd


def test_installed_at_import():
    path = os.path.join(_ROOT, "unsloth", "models", "_utils.py")
    tree = ast.parse(open(path, encoding = "utf-8").read())
    calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "patch_peft_dora_factored_norm"
    ]
    assert len(calls) == 1


def test_compiled_forward_does_not_recompile_per_call(dora_mod):
    # Unsloth compiles each decoder block; anything in the traced forward that changes per call (a Python
    # counter) recompiles every step until the block falls back to eager.
    mod, _ = dora_mod
    model = _make().bfloat16()
    model.train()
    x = torch.randn(2, 3, 160, dtype = torch.bfloat16)
    before = mod.STATS["factored"]
    model(x)
    assert mod.STATS["factored"] - before == 2  # the factored path is what gets compiled below
    torch._dynamo.reset()
    from torch._dynamo.utils import counters

    counters.clear()
    compiled = torch.compile(model, backend = "eager")
    compiled(x)
    graphs = counters["stats"]["unique_graphs"]
    for _ in range(4):
        compiled(x)
    assert counters["stats"]["unique_graphs"] == graphs
    assert not counters["recompiles"]
    torch._dynamo.reset()
