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
    for dtype, tol in ((torch.float64, 1e-12), (torch.float32, 1e-5), (torch.bfloat16, None)):
        before = mod.STATS["factored"]
        out, grads = _run(model, x, patched, dtype)
        assert mod.STATS["factored"] - before == 2  # both DoRA modules took the factored path
        peft_out, peft_grads = _run(model, x, original, dtype)
        err, peft_err = _rel(out, ref_out), _rel(peft_out, ref_out)
        if tol is not None:
            assert err < tol, (dtype, err)
        else:
            assert err <= peft_err * 1.05 + 1e-3, (dtype, err, peft_err)
        for name, g in ref_grads.items():
            e, pe = _rel(grads[name], g), _rel(peft_grads[name], g)
            if tol is not None:
                assert e < tol * 10, (dtype, name, e)
            else:
                assert e <= pe * 1.05 + 2e-2, (dtype, name, e, pe)


def test_norm_identity_fp64(dora_mod):
    mod, _ = dora_mod
    torch.manual_seed(1)
    W = torch.randn(64, 48, dtype = torch.float64)
    A = torch.randn(4, 48, dtype = torch.float64)
    B = torch.randn(64, 4, dtype = torch.float64)
    dense = torch.linalg.norm(W + 0.7 * (B @ A), dim = 1)
    fact = mod.factored_dora_weight_norm(W, A, B, 0.7, torch.float64)
    torch.testing.assert_close(fact, dense, rtol = 1e-6, atol = 0)  # float32 accumulation inside


def test_dropout_path_matches_peft(dora_mod):
    mod, original = dora_mod
    patched = peft_dora.DoraLinearLayer.forward
    model = _make(dropout = 0.1)
    model.train()
    x = torch.randn(3, 5, 160, dtype = torch.float64)
    torch.manual_seed(5)
    out_p, g_p = _run(model, x, patched, torch.float64)
    torch.manual_seed(5)
    out_o, g_o = _run(model, x, original, torch.float64)
    torch.testing.assert_close(out_p, out_o, rtol = 1e-10, atol = 1e-12)
    for n in g_o:
        torch.testing.assert_close(g_p[n], g_o[n], rtol = 1e-9, atol = 1e-12)


def test_eval_cache_and_off_switch_fall_back(dora_mod, monkeypatch):
    mod, original = dora_mod
    model = _make()
    x = torch.randn(2, 3, 160, dtype = torch.float64)
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
    model = _make().float()
    model.train()
    x = torch.randn(2, 3, 160)
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
