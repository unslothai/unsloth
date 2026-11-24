# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A tensor parallel LoRA projection rounds as the one-GPU compiled LoRA forward (unsloth_zoo compiler
`lora_forward` inside the generated PEFT forward): under forced float32 the low-rank product in float16; else,
after the generated input cast, in the LoRA dtype (dense Linear: result and x cast) or in the base output's dtype
(bitsandbytes: only x cast, dropout before the cast); the update added by one addmm, restored to the output's
dtype. Output and lora_A / lora_B gradients of one projection must be bitwise those of the one-GPU forward.

Under forced float32 (Gemma 3 / 4 on float16 GPUs) the tp product used to run in float32: per-token LoRA
gradients that underflow float16 on one GPU survived on the tp ranks, so Gemma-4 E4B 4-bit tp=2 trained on other
gradients (2 gloo ranks on one B200: max loss gap to the one-GPU median 0.064, above the 0.023 repeat spread).
The one-GPU reference is zoo's own template source, read without importing zoo; a hand copy when zoo is absent.
CPU, one process, no DTensor."""

import ast
import importlib.util
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_oracle import tpn  # noqa: E402  (tp_native without the GPU-only unsloth import)


def _zoo_templates():
    """COMPILED_LORA_FORWARD(_forced_float32) source from unsloth_zoo/compiler.py, or None."""
    try:
        spec = importlib.util.find_spec("unsloth_zoo")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    path = os.path.join(spec.submodule_search_locations[0], "compiler.py")
    if not os.path.exists(path):
        return None
    with open(path, encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("COMPILED_LORA_FORWARD"):
                    found[target.id] = node.value.value
    if {"COMPILED_LORA_FORWARD", "COMPILED_LORA_FORWARD_forced_float32"} - found.keys():
        return None
    return found


_ZOO = _zoo_templates()


def _hand_lora_forward(forced, no_compiled_addmm):
    """A copy of zoo's two templates, for an environment without unsloth_zoo."""

    def lora_forward(result, lora_A, lora_B, dropout, x, scaling):
        if forced:
            xA = dropout(x.to(torch.float16)) @ lora_A.weight.to(torch.float16).t()
        else:
            xA = dropout(x).to(result.dtype) @ lora_A.weight.to(result.dtype).t()
        dtype = result.dtype
        if no_compiled_addmm:
            return result + scaling * (xA.to(dtype) @ lora_B.weight.to(dtype).t())
        shape = result.shape
        return torch.addmm(
            result.view(-1, shape[-1]),
            xA.view(-1, xA.shape[-1]).to(dtype),
            lora_B.weight.to(dtype).t(),
            alpha = scaling,
            beta = 1,
        ).view(shape)

    return lora_forward


def _lora_forward(forced, no_compiled_addmm = False):
    """zoo's lora_forward for one template; `no_compiled_addmm` takes its torch < 2.10 compiled branch."""
    if _ZOO is None:
        return _hand_lora_forward(forced, no_compiled_addmm)
    name = "COMPILED_LORA_FORWARD_forced_float32" if forced else "COMPILED_LORA_FORWARD"
    scope = {"torch": torch}
    exec(_ZOO[name], scope)
    scope["torch_lora_no_compiled_addmm"] = no_compiled_addmm
    scope["torch_is_compiling"] = lambda: no_compiled_addmm
    return scope["lora_forward"]


# Stand-ins for PEFT's original forwards: zoo keeps PEFT's own x cast when the source branches on autocast
# (bitsandbytes Linear4bit / 8bitLt), else casts result and x to the LoRA dtype (dense Linear).
def _peft_dense_forward(self, x):
    return x


def _peft_bnb_forward(self, x):
    requires_conversion = not torch.is_autocast_enabled()
    return x if requires_conversion else x


class _LoraProj(nn.Module):
    def __init__(
        self,
        base_dtype,
        in_features = 64,
        out_features = 48,
        r = 8,
        seed = 0,
        scaling = 2.0,
        dropout = 0.0,
    ):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.base_layer = nn.Linear(in_features, out_features, bias = False).to(base_dtype)
        with torch.no_grad():
            self.base_layer.weight.copy_(torch.randn(out_features, in_features, generator = g) * 0.05)
        self.base_layer.weight.requires_grad_(False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(in_features, r, bias = False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, out_features, bias = False)})
        with torch.no_grad():
            self.lora_A["default"].weight.copy_(torch.randn(r, in_features, generator = g) * 0.05)
            self.lora_B["default"].weight.copy_(torch.randn(out_features, r, generator = g) * 0.05)
        drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_dropout = nn.ModuleDict({"default": drop})
        self.scaling = {"default": scaling}
        self.disable_adapters = False
        self.merged = False
        self.active_adapters = ["default"]
        self._unsloth_tp_style = tpn.COLWISE

    def _cast_input_dtype(self, x, dtype):
        return x if x.dtype == dtype else x.to(dtype)


def _with_forward(flag, peft = "dense"):
    """A layer class whose forward is Unsloth's generated LoRA forward built under `flag` over PEFT's `peft`
    ("dense" Linear or "bnb" Linear4bit) forward, or PEFT's own forward (flag None)."""
    if flag is None:
        return _LoraProj

    def unsloth_forward(self, x):
        return _one_gpu_forward(self, x, flag, peft)

    unsloth_forward._unsloth_force_float32 = flag
    unsloth_forward._unsloth_original_forward = (
        _peft_dense_forward if peft == "dense" else _peft_bnb_forward
    )
    return type("CompiledLoraProj", (_LoraProj,), {"forward": unsloth_forward})


def _peft_forward(proj, x):
    """PEFT's forward (no Unsloth compile): x cast to the LoRA weights' dtype, product in float32."""
    A = proj.lora_A["default"].weight
    B = proj.lora_B["default"].weight
    result = torch.nn.functional.linear(x, proj.base_layer.weight)
    xd = proj.lora_dropout["default"](x.to(A.dtype))
    return result + (
        torch.nn.functional.linear(torch.nn.functional.linear(xd, A), B) * proj.scaling["default"]
    ).to(result.dtype)


def _one_gpu_forward(
    proj,
    x,
    flag,
    peft,
    no_compiled_addmm = False,
):
    """The generated Linear / Linear4bit unsloth_forward (unsloth_compiled_cache/*_peft_forward.py), autocast off."""
    forced = str(flag) != "0"
    lora_A, lora_B = proj.lora_A["default"], proj.lora_B["default"]
    result = torch.nn.functional.linear(x, proj.base_layer.weight)
    result_dtype = result.dtype
    if not forced:
        if (
            peft == "dense"
        ):  # zoo: `result, x = result.to(lora_A.weight.dtype), x.to(lora_A.weight.dtype)`
            result, x = result.to(lora_A.weight.dtype), x.to(lora_A.weight.dtype)
        else:  # PEFT bnb: `x = self._cast_input_dtype(x, lora_A.weight.dtype)`
            x = proj._cast_input_dtype(x, lora_A.weight.dtype)
    result = _lora_forward(forced, no_compiled_addmm)(
        result, lora_A, lora_B, proj.lora_dropout["default"], x, proj.scaling["default"]
    )
    return result.to(result_dtype)


def _run(
    fn,
    proj,
    x,
    g,
    seed = 3,
):
    for p in (proj.lora_A["default"].weight, proj.lora_B["default"].weight):
        p.grad = None
    torch.manual_seed(seed)  # the same dropout mask in both arms
    y = fn(proj, x)
    y.backward(g)
    return (
        y.detach(),
        proj.lora_A["default"].weight.grad.clone(),
        proj.lora_B["default"].weight.grad.clone(),
    )


def _bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def _inputs(dtype):
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(2, 16, 64, generator = g) * 0.5).to(dtype)
    # Tiny upstream gradients, as a loss-scaled float16 backward sees in Gemma-4's upper layers: the one-GPU
    # float16 product flushes part of the low-rank gradient to zero, a float32 product keeps it.
    grad = (torch.randn(2, 16, 48, generator = g) * 3e-6).to(dtype)
    return x, grad


def _assert_same(ref, got):
    for name, a, b in zip(("output", "lora_A grad", "lora_B grad"), ref, got):
        assert a.dtype == b.dtype and a.shape == b.shape, name
        bad = int((_bits(a) != _bits(b)).sum())
        assert (
            bad == 0
        ), f"{name}: {bad} of {a.numel()} elements differ from the one-GPU LoRA forward"


def _check(
    flag,
    dtype,
    peft = "dense",
    scaling = 2.0,
    dropout = 0.0,
):
    proj = _with_forward(flag, peft)(dtype, scaling = scaling, dropout = dropout)
    x, grad = _inputs(dtype)
    one_gpu = (lambda p, xx: p.forward(xx)) if flag is not None else _peft_forward
    _assert_same(_run(one_gpu, proj, x, grad), _run(tpn._tp_projection_compute, proj, x, grad))


@pytest.mark.parametrize("scaling", [2.0, 1.7])
@pytest.mark.parametrize("peft", ["dense", "bnb"])
@pytest.mark.parametrize("flag", ["1", "0", None])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_tp_lora_matches_one_gpu_forward(flag, dtype, peft, scaling):
    if flag == "1" and dtype is torch.bfloat16:
        pytest.skip("forced float32 keeps float16 (or float32) base weights")
    if flag is None and peft == "bnb":
        pytest.skip("PEFT's own forward: one case")
    _check(flag, dtype, peft, scaling)


@pytest.mark.parametrize("peft", ["dense", "bnb"])
@pytest.mark.parametrize("flag", ["1", "0"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tp_lora_dropout_matches_one_gpu_forward(flag, dtype, peft):
    """Dropout runs where the generated forward runs it: on float16 x under forced float32, else on x after the
    cast to the LoRA dtype (a CPU dropout mask rounds 1 / (1 - p) to the activation dtype)."""
    if flag == "1" and dtype is torch.bfloat16:
        pytest.skip("forced float32 keeps float16 base weights")
    _check(flag, dtype, peft, dropout = 0.25)


def test_dense_default_product_runs_in_lora_dtype():
    """Dense Linear without forced float32: the generated forward casts result and x to the float32 LoRA dtype,
    so a bfloat16 product on the tp ranks is not the one-GPU arithmetic."""
    proj = _with_forward("0", "dense")(torch.bfloat16)
    assert tpn._one_gpu_lora_mode(proj) == "lora"
    proj = _with_forward("0", "bnb")(torch.bfloat16)
    assert tpn._one_gpu_lora_mode(proj) == "output"


@pytest.mark.parametrize("flag", ["true", "True", "2"])
def test_any_force_float32_but_zero_is_forced(flag):
    """zoo builds the forced template for every UNSLOTH_FORCE_FLOAT32 value but "0"."""
    proj = _with_forward(flag)(torch.float16)
    assert tpn._one_gpu_lora_mode(proj) == "float16"
    _check(flag, torch.float16)


def test_mode_needs_the_original_peft_forward():
    """No PEFT source to read the generated cast from: PEFT's float32 fallback, never a guess."""
    cls = _with_forward("0")
    del cls.forward._unsloth_original_forward
    assert tpn._one_gpu_lora_mode(cls(torch.bfloat16)) is None


def test_real_peft_forwards_map_to_their_mode():
    peft_layer = pytest.importorskip("peft.tuners.lora.layer")
    cases = [(peft_layer.Linear, "lora")]
    try:
        from peft.tuners.lora.bnb import Linear4bit
        cases.append((Linear4bit, "output"))
    except Exception:
        pass
    for cls, want in cases:

        def unsloth_forward(self, x):
            return x

        unsloth_forward._unsloth_force_float32 = "0"
        unsloth_forward._unsloth_original_forward = cls.forward
        fake = type("Fake", (_LoraProj,), {"forward": unsloth_forward})
        assert tpn._one_gpu_lora_mode(fake(torch.bfloat16)) == want, cls


@pytest.mark.parametrize("peft", ["dense", "bnb"])
@pytest.mark.parametrize("flag", ["1", "0"])
def test_old_torch_compiled_branch_matches_zoo(monkeypatch, flag, peft):
    """torch < 2.10 under torch.compile: zoo skips addmm (unsloth#3271); the tp update must take the same branch."""
    monkeypatch.setattr(tpn, "_TP_LORA_NO_COMPILED_ADDMM", True, raising = False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    proj = _with_forward(flag, peft)(torch.float16, scaling = 1.7)
    x, grad = _inputs(torch.float16)
    ref = _run(
        lambda p, xx: _one_gpu_forward(p, xx, flag, peft, no_compiled_addmm = True), proj, x, grad
    )
    _assert_same(ref, _run(tpn._tp_projection_compute, proj, x, grad))


def test_update_keeps_the_base_output_dtype():
    """A float32 base output (a 4-bit base restored to a float32 input's dtype) under CPU autocast, which the
    generated forward's no-argument (CUDA) query does not see: addmm comes back in bfloat16, and the one-GPU
    forward restores the base output's float32 (`.to(result.dtype)`)."""
    proj = _with_forward("0", "bnb")(torch.float32)
    x, _ = _inputs(torch.float32)
    out = torch.nn.functional.linear(x, proj.base_layer.weight)
    A, B = proj.lora_A["default"].weight, proj.lora_B["default"].weight
    with torch.autocast("cpu", dtype = torch.bfloat16):
        if torch.is_autocast_enabled():
            pytest.skip("this torch's no-argument autocast query sees CPU autocast")
        got = tpn._tp_lora_update(proj, "default", out, x, A, B, 2.0, "output")
    assert got.dtype == torch.float32


def test_forced_float32_one_gpu_flushes_what_float32_keeps():
    """The case that moved Gemma-4 tp=2 off one GPU: under forced float32 the one-GPU lora_A gradient loses
    entries a float32 product keeps, so a float32 tp product trains on different gradients."""
    proj = _with_forward("1")(torch.float16)
    x, grad = _inputs(torch.float16)
    _, a16, _ = _run(lambda p, xx: p.forward(xx), proj, x, grad)
    _, a32, _ = _run(_peft_forward, proj, x, grad)
    rel = float((a16 - a32).norm() / a32.norm())
    assert rel > 1e-2, rel  # a real difference, not rounding noise
