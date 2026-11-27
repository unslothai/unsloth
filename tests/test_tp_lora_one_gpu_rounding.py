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
import gc
import importlib.util
import inspect
import linecache
import os
import re
import sys
import textwrap

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_oracle import tpn  # noqa: E402  (tp_native without the GPU-only unsloth import)


_ZOO_FILE = "<unsloth_zoo compiler>"


def _zoo_tree():
    """unsloth_zoo/compiler.py parsed, without importing zoo, or None."""
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
        text = f.read()
    # inspect reads the exec'd zoo functions (the LoRA dispatcher) from their real lines, as in a zoo install.
    linecache.cache[_ZOO_FILE] = (len(text), None, text.splitlines(True), _ZOO_FILE)
    return ast.parse(text)


_PATCHERS = (
    "_patch_lora_update",
    "_patch_lora_input_cast",
    "_patch_lora_base_layer_input_cast",
)
_OPTIONAL = ("_install_lora_forward_variant",)  # newer zoo: per-flag LoRA forward dispatcher


def _zoo_parts():
    """zoo's two COMPILED_LORA_FORWARD templates and its real PEFT-forward rewriters, or None."""
    tree = _zoo_tree()
    if tree is None:
        return None
    templates, nodes = {}, []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if isinstance(node.value, ast.Constant) and any(
                n.startswith("COMPILED_LORA_FORWARD") for n in names
            ):
                for n in names:
                    templates[n] = node.value.value
            elif "_LORA_INPUT_CASTS" in names:
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in _PATCHERS + _OPTIONAL:
            nodes.append(node)
    scope = {"os": os, "re": re, "gc": gc, "torch": torch}
    exec(compile(ast.Module(body = nodes, type_ignores = []), _ZOO_FILE, "exec"), scope)
    if {"COMPILED_LORA_FORWARD", "COMPILED_LORA_FORWARD_forced_float32"} - templates.keys():
        return None
    if set(_PATCHERS) - scope.keys():
        return None
    return templates, scope


_ZOO = _zoo_parts()


def _peft_forward_of(peft):
    """PEFT's own forward for "dense" (lora Linear) or "bnb" (lora Linear4bit), or None."""
    try:
        if peft == "dense":
            from peft.tuners.lora.layer import Linear
        else:
            from peft.tuners.lora.bnb import Linear4bit as Linear
    except Exception:
        return None
    # A FastModel load earlier in this process compiles PEFT's forward onto the class in place (zoo
    # patch_lora_forwards); the generated forward keeps PEFT's own as _unsloth_original_forward.
    forward = Linear.forward
    return getattr(forward, "_unsloth_original_forward", forward)


def _generated_forward(flag, peft):
    """Unsloth's generated unsloth_forward (zoo compiler patch_lora_forwards) over PEFT's real forward, built with
    zoo's own rewriters and template; the source registered so inspect reads it like unsloth_compiled_cache."""
    if _ZOO is None:
        pytest.skip("unsloth_zoo is not installed")
    original = _peft_forward_of(peft)
    if original is None:
        pytest.skip(f"PEFT {peft} LoRA layer is not importable")
    templates, z = _ZOO
    child = "Linear" if peft == "dense" else "Linear4bit"
    forced = str(flag) != "0"
    source = textwrap.dedent(inspect.getsource(original))
    source = source.replace("result = result.clone()", "")
    source = z["_patch_lora_update"](source)
    source = source.replace("def forward", "def unsloth_forward", 1)
    source = z["_patch_lora_input_cast"](source, force_float32 = forced)
    check = "self._check_forward_args(x, *args, **kwargs)"
    inline = "4bit" in child.lower() and check in source
    source = source.replace(
        check,
        "if not x.is_floating_point(): x = _unsloth_lora_integer_input(self, x)" if inline else "",
    )
    source = z["_patch_lora_base_layer_input_cast"](source)
    template = templates[
        "COMPILED_LORA_FORWARD_forced_float32" if forced else "COMPILED_LORA_FORWARD"
    ]
    full = template + source
    filename = f"<unsloth_compiled_cache {child}_peft_forward {flag}>"
    linecache.cache[filename] = (len(full), None, full.splitlines(True), filename)
    scope = dict(sys.modules[original.__module__].__dict__)
    exec(compile(full, filename, "exec"), scope)
    forward = scope["unsloth_forward"]
    forward._unsloth_force_float32 = flag
    forward._unsloth_original_forward = original
    return forward


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
        adapters = ("default",),
        active = None,
    ):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.base_layer = nn.Linear(in_features, out_features, bias = False).to(base_dtype)
        with torch.no_grad():
            self.base_layer.weight.copy_(torch.randn(out_features, in_features, generator = g) * 0.05)
        self.base_layer.weight.requires_grad_(False)
        self.lora_A = nn.ModuleDict({a: nn.Linear(in_features, r, bias = False) for a in adapters})
        self.lora_B = nn.ModuleDict({a: nn.Linear(r, out_features, bias = False) for a in adapters})
        with torch.no_grad():
            for a in adapters:
                self.lora_A[a].weight.copy_(torch.randn(r, in_features, generator = g) * 0.05)
                self.lora_B[a].weight.copy_(torch.randn(out_features, r, generator = g) * 0.05)
        drop = (lambda: nn.Dropout(dropout)) if dropout > 0 else nn.Identity
        self.lora_dropout = nn.ModuleDict({a: drop() for a in adapters})
        self.scaling = {a: scaling * (1 + i / 4) for i, a in enumerate(adapters)}
        self.lora_variant = {}
        self.disable_adapters = False
        self.merged = False
        self.active_adapters = list(adapters if active is None else active)
        self._unsloth_tp_style = tpn.COLWISE

    def _cast_input_dtype(self, x, dtype):
        return x if x.dtype == dtype else x.to(dtype)


def _with_forward(flag, peft = "dense"):
    """A layer class whose forward is Unsloth's generated LoRA forward built under `flag` over PEFT's `peft`
    ("dense" Linear or "bnb" Linear4bit) forward, or PEFT's own forward (flag None)."""
    if flag is None:
        return _LoraProj
    return type("CompiledLoraProj", (_LoraProj,), {"forward": _generated_forward(flag, peft)})


def _peft_forward(proj, x):
    """PEFT's forward (no Unsloth compile): x cast to the LoRA weights' dtype, product in float32."""
    A = proj.lora_A["default"].weight
    B = proj.lora_B["default"].weight
    result = torch.nn.functional.linear(x, proj.base_layer.weight)
    xd = proj.lora_dropout["default"](x.to(A.dtype))
    return result + (
        torch.nn.functional.linear(torch.nn.functional.linear(xd, A), B) * proj.scaling["default"]
    ).to(result.dtype)


def _guarded_zoo():
    """True when zoo casts a dense single vanilla adapter's result and x only for variants / several adapters."""
    if _ZOO is None:
        return False
    line = "x = self._cast_input_dtype(x, lora_A.weight.dtype)"
    return "len(self.active_adapters) > 1" in _ZOO[1]["_patch_lora_input_cast"](
        "lora_forward(\n" + line, force_float32 = False
    )


def _run(
    fn,
    proj,
    x,
    g,
    seed = 3,
):
    params = [m.weight for name in proj.lora_A for m in (proj.lora_A[name], proj.lora_B[name])]
    for p in params:
        p.grad = None
    torch.manual_seed(seed)  # the same dropout mask in both arms
    y = fn(proj, x)
    y.backward(g)
    return (y.detach(), *(p.grad.clone() for p in params))


def _bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def _inputs(dtype):
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(2, 16, 64, generator = g) * 0.5).to(dtype)
    # Gemma-4 upper-layer grads: float16 flushes some low-rank grad to zero; float32 keeps it.
    grad = (torch.randn(2, 16, 48, generator = g) * 3e-6).to(dtype)
    return x, grad


def _assert_same(ref, got):
    assert len(ref) == len(got)
    names = ["output"] + [f"lora {i // 2} {'AB'[i % 2]} grad" for i in range(len(ref) - 1)]
    for name, a, b in zip(names, ref, got):
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
    **kwargs,
):
    proj = _with_forward(flag, peft)(dtype, scaling = scaling, dropout = dropout, **kwargs)
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


@pytest.mark.parametrize("scaling", [2.0, 1.7])
@pytest.mark.parametrize("dropout", [0.0, 0.25])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tp_lora_several_active_adapters_match_one_gpu_forward(dtype, dropout, scaling):
    """With several active adapters, zoo's guard casts result and x to the float32 LoRA dtype."""
    _check("0", dtype, "dense", scaling, dropout, active = ("default", "other"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tp_lora_two_adapters_on_one_layer_match_one_gpu_forward(dtype):
    """Adapters accumulate in LoRA dtype, cast to base once; tp must not round between them."""
    _check("0", dtype, "dense", adapters = ("default", "second"))


def test_dense_mode_follows_zoo_input_cast():
    """Dense Linear without forced float32 follows zoo's generated cast line: always cast ("lora"), or in newer zoo
    only for a variant or several active adapters ("lora_guarded"); bitsandbytes keeps PEFT's x cast."""
    want = "lora_guarded" if _guarded_zoo() else "lora"
    proj = _with_forward("0", "dense")(torch.bfloat16)
    assert tpn._one_gpu_lora_mode(proj) == want
    if _peft_forward_of("bnb") is not None:
        proj = _with_forward("0", "bnb")(torch.bfloat16)
        assert tpn._one_gpu_lora_mode(proj) == "output"


def test_single_vanilla_adapter_keeps_the_base_dtype_under_guarded_zoo():
    """Guard case: bf16 base, fp32 LoRA; one-GPU runs bf16 arithmetic, so fp32 tp would differ."""
    if not _guarded_zoo():
        pytest.skip("this unsloth_zoo casts every dense adapter")
    proj = _with_forward("0", "dense")(torch.bfloat16)
    x, grad = _inputs(torch.bfloat16)
    ref = _run(lambda p, xx: p.forward(xx), proj, x, grad)
    proj.__dict__["_unsloth_tp_lora_mode"] = "lora"  # the always-cast arithmetic
    cast = _run(tpn._tp_projection_compute, proj, x, grad)
    assert any(bool((_bits(a) != _bits(b)).any()) for a, b in zip(ref, cast))
    proj.__dict__.pop("_unsloth_tp_lora_mode")
    _assert_same(ref, _run(tpn._tp_projection_compute, proj, x, grad))


def test_unknown_cast_guard_is_not_guessed():
    """A generated cast line under a guard tp does not know: PEFT's float32 fallback, never a guess."""
    known = tpn._generated_lora_cast_mode(
        "    if not torch.is_autocast_enabled(): result, x = result.to(lora_A.weight.dtype), "
        "x.to(lora_A.weight.dtype)\n"
    )
    assert known == "lora"
    unknown = tpn._generated_lora_cast_mode(
        "    if not torch.is_autocast_enabled() and self.foo: result, x = result.to(lora_A.weight.dtype), "
        "x.to(lora_A.weight.dtype)\n"
    )
    assert unknown is None
    assert tpn._generated_lora_cast_mode("x = x.to(lora_A.weight.dtype)\n") is None


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


@pytest.mark.parametrize("peft", ["dense", "bnb"])
@pytest.mark.parametrize("flag", ["1", "0"])
def test_old_torch_compiled_branch_matches_zoo(monkeypatch, flag, peft):
    """torch < 2.10 under torch.compile: zoo skips addmm (unsloth#3271); the tp update must take the same branch."""
    proj = _with_forward(flag, peft)(torch.float16, scaling = 1.7)
    scope = type(proj).forward.__globals__
    x, grad = _inputs(torch.float16)
    monkeypatch.setitem(scope, "torch_lora_no_compiled_addmm", True)
    monkeypatch.setitem(scope, "torch_is_compiling", lambda: True)
    ref = _run(lambda p, xx: p.forward(xx), proj, x, grad)
    monkeypatch.setattr(tpn, "_TP_LORA_NO_COMPILED_ADDMM", True, raising = False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
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
    """Forced float32 one-GPU lora_A grad loses entries a float32 product keeps; tp must match."""
    proj = _with_forward("1")(torch.float16)
    x, grad = _inputs(torch.float16)
    _, a16, _ = _run(lambda p, xx: p.forward(xx), proj, x, grad)
    _, a32, _ = _run(_peft_forward, proj, x, grad)
    rel = float((a16 - a32).norm() / a32.norm())
    assert rel > 1e-2, rel  # a real difference, not rounding noise


def _dispatched(
    first,
    second,
    peft = "dense",
):
    """zoo's real _install_lora_forward_variant: a class built under `first`, then another family's load under
    `second` (unsloth_zoo compiler patch_lora_forwards), so the class forward is the `unsloth_forward` dispatcher
    over both generated variants."""
    if _ZOO is None or "_install_lora_forward_variant" not in _ZOO[1]:
        pytest.skip("this unsloth_zoo has no per-flag LoRA forward dispatcher")
    cls = _with_forward(first, peft)
    _ZOO[1]["_install_lora_forward_variant"](cls, cls.forward, _generated_forward(second, peft))
    assert set(cls.forward._unsloth_variants) == {first, second}
    return cls


@pytest.mark.parametrize("order", [("1", "0"), ("0", "1")])
@pytest.mark.parametrize("stamp", ["0", "1", None])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dispatcher_resolves_the_stamped_variant(dtype, stamp, order):
    """Zoo's dispatcher has no cast line of its own, so tp must read the variant it picks (layer
    stamp, else last build) and match that variant's arithmetic bit for bit."""
    picked = order[1] if stamp is None else stamp
    if picked == "1" and dtype is torch.bfloat16:
        pytest.skip("forced float32 keeps float16 base weights")
    proj = _dispatched(*order)(dtype)
    if stamp is not None:
        proj.__dict__["_unsloth_force_float32"] = stamp
    want = "float16" if picked == "1" else ("lora_guarded" if _guarded_zoo() else "lora")
    assert tpn._one_gpu_lora_mode(proj) == want
    x, grad = _inputs(dtype)
    ref = _run(lambda p, xx: p.forward(xx), proj, x, grad)
    _assert_same(ref, _run(tpn._tp_projection_compute, proj, x, grad))


def test_dispatcher_reads_the_base_layer_stamp():
    """Unstamped wrapper: the dispatcher reads its base layer's stamp, and tp must read it too."""
    proj = _dispatched("0", "1")(torch.float16)
    proj.base_layer.__dict__["_unsloth_force_float32"] = "0"
    assert tpn._one_gpu_lora_mode(proj) == ("lora_guarded" if _guarded_zoo() else "lora")
    proj.base_layer.__dict__["_unsloth_force_float32"] = "1"
    assert tpn._one_gpu_lora_mode(proj) == "float16"
