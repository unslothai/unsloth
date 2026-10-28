# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FSDP2 all-gathers a bitsandbytes 4-bit weight into a plain Parameter without `quant_state`. The fused
LoRA kernels must then read the Linear4bit module's own QuantState, not treat the packed (N, 1) bytes as
a dense weight."""

import ast
from pathlib import Path

import torch

_SOURCE = Path(__file__).parents[2] / "unsloth" / "kernels" / "utils.py"
_HELPERS = ("_packed_base", "_is_packed_state", "_base_quant_state")


def _load_function(name, fsdp2_qlora_prepared = True):
    # Extract functions from kernels/utils.py without importing unsloth (which needs a GPU).
    tree = ast.parse(_SOURCE.read_text(encoding = "utf-8"))
    funcs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in (name, *_HELPERS)
    ]
    assert [f.name for f in funcs].count(name) == 1, (name, funcs)
    namespace = {
        "torch_Tensor": torch.Tensor,
        "_FP8_WEIGHT_DTYPES": (),
        "_FSDP2_QLORA_PREPARED": fsdp2_qlora_prepared,
    }
    module = ast.Module(body = funcs, type_ignores = [])
    ast.fix_missing_locations(module)
    exec(compile(module, str(_SOURCE), "exec"), namespace)
    return namespace[name]


class _Obj:
    pass


class Linear4bit(torch.nn.Module):
    # Stands in for bitsandbytes.nn.Linear4bit, matched by class name.
    pass


class _LoRA:
    def __init__(self, base_layer):
        self.base_layer = base_layer
        self.disable_adapters = False
        self.merged = False
        self.active_adapters = ["default"]
        a, b = _Obj(), _Obj()
        a.weight, b.weight = torch.zeros(4, 64), torch.zeros(32, 4)
        self.lora_A = {"default": a}
        self.lora_B = {"default": b}
        self.scaling = {"default": 2.0}
        self.lora_bias = {}


def _fsdp2_unsharded_4bit(base_cls = Linear4bit):
    base_layer = base_cls()
    state = _Obj()
    state.dtype = torch.float16
    base_layer.quant_state = state
    # The all-gathered packed bytes: (32 * 64 / 4, 1) in fp16 storage, a plain Parameter with no quant_state.
    base_layer.weight = torch.nn.Parameter(
        torch.zeros(512, 1, dtype = torch.float16), requires_grad = False
    )
    base_layer.bias = None
    return _LoRA(base_layer), state


def test_get_lora_parameters_falls_back_to_module_quant_state():
    get_lora_parameters = _load_function("get_lora_parameters")
    proj, state = _fsdp2_unsharded_4bit()
    W, W_quant = get_lora_parameters(proj)[:2]
    assert W is proj.base_layer.weight
    assert W_quant is state


def test_get_lora_parameters_bias_falls_back_to_module_quant_state():
    get_lora_parameters_bias = _load_function("get_lora_parameters_bias")
    proj, state = _fsdp2_unsharded_4bit()
    assert get_lora_parameters_bias(proj)[1] is state


def test_weight_quant_state_still_wins():
    get_lora_parameters = _load_function("get_lora_parameters")
    proj, state = _fsdp2_unsharded_4bit()
    own = _Obj()
    proj.base_layer.weight.quant_state = own
    assert get_lora_parameters(proj)[1] is own


def test_no_fallback_for_dense_or_non_bnb_weights():
    get_lora_parameters = _load_function("get_lora_parameters")
    # A dense weight is never paired with a stray module quant_state.
    proj, _ = _fsdp2_unsharded_4bit()
    proj.base_layer.weight = torch.nn.Parameter(torch.zeros(32, 64), requires_grad = False)
    assert get_lora_parameters(proj)[1] is None
    # Nor is a module that is not a bitsandbytes Linear4bit.
    proj, _ = _fsdp2_unsharded_4bit(base_cls = torch.nn.Module)
    assert get_lora_parameters(proj)[1] is None


def test_dense_weight_never_takes_the_module_getattr_path():
    # Every 16-bit LoRA forward (training and decode) asks for the quant state; a dense weight must
    # not pay for nn.Module.__getattr__ raising and catching AttributeError on the base layer.
    get_lora_parameters = _load_function("get_lora_parameters")
    lookups = []

    class _CountingLinear(torch.nn.Linear):
        def __getattr__(self, name):
            lookups.append(name)
            return super().__getattr__(name)

    base_layer = _CountingLinear(64, 32, bias = False)
    proj = _LoRA(base_layer)
    lookups.clear()
    assert get_lora_parameters(proj)[1] is None
    assert "quant_state" not in lookups, lookups


def test_fallback_reads_no_process_global():
    # The fallback is gated per weight on the packed (N, 1) shape, never on a module-global flag: a
    # global read inside a compiled LoRA forward is a dynamo guard, so preparing one FSDP2 QLoRA model
    # recompiled every compiled dense LoRA forward (#13880 review).
    for name in ("get_lora_parameters", "get_lora_parameters_bias"):
        fn = _load_function(name)
        assert "_FSDP2_QLORA_PREPARED" not in fn.__code__.co_names, name
        proj, state = _fsdp2_unsharded_4bit()
        assert fn(proj)[1] is state


def test_compiled_dense_lookup_does_not_recompile_when_flag_flips():
    # The review's repro: compile a dense LoRA lookup, flip the old global, call again: one graph.
    import torch._dynamo

    get_lora_parameters = _load_function("get_lora_parameters", fsdp2_qlora_prepared = False)
    base_layer = torch.nn.Linear(64, 32, bias = False)
    proj = _LoRA(base_layer)
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm

    torch._dynamo.reset()

    @torch.compile(backend = backend, fullgraph = True)
    def lookup(x):
        W, W_quant, A, B, s = get_lora_parameters(proj)
        return x @ W.t() + (x @ A.t()) @ B.t() * s

    x = torch.randn(2, 64)
    out = lookup(x)
    get_lora_parameters.__globals__["_FSDP2_QLORA_PREPARED"] = True
    assert torch.equal(lookup(x), out)
    assert len(graphs) == 1, len(graphs)
    torch._dynamo.reset()
