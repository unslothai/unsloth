"""FSDP2 all-gathers a bitsandbytes 4-bit weight into a plain Parameter without `quant_state`. The fused
LoRA kernels must then read the Linear4bit module's own QuantState, not treat the packed (N, 1) bytes as
a dense weight."""

import ast
from pathlib import Path

import torch

_SOURCE = Path(__file__).parents[2] / "unsloth" / "kernels" / "utils.py"
_HELPERS = ("_packed_base", "_is_packed_state", "_base_quant_state")


def _load_function(name):
    # Extract functions from kernels/utils.py without importing unsloth (which needs a GPU).
    tree = ast.parse(_SOURCE.read_text(encoding = "utf-8"))
    funcs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in (name, *_HELPERS)
    ]
    assert [f.name for f in funcs].count(name) == 1, (name, funcs)
    namespace = {"torch_Tensor": torch.Tensor, "_FP8_WEIGHT_DTYPES": ()}
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
    base_layer.weight = torch.nn.Parameter(torch.zeros(512, 1, dtype = torch.float16), requires_grad = False)
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
