"""CPU regressions for the guarded SentenceTransformer LoRA fast path.

The kernel is loaded under a private package with its CUDA-facing dependencies
stubbed. This exercises the real Python autograd implementation without running
Unsloth's package initialization or requiring a GPU.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import types

import pytest


torch = pytest.importorskip("torch")
F = torch.nn.functional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAST_LORA_PATH = _REPO_ROOT / "unsloth" / "kernels" / "fast_lora.py"


@pytest.fixture(scope = "module")
def fast_lora_module():
    package_name = "_unsloth_fast_lora_st_test"
    kernels_name = f"{package_name}.kernels"
    utils_name = f"{kernels_name}.utils"
    swiglu_name = f"{kernels_name}.swiglu"
    geglu_name = f"{kernels_name}.geglu"
    module_name = f"{kernels_name}.fast_lora"
    shadowed_names = (
        package_name,
        kernels_name,
        utils_name,
        swiglu_name,
        geglu_name,
        module_name,
    )
    saved_modules = {name: sys.modules.get(name) for name in shadowed_names}

    package = types.ModuleType(package_name)
    package.__path__ = []
    kernels = types.ModuleType(kernels_name)
    kernels.__path__ = []
    utils = types.ModuleType(utils_name)
    swiglu = types.ModuleType(swiglu_name)
    geglu = types.ModuleType(geglu_name)

    def identity_decorator(function):
        return function

    def matmul_lora(x, weight, _quant_state, lora_a, lora_b, scaling):
        result = F.linear(x, weight)
        if lora_a is not None and lora_b is not None:
            result = result + F.linear(F.linear(x, lora_a), lora_b) * scaling
        return result

    utils._maybe_fake_quantize_activations = lambda x, _module: x
    utils.fast_dequantize = lambda weight, _quant_state: weight
    utils.QUANT_STATE = lambda _weight: None
    utils.get_lora_parameters = lambda _module: None
    utils.get_lora_parameters_bias = lambda _module: None
    utils.matmul_lora = matmul_lora
    utils.torch_amp_custom_fwd = identity_decorator
    utils.torch_amp_custom_bwd = identity_decorator
    swiglu.swiglu_fg_kernel = lambda *_args, **_kwargs: None
    swiglu.swiglu_DWf_DW_dfg_kernel = lambda *_args, **_kwargs: None
    geglu.geglu_exact_forward_kernel = lambda *_args, **_kwargs: None
    geglu.geglu_exact_backward_kernel = lambda *_args, **_kwargs: None
    geglu.geglu_approx_forward_kernel = lambda *_args, **_kwargs: None
    geglu.geglu_approx_backward_kernel = lambda *_args, **_kwargs: None

    sys.modules[package_name] = package
    sys.modules[kernels_name] = kernels
    sys.modules[utils_name] = utils
    sys.modules[swiglu_name] = swiglu
    sys.modules[geglu_name] = geglu

    spec = importlib.util.spec_from_file_location(module_name, _FAST_LORA_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    try:
        yield module
    finally:
        for name, saved in saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved


def _input_values(kind: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7319)
    if kind == "2d":
        return torch.randn((5, 7), generator = generator)
    if kind == "3d":
        return torch.randn((2, 3, 7), generator = generator)
    if kind == "noncontiguous":
        values = torch.randn((4, 3, 7), generator = generator).transpose(0, 1)
        assert not values.is_contiguous()
        return values
    raise AssertionError(f"unknown input kind: {kind}")


def _leaf_copy(value: torch.Tensor, *, requires_grad: bool = True) -> torch.Tensor:
    return value.detach().clone(memory_format = torch.preserve_format).requires_grad_(requires_grad)


@pytest.mark.parametrize("input_kind", ("2d", "3d", "noncontiguous"))
def test_lora_w_matches_dense_reference_outputs_and_gradients(fast_lora_module, input_kind):
    generator = torch.Generator().manual_seed(20260710)
    input_values = _input_values(input_kind)
    weight_values = torch.randn((11, 7), generator = generator)
    lora_a_values = torch.randn((3, 7), generator = generator)
    lora_b_values = torch.randn((11, 3), generator = generator)
    scale = 0.375

    fast_x = _leaf_copy(input_values)
    fast_weight = _leaf_copy(weight_values, requires_grad = False)
    fast_a = _leaf_copy(lora_a_values)
    fast_b = _leaf_copy(lora_b_values)
    reference_x = _leaf_copy(input_values)
    reference_weight = _leaf_copy(weight_values, requires_grad = False)
    reference_a = _leaf_copy(lora_a_values)
    reference_b = _leaf_copy(lora_b_values)

    fast_output = fast_lora_module.LoRA_W.apply(fast_x, fast_weight, None, fast_a, fast_b, scale)
    reference_output = F.linear(reference_x, reference_weight) + (
        F.linear(F.linear(reference_x, reference_a), reference_b) * scale
    )
    output_gradient = torch.randn(fast_output.shape, generator = generator, dtype = fast_output.dtype)
    (fast_output * output_gradient).sum().backward()
    (reference_output * output_gradient).sum().backward()

    torch.testing.assert_close(fast_output, reference_output, rtol = 0, atol = 0)
    torch.testing.assert_close(fast_x.grad, reference_x.grad, rtol = 1e-6, atol = 1e-6)
    torch.testing.assert_close(fast_a.grad, reference_a.grad, rtol = 1e-6, atol = 1e-6)
    torch.testing.assert_close(fast_b.grad, reference_b.grad, rtol = 1e-6, atol = 1e-6)


class _CountingLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, *, bias):
        super().__init__(in_features, out_features, bias = bias)
        self.forward_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        return super().forward(x)


class _ToyLoraLinear(torch.nn.Module):
    def __init__(self, fast_forward):
        super().__init__()
        self._fast_forward = fast_forward
        self.base_layer = _CountingLinear(7, 11, bias = True)
        self.base_layer.weight.requires_grad_(False)
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(7, 3, bias = False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 11, bias = False)})
        self.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Identity()})
        self.scaling = {"default": 0.375}
        self.use_dora = {"default": False}
        self.lora_bias = {"default": False}
        self.active_adapters = ["default"]
        self.disable_adapters = False
        self.merged = False
        self.unmerge_calls = 0
        self.mixed_batch_calls = []

    def _check_forward_args(self, _x, *_args, **_kwargs):
        return None

    def _mixed_batch_forward(self, x, *args, adapter_names, **kwargs):
        self.mixed_batch_calls.append((tuple(adapter_names), args, kwargs))
        return self.base_layer(x) + 17.0

    def unmerge(self):
        self.unmerge_calls += 1
        self.merged = False

    def forward(self, x, *args, **kwargs):
        return self._fast_forward(self, x, *args, **kwargs)


def _reference_lora_forward(module: _ToyLoraLinear, x: torch.Tensor) -> torch.Tensor:
    result = module.base_layer(x).clone()
    for adapter in module.active_adapters:
        if adapter not in module.lora_A:
            continue
        adapted_x = module.lora_dropout[adapter](x)
        result = result + (
            module.lora_B[adapter](module.lora_A[adapter](adapted_x)) * module.scaling[adapter]
        )
    return result


@pytest.mark.parametrize("input_kind", ("2d", "3d", "noncontiguous"))
def test_sentence_transformer_fast_forward_matches_peft_reference_and_bias_grad(
    fast_lora_module, input_kind
):
    torch.manual_seed(991)
    fast_module = _ToyLoraLinear(fast_lora_module.fast_lora_forward_st)
    reference_module = copy.deepcopy(fast_module)
    input_values = _input_values(input_kind)
    fast_x = _leaf_copy(input_values)
    reference_x = _leaf_copy(input_values)

    fast_output = fast_module(fast_x)
    reference_output = _reference_lora_forward(reference_module, reference_x)
    output_gradient = torch.linspace(
        -0.75, 0.5, fast_output.numel(), dtype = fast_output.dtype
    ).reshape(fast_output.shape)
    (fast_output * output_gradient).sum().backward()
    (reference_output * output_gradient).sum().backward()

    assert fast_module.base_layer.forward_calls == 0
    assert reference_module.base_layer.forward_calls == 1
    torch.testing.assert_close(fast_output, reference_output, rtol = 1e-6, atol = 1e-6)
    torch.testing.assert_close(fast_x.grad, reference_x.grad, rtol = 1e-6, atol = 1e-6)
    torch.testing.assert_close(
        fast_module.lora_A["default"].weight.grad,
        reference_module.lora_A["default"].weight.grad,
        rtol = 1e-6,
        atol = 1e-6,
    )
    torch.testing.assert_close(
        fast_module.lora_B["default"].weight.grad,
        reference_module.lora_B["default"].weight.grad,
        rtol = 1e-6,
        atol = 1e-6,
    )
    torch.testing.assert_close(
        fast_module.base_layer.bias.grad,
        reference_module.base_layer.bias.grad,
        rtol = 0,
        atol = 0,
    )


def _assert_standard_fallback_matches_reference(fast_lora_module, configure):
    torch.manual_seed(127)
    module = _ToyLoraLinear(fast_lora_module.fast_lora_forward_st)
    configure(module)
    reference = copy.deepcopy(module)
    # Keep dropout deterministic while retaining a non-Identity dropout module,
    # which is the condition that must disable the custom autograd fast path.
    module.eval()
    reference.eval()
    for dropout in module.lora_dropout.values():
        dropout.eval()
    for dropout in reference.lora_dropout.values():
        dropout.eval()
    values = _input_values("3d")

    if module.disable_adapters or module.merged:
        expected = reference.base_layer(values)
    else:
        expected = _reference_lora_forward(reference, values)
    actual = module(values)

    assert module.base_layer.forward_calls == 1
    torch.testing.assert_close(actual, expected, rtol = 0, atol = 0)
    return module


@pytest.mark.parametrize(
    "configure",
    (
        pytest.param(
            lambda module: module.lora_dropout.__setitem__("default", torch.nn.Dropout(p = 0.25)),
            id = "dropout",
        ),
        pytest.param(
            lambda module: setattr(module, "disable_adapters", True),
            id = "disabled",
        ),
        pytest.param(
            lambda module: setattr(module, "merged", True),
            id = "merged",
        ),
    ),
)
def test_sentence_transformer_fast_forward_falls_back_for_dynamic_state(
    fast_lora_module, configure
):
    _assert_standard_fallback_matches_reference(fast_lora_module, configure)


def test_disabled_merged_adapter_is_unmerged_before_base_forward(fast_lora_module):
    def configure(module):
        module.disable_adapters = True
        module.merged = True

    module = _assert_standard_fallback_matches_reference(fast_lora_module, configure)
    assert module.unmerge_calls == 1
    assert module.merged is False


def test_multiple_active_adapters_use_standard_fallback(fast_lora_module):
    def configure(module):
        module.lora_A["secondary"] = torch.nn.Linear(7, 2, bias = False)
        module.lora_B["secondary"] = torch.nn.Linear(2, 11, bias = False)
        module.lora_dropout["secondary"] = torch.nn.Identity()
        module.scaling["secondary"] = 0.25
        module.use_dora["secondary"] = False
        module.lora_bias["secondary"] = False
        module.active_adapters = ["default", "secondary"]

    _assert_standard_fallback_matches_reference(fast_lora_module, configure)


def test_adapter_names_delegate_to_mixed_batch_forward(fast_lora_module):
    torch.manual_seed(81)
    module = _ToyLoraLinear(fast_lora_module.fast_lora_forward_st)
    values = _input_values("2d")

    result = module(values, adapter_names = ["default"] * values.shape[0])

    assert module.base_layer.forward_calls == 1
    assert module.mixed_batch_calls == [(("default",) * values.shape[0], (), {})]
    torch.testing.assert_close(
        result, F.linear(values, module.base_layer.weight, module.base_layer.bias) + 17.0
    )
