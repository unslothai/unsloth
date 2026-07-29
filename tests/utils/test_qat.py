from unsloth import FastLanguageModel

from typing import Dict

import pytest
import torch

try:
    from torchao.quantization.qat import FakeQuantizedLinear
    from torchao.quantization.qat.fake_quantizer import (
        FakeQuantizerBase,
        Float8FakeQuantizer,
        Int4WeightFakeQuantizer,
        IntxFakeQuantizer,
    )
except ImportError:
    print(
        "Missing torchao import, please install or upgrade torchao with: pip install 'torchao>=0.15.0'"
    )


class _CountingFakeQuantizer(torch.nn.Module):
    """Fake quantizer that counts how many times it was called."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.count += 1
        return x


def _get_model(qat_scheme: str, full_finetuning: bool):
    """Return (model, tokenizer) configured for QAT; LoRA model when full_finetuning is False."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-1.7B",
        load_in_4bit = False,
        full_finetuning = full_finetuning,
        qat_scheme = qat_scheme if full_finetuning else None,
    )
    if not full_finetuning:
        model = FastLanguageModel.get_peft_model(
            model,
            qat_scheme = qat_scheme,
        )
    return model, tokenizer


def _test_linear_is_fake_quantized(linear: torch.nn.Linear, qat_scheme: str):
    """Verify the linear contains fake quantizers matching `qat_scheme`."""
    weight_only = False
    if qat_scheme == "fp8-int4":
        act_fq_class = Float8FakeQuantizer
        weight_fq_class = Int4WeightFakeQuantizer
        min_in_features = 128
    elif qat_scheme == "fp8-fp8":
        act_fq_class = Float8FakeQuantizer
        weight_fq_class = Float8FakeQuantizer
        min_in_features = -1
    elif qat_scheme == "int8":
        act_fq_class = None
        weight_fq_class = IntxFakeQuantizer
        min_in_features = 128
        weight_only = True
    elif qat_scheme == "cactus":
        act_fq_class = None
        weight_fq_class = IntxFakeQuantizer
        min_in_features = 32
        weight_only = True
    else:
        raise ValueError(f"Unknown qat_scheme: {qat_scheme}")

    # Check base layer activations and weights.
    base_layer = getattr(linear, "base_layer", linear)
    if base_layer.in_features >= min_in_features:
        assert isinstance(base_layer, FakeQuantizedLinear)
        if not weight_only:
            assert isinstance(base_layer.activation_fake_quantizer, act_fq_class)
        assert isinstance(base_layer.weight_fake_quantizer, weight_fq_class)

    # Check lora A and B (full_finetuning=False only).
    if hasattr(linear, "lora_A") and hasattr(linear, "lora_B"):
        lora_A = linear.lora_A.default
        lora_B = linear.lora_B.default
        if lora_A.in_features >= min_in_features:
            assert isinstance(lora_A, FakeQuantizedLinear)
            if not weight_only:
                assert isinstance(lora_A.activation_fake_quantizer, act_fq_class)
            assert isinstance(lora_A.weight_fake_quantizer, weight_fq_class)
        if lora_B.in_features >= min_in_features:
            assert isinstance(lora_B, FakeQuantizedLinear)
            if not weight_only:
                assert isinstance(lora_B.activation_fake_quantizer, act_fq_class)
            assert isinstance(lora_B.weight_fake_quantizer, weight_fq_class)


def _test_fake_quantizers_are_called(
    model: torch.nn.Module, example_inputs: Dict, full_finetuning: bool, qat_scheme: str
):
    """Verify the fake quantizers are actually called during a forward pass."""
    weight_only = qat_scheme in ["int8", "cactus"]

    def _swap_fake_quantizers(model: torch.nn.Module):
        for name, child in model.named_children():
            if isinstance(child, FakeQuantizerBase):
                setattr(model, name, _CountingFakeQuantizer())

    def _assert_fake_quantizers_are_called(model: torch.nn.Module):
        for name, child in model.named_children():
            if full_finetuning:
                if isinstance(child, FakeQuantizedLinear):
                    if not weight_only:
                        assert child.activation_fake_quantizer.count == 1
                    assert child.weight_fake_quantizer.count == 1
            else:
                # LoRA fake-quantizes input activations once per block:
                # self_attn via q_proj, mlp via gate_proj.
                if name == "self_attn":
                    base_layer = child.q_proj.base_layer
                    if not weight_only:
                        assert hasattr(base_layer, "activation_fake_quantizer")
                        assert base_layer.activation_fake_quantizer.count == 1
                elif name == "mlp":
                    base_layer = child.gate_proj.base_layer
                    if not weight_only:
                        assert hasattr(base_layer, "activation_fake_quantizer")
                        assert base_layer.activation_fake_quantizer.count == 1
                elif isinstance(child, FakeQuantizedLinear):
                    # Weight fake quantizers must always be called.
                    assert child.weight_fake_quantizer.count == 1

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        pytest.skip("No GPU available")
    for k, v in example_inputs.items():
        example_inputs[k] = v.to(device)
    model.apply(_swap_fake_quantizers)
    model(**example_inputs)
    model.apply(_assert_fake_quantizers_are_called)


def _test_model_fake_quantize(qat_scheme: str, full_finetuning: bool):
    """All linear layers in the model are fake quantized per `qat_scheme`."""
    model, tokenizer = _get_model(qat_scheme, full_finetuning)
    if full_finetuning:
        model = model.model
    else:
        model = model.base_model.model.model
    for layer in model.layers:
        _test_linear_is_fake_quantized(layer.self_attn.q_proj, qat_scheme)
        _test_linear_is_fake_quantized(layer.self_attn.k_proj, qat_scheme)
        _test_linear_is_fake_quantized(layer.self_attn.v_proj, qat_scheme)
        _test_linear_is_fake_quantized(layer.mlp.gate_proj, qat_scheme)
        _test_linear_is_fake_quantized(layer.mlp.up_proj, qat_scheme)
        _test_linear_is_fake_quantized(layer.mlp.down_proj, qat_scheme)
    inputs = tokenizer("How are you?", return_tensors = "pt")
    _test_fake_quantizers_are_called(model, inputs, full_finetuning, qat_scheme)


# TODO: there are bad interactions across tests right now, need to figure out
# how to disable model caching before re-enabling this test
@pytest.mark.parametrize("qat_scheme", ["fp8-int4", "fp8-fp8", "int8", "cactus"])
def _test_full_model_fake_quantize(qat_scheme: str):
    _test_model_fake_quantize(qat_scheme, full_finetuning = True)


@pytest.mark.parametrize("qat_scheme", ["fp8-int4", "fp8-fp8", "int8", "cactus"])
def test_lora_model_fake_quantize(qat_scheme: str):
    _test_model_fake_quantize(qat_scheme, full_finetuning = False)
