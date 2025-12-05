from unsloth import FastLanguageModel

from typing import Dict

import pytest
import torch
from torchao.quantization.qat import FakeQuantizedLinear
from torchao.quantization.qat.fake_quantizer import (
    FakeQuantizerBase,
    Float8FakeQuantizer,
    Int4WeightPreshuffledFakeQuantizer,
)


class _CountingFakeQuantizer(torch.nn.Module):
    """
    Dummy fake quantizer that counts the number of times it has been called.
    """

    def __init__(self):
        super().__init__()
        self.count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.count += 1
        return x


def _get_model(qat_scheme: str, full_finetuning: bool):
    """
    Return a 2-tuple of (model, tokenizer), where the model has been configured
    to use QAT. If `full_finetuning` is False, return the PEFT (LoRA) model.
    """
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
    """
    Verify that the given linear contains fake quantizers according to the `qat_scheme`.
    """
    if qat_scheme == "fp8-int4":
        act_fq_class = Float8FakeQuantizer
        weight_fq_class = Int4WeightPreshuffledFakeQuantizer
        min_in_features = 128
    elif qat_scheme == "fp8-fp8":
        act_fq_class = Float8FakeQuantizer
        weight_fq_class = Float8FakeQuantizer
        min_in_features = -1
    else:
        raise ValueError(f"Unknown qat_scheme: {qat_scheme}")

    # Check base layer activations and weights
    base_layer = getattr(linear, "base_layer", linear)
    if base_layer.in_features >= min_in_features:
        assert isinstance(base_layer, FakeQuantizedLinear)
        assert isinstance(base_layer.activation_fake_quantizer, act_fq_class)
        assert isinstance(base_layer.weight_fake_quantizer, weight_fq_class)

    # Check lora A and B (only for full_finetuning=False)
    if hasattr(linear, "lora_A") and hasattr(linear, "lora_B"):
        lora_A = linear.lora_A.default
        lora_B = linear.lora_B.default
        if lora_A.in_features >= min_in_features:
            assert isinstance(lora_A, FakeQuantizedLinear)
            assert isinstance(lora_A.activation_fake_quantizer, act_fq_class)
            assert isinstance(lora_A.weight_fake_quantizer, weight_fq_class)
        if lora_B.in_features >= min_in_features:
            assert isinstance(lora_B, FakeQuantizedLinear)
            assert isinstance(lora_B.activation_fake_quantizer, act_fq_class)
            assert isinstance(lora_B.weight_fake_quantizer, weight_fq_class)


def _test_fake_quantizers_are_called(
    model: torch.nn.Module,
    example_inputs: Dict,
    full_finetuning: bool,
):
    """
    Verify that the fake quantizers are actually called when the model is called.
    """

    def _swap_fake_quantizers(model: torch.nn.Module):
        for name, child in model.named_children():
            if isinstance(child, FakeQuantizerBase):
                setattr(model, name, _CountingFakeQuantizer())

    def _assert_fake_quantizers_are_called(model: torch.nn.Module):
        for name, child in model.named_children():
            if full_finetuning:
                if isinstance(child, FakeQuantizedLinear):
                    assert child.activation_fake_quantizer.count == 1
                    assert child.weight_fake_quantizer.count == 1
            else:
                # For LoRA, we only fake quantize the input activations once per block:
                # For self_attn, we only fake quantize the q_proj's input activations
                # For mlp, we only fake quantize the gate_proj's input activations
                if name == "self_attn":
                    base_layer = child.q_proj.base_layer
                    assert hasattr(base_layer, "activation_fake_quantizer")
                    assert base_layer.activation_fake_quantizer.count == 1
                elif name == "mlp":
                    base_layer = child.gate_proj.base_layer
                    assert hasattr(base_layer, "activation_fake_quantizer")
                    assert base_layer.activation_fake_quantizer.count == 1
                elif isinstance(child, FakeQuantizedLinear):
                    # Weight fake quantizers should always be called
                    assert child.weight_fake_quantizer.count == 1

    for k, v in example_inputs.items():
        example_inputs[k] = v.cuda()
    model.apply(_swap_fake_quantizers)
    model(**example_inputs)
    model.apply(_assert_fake_quantizers_are_called)


def _test_model_fake_quantize(qat_scheme: bool, full_finetuning: bool):
    """
    Test that all linear layers in the model are fake quantized according to the `qat_scheme`.
    """
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
    _test_fake_quantizers_are_called(model, inputs, full_finetuning)


# TODO: there are bad interactions across tests right now, need to figure out
# how to disable model caching before re-enabling this test
@pytest.mark.parametrize("qat_scheme", ["fp8-int4", "fp8-fp8"])
def _test_full_model_fake_quantize(qat_scheme: bool):
    _test_model_fake_quantize(qat_scheme, full_finetuning = True)


@pytest.mark.parametrize("qat_scheme", ["fp8-int4", "fp8-fp8"])
def test_lora_model_fake_quantize(qat_scheme: bool):
    _test_model_fake_quantize(qat_scheme, full_finetuning = False)
