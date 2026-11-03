# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""transformers 5.21+ passes the model to update_weight_conversions. Unsloth's 4-bit subclasses (installed on the first
compressed-tensors or fp8-to-NF4 load, then kept) and the Mistral FP8 wrapper took one argument and raised TypeError."""

import inspect

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.quantizers.base import HfQuantizer

if not hasattr(HfQuantizer, "update_weight_conversions"):
    pytest.skip("transformers has no update_weight_conversions hook", allow_module_level = True)

import unsloth  # noqa: F401,E402
from transformers import BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM  # noqa: E402
from transformers.quantizers import AutoHfQuantizer  # noqa: E402
from transformers.quantizers import auto as quantizers_auto  # noqa: E402

from unsloth.models import compressed_tensors_bnb, fp8_to_nf4, mistral_format  # noqa: E402

PASSES_MODEL = "model" in inspect.signature(HfQuantizer.update_weight_conversions).parameters


def _model():
    from accelerate import init_empty_weights
    with init_empty_weights():
        return LlamaForCausalLM(
            LlamaConfig(
                hidden_size = 8,
                num_hidden_layers = 1,
                num_attention_heads = 2,
                intermediate_size = 8,
                vocab_size = 16,
            )
        )


def _call(quantizer, model):
    # The way the installed transformers calls it (conversion_mapping.get_model_conversion_mapping).
    if PASSES_MODEL:
        return quantizer.update_weight_conversions([], model)
    return quantizer.update_weight_conversions([])


def _accepts_extra_positional(fn):
    return any(
        p.kind is inspect.Parameter.VAR_POSITIONAL
        for p in inspect.signature(fn).parameters.values()
    )


def _installed_bnb_overrides():
    fp8_to_nf4.install_fp8_to_nf4_quantizer()
    compressed_tensors_bnb.install_compressed_tensors_bnb_quantizer()
    cls = quantizers_auto.AUTO_QUANTIZER_MAPPING["bitsandbytes_4bit"]
    return [
        k
        for k in cls.__mro__
        if k.__module__.startswith("unsloth.") and "update_weight_conversions" in vars(k)
    ]


def test_bnb_4bit_overrides_forward_extra_arguments():
    overrides = _installed_bnb_overrides()
    if not overrides:
        pytest.skip("no Unsloth bitsandbytes 4-bit subclass is installed on this transformers")
    for cls in overrides:
        assert _accepts_extra_positional(vars(cls)["update_weight_conversions"]), cls.__qualname__


def test_bnb_4bit_quantizer_takes_the_installed_call():
    if not _installed_bnb_overrides():
        pytest.skip("no Unsloth bitsandbytes 4-bit subclass is installed on this transformers")
    quantizer = AutoHfQuantizer.from_config(
        BitsAndBytesConfig(load_in_4bit = True), pre_quantized = False
    )
    assert isinstance(_call(quantizer, _model()), list)


def test_mistral_fp8_dequantize_wrapper_takes_the_installed_call():
    fp8 = pytest.importorskip("transformers.quantizers.quantizer_finegrained_fp8")
    from transformers import FineGrainedFP8Config

    if "update_weight_conversions" not in vars(fp8.FineGrainedFP8HfQuantizer):
        pytest.skip("FineGrainedFP8HfQuantizer has no update_weight_conversions of its own")
    with mistral_format._mistral_format_conversions():
        wrapper = vars(fp8.FineGrainedFP8HfQuantizer)["update_weight_conversions"]
        assert _accepts_extra_positional(wrapper)
        quantizer = fp8.FineGrainedFP8HfQuantizer(FineGrainedFP8Config(dequantize = True))
        quantizer.pre_quantized = True
        assert isinstance(_call(quantizer, _model()), list)
