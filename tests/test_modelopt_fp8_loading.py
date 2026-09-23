# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""NVIDIA ModelOpt FP8 checkpoints load through the transformers fp8 quantizer.

transformers has no ``modelopt`` quantizer, so ``sarvamai/sarvam-105b-fp8`` (and every
``quant_method: modelopt`` / ``quant_algo: FP8`` checkpoint) stopped in the loader with
``KeyError: 'modelopt'``. The checkpoint is a static per-tensor fp8 one under other names:
``weight_scale`` / ``input_scale`` where transformers says ``weight_scale_inv`` /
``activation_scale``. These tests pin the classifier, the config rewrite, the key mapping
hand-off, and (on a GPU) the round trip of a tiny ModelOpt-format Llama in both the fp8 and
the dequantized 16-bit form against the same weights dequantized by hand.

Also here: transformers 5 ``replace_with_fp8_linear`` swaps any ``*.experts`` module for a
stacked ``FP8Experts``, which breaks remote-code MoE models whose experts are an
``nn.ModuleList`` of Linears (sarvam: ``FP8Experts has no attribute `0```).
"""

import json
import os
import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import unsloth  # noqa: F401
from unsloth.models.modelopt_fp8 import (
    MODELOPT_FP8_KEY_MAPPING,
    _transformers_accepts_fp8_plan,
    UNSLOTH_MODELOPT_KEY_MAPPING_ATTR,
    arm_modelopt_fp8_loading,
    modelopt_fp8_plan,
    pop_modelopt_key_mapping,
)
from unsloth.models.loader_utils import check_and_disable_bitsandbytes_loading


def _sarvam_quant(**overrides):
    quant = {
        "config_groups": {
            "group_0": {
                "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
                "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
                "targets": ["Linear"],
            }
        },
        "ignore": ["lm_head"],
        "quant_algo": "FP8",
        "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
        "producer": {"name": "modelopt", "version": "0.42.0"},
        "quant_method": "modelopt",
    }
    quant.update(overrides)
    return quant


# transformers 4.x only has dynamic block fp8, so the rewrite is skipped there.
needs_per_tensor_fp8 = pytest.mark.skipif(
    not _transformers_accepts_fp8_plan(
        modelopt_fp8_plan(SimpleNamespace(quantization_config = _sarvam_quant()))
    ),
    reason = "this transformers has no per-tensor fp8, so ModelOpt configs are left as is",
)


def test_plan_maps_sarvam_block_to_static_per_tensor_fp8():
    plan = modelopt_fp8_plan(SimpleNamespace(quantization_config = _sarvam_quant()))
    assert plan == {
        "quant_method": "fp8",
        "weight_block_size": None,
        "modules_to_not_convert": ["lm_head"],
        "activation_scheme": "static",
    }


def test_plan_accepts_hf_quant_config_spelling_and_weight_only():
    legacy = {
        "quant_method": "modelopt",
        "quantization": {"quant_algo": "FP8", "exclude_modules": ["lm_head", "mlp.gate"]},
    }
    plan = modelopt_fp8_plan(SimpleNamespace(quantization_config = legacy))
    assert plan["modules_to_not_convert"] == ["lm_head", "mlp.gate"]
    assert plan["activation_scheme"] == "static"

    weight_only = _sarvam_quant()
    weight_only["config_groups"]["group_0"]["input_activations"] = None
    plan = modelopt_fp8_plan(SimpleNamespace(quantization_config = weight_only))
    assert plan["activation_scheme"] == "dynamic"


def test_plan_declines_everything_else():
    cases = {
        "none": None,
        "native fp8": {"quant_method": "fp8", "weight_block_size": [128, 128]},
        "nvfp4": _sarvam_quant(quant_algo = "NVFP4"),
        "int4 awq": _sarvam_quant(quant_algo = "W4A8_AWQ"),
        "fp4 weights": _sarvam_quant(
            config_groups = {
                "g": {"weights": {"num_bits": 4, "type": "float"}, "input_activations": None}
            }
        ),
        "block strategy": _sarvam_quant(
            config_groups = {
                "g": {
                    "weights": {"num_bits": 8, "type": "float", "strategy": "block"},
                    "input_activations": None,
                }
            }
        ),
        "no groups": _sarvam_quant(config_groups = {}),
        "targets narrower than every Linear": _sarvam_quant(
            config_groups = {
                "g": {
                    "weights": {"num_bits": 8, "type": "float"},
                    "input_activations": {"num_bits": 8, "type": "float"},
                    "targets": ["re:.*mlp.*"],
                }
            }
        ),
        "static channel activations": _sarvam_quant(
            config_groups = {
                "g": {
                    "weights": {"num_bits": 8, "type": "float"},
                    "input_activations": {
                        "num_bits": 8,
                        "type": "float",
                        "dynamic": False,
                        "strategy": "channel",
                    },
                }
            }
        ),
        "int8 activations": _sarvam_quant(
            config_groups = {
                "g": {
                    "weights": {"num_bits": 8, "type": "float"},
                    "input_activations": {"num_bits": 8, "type": "int", "dynamic": False},
                }
            }
        ),
    }
    for name, quant in cases.items():
        assert modelopt_fp8_plan(SimpleNamespace(quantization_config = quant)) is None, name


@needs_per_tensor_fp8
def test_arm_rewrites_config_and_hands_mapping_to_kwargs():
    config = SimpleNamespace(quantization_config = _sarvam_quant())
    plan = arm_modelopt_fp8_loading(config, verbose = False)
    assert config.quantization_config == plan
    assert config.quantization_config["quant_method"] == "fp8"

    kwargs = {"key_mapping": {r"\.weight_scale$": "user_wins", r"^old\.": "new."}}
    pop_modelopt_key_mapping(config, kwargs)
    assert not hasattr(config, UNSLOTH_MODELOPT_KEY_MAPPING_ATTR)
    assert kwargs["key_mapping"] == {
        r"\.weight_scale$": "user_wins",
        r"^old\.": "new.",
        r"\.input_scale$": ".activation_scale",
    }
    # Idempotent: nothing parked, nothing touched.
    before = dict(kwargs)
    pop_modelopt_key_mapping(config, kwargs)
    assert kwargs == before


def test_transformers_without_per_tensor_fp8_keep_the_config():
    if _transformers_accepts_fp8_plan({"quant_method": "fp8", "weight_block_size": None}):
        pytest.skip("this transformers loads per-tensor fp8")
    config = SimpleNamespace(quantization_config = _sarvam_quant())
    assert arm_modelopt_fp8_loading(config, verbose = False) is None
    assert config.quantization_config == _sarvam_quant()
    assert not hasattr(config, UNSLOTH_MODELOPT_KEY_MAPPING_ATTR)


def test_key_mapping_is_anchored():
    renamed = {}
    for key in (
        "model.layers.0.mlp.experts.3.down_proj.weight_scale",
        "model.layers.0.mlp.experts.3.down_proj.input_scale",
        "model.layers.0.mlp.experts.3.down_proj.weight_scale_inv",
        "model.layers.0.mlp.experts.3.down_proj.weight_scale_2",
        "model.layers.0.mlp.experts.3.down_proj.weight",
    ):
        new = key
        for pattern, target in MODELOPT_FP8_KEY_MAPPING.items():
            new = re.sub(pattern, target, new)
        renamed[key] = new
    assert list(renamed.values()) == [
        "model.layers.0.mlp.experts.3.down_proj.weight_scale_inv",
        "model.layers.0.mlp.experts.3.down_proj.activation_scale",
        "model.layers.0.mlp.experts.3.down_proj.weight_scale_inv",
        "model.layers.0.mlp.experts.3.down_proj.weight_scale_2",
        "model.layers.0.mlp.experts.3.down_proj.weight",
    ]


@needs_per_tensor_fp8
def test_check_and_disable_rewrites_only_when_asked():
    config = SimpleNamespace(quantization_config = _sarvam_quant())
    load_in_4bit, load_in_8bit, method = check_and_disable_bitsandbytes_loading(
        config, load_in_4bit = True, verbose = False, rewrite_modelopt = False
    )
    assert method == "modelopt" and config.quantization_config["quant_method"] == "modelopt"
    assert not hasattr(config, UNSLOTH_MODELOPT_KEY_MAPPING_ATTR)

    load_in_4bit, load_in_8bit, method = check_and_disable_bitsandbytes_loading(
        config, load_in_4bit = True, verbose = False
    )
    assert method == "fp8" and (load_in_4bit, load_in_8bit) == (False, False)
    assert config.quantization_config["quant_method"] == "fp8"
    assert hasattr(config, UNSLOTH_MODELOPT_KEY_MAPPING_ATTR)


# ----------------------------------------------------------------------------- ModuleList experts


def _fp8_quantizer(**config_kwargs):
    try:
        from transformers import FineGrainedFP8Config
        from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
    except Exception as e:
        pytest.skip(f"transformers fp8 quantizer unavailable: {e}")
    config = FineGrainedFP8Config(
        activation_scheme = "static", weight_block_size = None, **config_kwargs
    )
    quantizer = FineGrainedFP8HfQuantizer(config)
    quantizer.pre_quantized = True
    return quantizer


def _tiny_remote_moe():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        vocab_size = 128,
    )
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    for layer in model.model.layers:
        mlp = nn.Module()
        mlp.gate = nn.Linear(64, 4, bias = False, device = "meta")
        mlp.experts = nn.ModuleList()
        for _ in range(4):
            expert = nn.Module()
            expert.gate_proj = nn.Linear(64, 32, bias = False, device = "meta")
            expert.up_proj = nn.Linear(64, 32, bias = False, device = "meta")
            expert.down_proj = nn.Linear(32, 64, bias = False, device = "meta")
            mlp.experts.append(expert)
        layer.mlp = mlp
    return model


@needs_per_tensor_fp8
def test_modulelist_experts_become_fp8_linears():
    quantizer = _fp8_quantizer()
    from transformers.integrations.finegrained_fp8 import FP8Linear

    model = _tiny_remote_moe()
    quantizer.quantization_config.modules_to_not_convert = [
        "lm_head",
        "model.layers.1.mlp.experts.2.down_proj",
    ]
    quantizer._process_model_before_weight_loading(model)

    names = [name for name, _ in model.named_modules()]
    assert not any("_unsloth_modulelist" in name for name in names)
    for i, layer in enumerate(model.model.layers):
        assert isinstance(layer.mlp.experts, nn.ModuleList)
        assert len(layer.mlp.experts) == 4
        for j, expert in enumerate(layer.mlp.experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                module = getattr(expert, proj)
                excluded = (i, j, proj) == (1, 2, "down_proj")
                assert isinstance(module, FP8Linear) != excluded, (i, j, proj)
    assert type(model.lm_head) is nn.Linear
    # The child order the checkpoint keys index into is unchanged.
    assert list(model.model.layers[0].mlp._modules) == ["gate", "experts"]


@needs_per_tensor_fp8
def test_wrapper_hides_only_modulelist_experts():
    from unsloth.import_fixes import _wrap_fp8_replace_for_modulelist_experts

    model = _tiny_remote_moe()

    class StackedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.empty(4, 64, 64, device = "meta"))

    model.model.layers[0].mlp.experts = StackedExperts()
    seen = {}

    def original(
        model,
        modules_to_not_convert = None,
        quantization_config = None,
        pre_quantized = False,
    ):
        seen["experts"] = [n for n, _ in model.named_modules() if n.endswith(".experts")]
        seen["patterns"] = modules_to_not_convert
        return model

    wrapped = _wrap_fp8_replace_for_modulelist_experts(original)
    assert _wrap_fp8_replace_for_modulelist_experts(wrapped) is wrapped
    wrapped(model, modules_to_not_convert = ["lm_head", "model.layers.1.mlp.experts.0.up_proj"])
    # The stacked module still reaches the by-name FP8Experts branch; the list does not.
    assert seen["experts"] == ["model.layers.0.mlp.experts"]
    assert seen["patterns"][0] == "lm_head"
    assert any("experts_unsloth_modulelist\\.0\\.up_proj$" in p for p in seen["patterns"])
    assert isinstance(model.model.layers[1].mlp.experts, nn.ModuleList)
    assert list(model.model.layers[1].mlp._modules) == ["gate", "experts"]

    # An exception inside the original still restores the names.
    def boom(
        model,
        modules_to_not_convert = None,
        quantization_config = None,
        pre_quantized = False,
    ):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _wrap_fp8_replace_for_modulelist_experts(boom)(model)
    assert list(model.model.layers[1].mlp._modules) == ["gate", "experts"]


# ----------------------------------------------------------------------------- GPU round trip


def _write_tiny_modelopt_llama(path):
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size = 256,
        intermediate_size = 512,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        vocab_size = 512,
    )
    reference = LlamaForCausalLM(config).to(torch.bfloat16)
    tensors, dequantized = {}, {}
    for key, value in reference.state_dict().items():
        if key.endswith("proj.weight"):
            scale = value.float().abs().amax() / 448.0
            packed = (value.float() / scale).to(torch.float8_e4m3fn)
            tensors[key] = packed
            tensors[key.replace(".weight", ".weight_scale")] = scale.reshape(()).float()
            tensors[key.replace(".weight", ".input_scale")] = torch.tensor(0.02)
            dequantized[key] = (packed.float() * scale).to(torch.bfloat16)
        else:
            tensors[key] = value.contiguous()
            dequantized[key] = value
    save_file(tensors, os.path.join(path, "model.safetensors"))
    reference.load_state_dict(dequantized)
    raw = config.to_dict()
    raw["architectures"] = ["LlamaForCausalLM"]
    raw["quantization_config"] = _sarvam_quant()
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(raw, f)
    return reference


@needs_per_tensor_fp8
@pytest.mark.skipif(not torch.cuda.is_available(), reason = "fp8 kernels need CUDA")
@pytest.mark.parametrize("dequantize", [False, True])
def test_tiny_modelopt_llama_round_trip(tmp_path, dequantize):
    from transformers import AutoConfig, AutoModelForCausalLM, FineGrainedFP8Config

    if torch.cuda.get_device_capability()[0] < 9 and not dequantize:
        pytest.skip("fp8 matmul needs sm_89+")
    reference = _write_tiny_modelopt_llama(str(tmp_path)).cuda()
    config = AutoConfig.from_pretrained(str(tmp_path))
    plan = arm_modelopt_fp8_loading(config, verbose = False)
    kwargs = {}
    pop_modelopt_key_mapping(config, kwargs)
    extra = {"dequantize": True} if dequantize else {}
    kwargs["quantization_config"] = FineGrainedFP8Config.from_dict(dict(plan), **extra)
    model = AutoModelForCausalLM.from_pretrained(
        str(tmp_path), config = config, dtype = torch.bfloat16, device_map = "cuda", **kwargs
    )
    q_proj = model.model.layers[0].self_attn.q_proj
    if dequantize:
        assert q_proj.weight.dtype == torch.bfloat16
        assert torch.equal(q_proj.weight, reference.model.layers[0].self_attn.q_proj.weight)
    else:
        assert q_proj.weight.dtype == torch.float8_e4m3fn
        assert float(q_proj.activation_scale) == pytest.approx(0.02)
    x = torch.randint(0, 512, (2, 32), device = "cuda")
    with torch.no_grad():
        got = model(x).logits.float()
        want = reference(x).logits.float()
    rel = ((got - want).norm() / want.norm()).item()
    # Dequantized weights are exact; the fp8 path also rounds activations to e4m3.
    assert rel < (0.01 if dequantize else 0.08), rel


def test_rewrite_follows_who_loads_the_weights():
    # vLLM reads ModelOpt natively, but only when it really owns the load: a num_labels
    # classification load and a missing vLLM both stay in process and need the rewrite.
    import inspect
    from unsloth.models import llama, vision

    llama_source = inspect.getsource(llama.FastLlamaModel.from_pretrained)
    assert (
        "rewrite_modelopt = not _vllm_will_load_weights(fast_inference, num_labels)" in llama_source
    )
    assert llama._vllm_will_load_weights(True, num_labels = 2) is False
    vision_source = inspect.getsource(vision.FastBaseModel.from_pretrained)
    assert "rewrite_modelopt = not (fast_inference and is_vLLM_available())" in vision_source
    # The un-rewritten ModelOpt config must not be looked up in transformers' quantizer map.
    assert "AUTO_QUANTIZATION_CONFIG_MAPPING.get(quant_method)" in vision_source
    assert "AUTO_QUANTIZATION_CONFIG_MAPPING[quant_method]" not in vision_source
