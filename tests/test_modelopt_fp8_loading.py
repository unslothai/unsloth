# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""ModelOpt FP8 checkpoints (sarvam-105b-fp8) load via the transformers fp8 quantizer."""

import inspect
import json
import os
import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from real_accelerator import has_real_cuda

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
    from transformers.quantizers.quantizers_utils import should_convert_module

    plan = modelopt_fp8_plan(SimpleNamespace(quantization_config = _sarvam_quant()))
    skip = plan.pop("modules_to_not_convert")
    assert plan == {"quant_method": "fp8", "weight_block_size": None, "activation_scheme": "static"}
    assert not should_convert_module("lm_head", skip)
    assert should_convert_module("model.layers.0.self_attn.q_proj", skip)


def test_plan_accepts_hf_quant_config_spelling_and_weight_only():
    legacy = {
        "quant_method": "modelopt",
        "quantization": {"quant_algo": "FP8", "exclude_modules": ["lm_head", "mlp.gate"]},
    }
    from transformers.quantizers.quantizers_utils import should_convert_module

    plan = modelopt_fp8_plan(SimpleNamespace(quantization_config = legacy))
    skip = plan["modules_to_not_convert"]
    assert not should_convert_module("lm_head", skip)
    assert not should_convert_module("model.layers.3.mlp.gate", skip)
    assert should_convert_module("model.layers.3.mlp.gate_proj", skip)
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
        **{
            f"dynamic {strategy} activations": _sarvam_quant(
                config_groups = {
                    "g": {
                        "weights": {"num_bits": 8, "type": "float"},
                        "input_activations": {
                            "num_bits": 8,
                            "type": "float",
                            "dynamic": True,
                            "strategy": strategy,
                        },
                    }
                }
            )
            for strategy in ("channel", "group", "block")
        },
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
    before = dict(kwargs)
    pop_modelopt_key_mapping(config, kwargs)
    assert kwargs == before


def _pin_vlm_names(monkeypatch, names):
    """Point both places transformers has kept its VLM name list at ``names``."""
    import importlib
    for module_name in ("transformers.conversion_mapping", "transformers.modeling_utils"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        monkeypatch.setattr(module, "VLMS", list(names), raising = False)


def test_key_mapping_keeps_the_vlm_checkpoint_renames(monkeypatch):
    """transformers 5.3 applies a VLM's class renames only without a caller key_mapping."""
    _pin_vlm_names(monkeypatch, ["llava"])

    class LlavaForConditionalGeneration:
        _checkpoint_conversion_mapping = {r"^language_model\.model": "model.language_model"}

    config = SimpleNamespace(quantization_config = _sarvam_quant())
    if arm_modelopt_fp8_loading(config, verbose = False) is None:
        pytest.skip("this transformers has no per-tensor fp8")
    kwargs = {}
    pop_modelopt_key_mapping(config, kwargs, LlavaForConditionalGeneration)
    assert kwargs["key_mapping"] == {
        r"^language_model\.model": "model.language_model",
        r"\.weight_scale$": ".weight_scale_inv",
        r"\.input_scale$": ".activation_scale",
    }


def test_key_mapping_adds_no_class_renames_transformers_would_not_apply(monkeypatch):
    _pin_vlm_names(monkeypatch, ["llava"])
    renames = {r"^language_model\.model": "model.language_model"}

    class MistralForCausalLM:
        _checkpoint_conversion_mapping = dict(renames)

    class LlavaForConditionalGeneration:
        _checkpoint_conversion_mapping = dict(renames)

    scale_only = {
        r"\.weight_scale$": ".weight_scale_inv",
        r"\.input_scale$": ".activation_scale",
    }
    for model_class, kwargs in (
        (MistralForCausalLM, {}),  # not a VLM
        (None, {}),  # class not resolved
        (LlavaForConditionalGeneration, {"key_mapping": {}}),  # the caller's mapping wins
    ):
        config = SimpleNamespace(quantization_config = _sarvam_quant())
        if arm_modelopt_fp8_loading(config, verbose = False) is None:
            pytest.skip("this transformers has no per-tensor fp8")
        pop_modelopt_key_mapping(config, kwargs, model_class)
        assert kwargs["key_mapping"] == scale_only, model_class


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
@pytest.mark.skipif(not has_real_cuda(), reason = "fp8 kernels need CUDA")
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


@pytest.mark.skipif(not has_real_cuda(), reason = "FastModel loads need an accelerator")
def test_a_declined_modelopt_format_still_refuses_to_load_in_process(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM
    from unsloth import FastModel

    config = LlamaConfig(
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 1,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        vocab_size = 128,
    )
    LlamaForCausalLM(config).save_pretrained(tmp_path)
    raw = json.loads((tmp_path / "config.json").read_text())
    raw["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(KeyError, match = "cannot load this `modelopt` checkpoint"):
        FastModel.from_pretrained(str(tmp_path), load_in_4bit = False, load_in_16bit = True)


@needs_per_tensor_fp8
@pytest.mark.skipif(not has_real_cuda(), reason = "FastLanguageModel loads need an accelerator")
def test_fast_llama_checks_fp8_hardware_on_the_rewritten_config(tmp_path):
    """Subprocess: FastLanguageModel patches the Llama classes process-wide."""
    import subprocess
    import sys

    _write_tiny_modelopt_llama(str(tmp_path))
    code = f"""
import unsloth
from unsloth import FastLanguageModel
from unsloth.models import llama
from unsloth.models._utils import get_quant_type
seen = []
llama.verify_fp8_support_if_applicable = lambda config: seen.append(get_quant_type(config))
try:
    FastLanguageModel.from_pretrained({str(tmp_path)!r}, load_in_4bit = False, max_seq_length = 64)
except Exception:
    pass  # the tiny checkpoint ships no tokenizer; the checks run before that
print("SEEN", seen)
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, timeout = 600)
    seen = [line for line in out.stdout.splitlines() if line.startswith("SEEN")]
    assert seen and "'fp8'" in seen[-1], (out.stdout[-2000:], out.stderr[-2000:])


def test_config_branch_moves_rope_extension_onto_the_config():
    import inspect
    from unsloth.models import llama

    source = inspect.getsource(llama.FastLlamaModel.from_pretrained)
    branch = source.split("if user_config is not None or _modelopt_rewritten:", 1)[1]
    branch = branch.split("AutoModelForCausalLM.from_pretrained(", 1)[0]
    assert 'kwargs.pop("rope_scaling", None)' in branch


def test_task_heads_stay_out_of_the_rewritten_plan_only_for_task_loads():
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        LlamaForCausalLM,
        LlamaForSequenceClassification,
    )
    from unsloth.models.modelopt_fp8 import keep_task_heads_unquantized

    def rewritten():
        return SimpleNamespace(
            quantization_config = {"quant_method": "fp8", "modules_to_not_convert": ["lm_head"]}
        )

    config = rewritten()
    assert keep_task_heads_unquantized(config, AutoModelForSequenceClassification)
    assert config.quantization_config["modules_to_not_convert"] == [
        "lm_head",
        "score",
        "classifier",
        "classification_head",
        "qa_outputs",
    ]
    assert keep_task_heads_unquantized(config, None, LlamaForSequenceClassification)
    assert len(config.quantization_config["modules_to_not_convert"]) == 5

    for causal in (AutoModelForCausalLM, LlamaForCausalLM):
        config = rewritten()
        assert not keep_task_heads_unquantized(config, causal)
        assert config.quantization_config["modules_to_not_convert"] == ["lm_head"]
    other = SimpleNamespace(quantization_config = {"quant_method": "gptq"})
    assert not keep_task_heads_unquantized(other, AutoModelForSequenceClassification)
    assert "modules_to_not_convert" not in other.quantization_config


@pytest.mark.parametrize(
    "task",
    [
        "AutoModelForMultipleChoice",
        "AutoModelForImageClassification",
        "AutoModelForAudioClassification",
    ],
)
def test_other_task_auto_classes_keep_their_head_out(task):
    import transformers
    from unsloth.models.modelopt_fp8 import keep_task_heads_unquantized

    if not hasattr(transformers, task):
        pytest.skip(f"no {task}")
    config = SimpleNamespace(
        quantization_config = {"quant_method": "fp8", "modules_to_not_convert": ["lm_head"]}
    )
    assert keep_task_heads_unquantized(config, getattr(transformers, task))
    assert {"score", "classifier"} <= set(config.quantization_config["modules_to_not_convert"])


def test_every_transformers_task_head_name_is_kept_out():
    """Bart-style sequence classification names its fresh head `classification_head`."""
    from transformers import BartConfig, BartForSequenceClassification
    from unsloth.models.modelopt_fp8 import _TASK_HEAD_MODULES

    model = BartForSequenceClassification(
        BartConfig(
            d_model = 16,
            encoder_layers = 1,
            decoder_layers = 1,
            encoder_attention_heads = 2,
            decoder_attention_heads = 2,
            encoder_ffn_dim = 32,
            decoder_ffn_dim = 32,
            vocab_size = 64,
        )
    )
    fresh = {name.split(".")[0] for name, _ in model.named_children()} - {"model"}
    assert fresh <= set(_TASK_HEAD_MODULES), fresh


def test_both_loaders_keep_task_heads_out_of_the_rewrite():
    import inspect
    from unsloth.models import llama, vision

    llama_source = inspect.getsource(llama.FastLlamaModel.from_pretrained)
    assert (
        "keep_task_heads_unquantized(model_config, AutoModelForSequenceClassification)"
        in llama_source
    )
    vision_source = inspect.getsource(vision.FastBaseModel.from_pretrained)
    assert "keep_task_heads_unquantized(auto_config, auto_model, model_class)" in vision_source


@needs_per_tensor_fp8
@pytest.mark.skipif(not has_real_cuda(), reason = "fp8 kernels need CUDA")
def test_tiny_modelopt_llama_loads_a_classification_head(tmp_path):
    from transformers import AutoConfig, AutoModelForSequenceClassification
    from unsloth.models.modelopt_fp8 import keep_task_heads_unquantized

    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("fp8 matmul needs sm_89+")
    _write_tiny_modelopt_llama(str(tmp_path))
    config = AutoConfig.from_pretrained(str(tmp_path), num_labels = 2, pad_token_id = 0)
    arm_modelopt_fp8_loading(config, verbose = False)
    assert keep_task_heads_unquantized(config, AutoModelForSequenceClassification)
    kwargs = {}
    pop_modelopt_key_mapping(config, kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(tmp_path), config = config, dtype = torch.bfloat16, device_map = "cuda", **kwargs
    )
    assert type(model.score) is nn.Linear
    assert model.score.weight.dtype == torch.bfloat16
    assert model.model.layers[0].self_attn.q_proj.weight.dtype == torch.float8_e4m3fn
    x = torch.randint(1, 512, (2, 16), device = "cuda")
    out = model(x, labels = torch.tensor([0, 1], device = "cuda"))
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert model.score.weight.grad is not None and model.score.weight.grad.abs().sum() > 0


def test_both_loaders_hand_the_planner_the_rewritten_plan():
    import inspect
    from unsloth.models import llama, vision

    llama_source = inspect.getsource(llama.FastLlamaModel.from_pretrained)
    assert (
        "rewritten_quantization_config = modelopt_planner_quantization_config(model_config)"
        in llama_source
    )
    vision_source = inspect.getsource(vision.FastBaseModel.from_pretrained)
    # A 16-bit load dequantizes the fp8 weights, so the planner must size them at bf16.
    assert "rewritten_quantization_config = modelopt_planner_quantization_config(" in vision_source
    assert "auto_config, dequantize = load_in_16bit" in vision_source


@needs_per_tensor_fp8
def test_the_planner_sizes_a_modelopt_checkpoint_from_the_rewritten_plan(tmp_path):
    from transformers import AutoConfig, LlamaConfig
    from unsloth.models.loader_utils import planner_quantization_kwargs
    from unsloth.models.modelopt_fp8 import modelopt_planner_quantization_config

    planner = pytest.importorskip("unsloth_zoo.device_map_planner")
    if "rewritten_quantization_config" not in inspect.getsource(planner.build_meta_model):
        pytest.skip("this unsloth_zoo planner cannot size a rewritten quantization method")
    config = LlamaConfig(
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        vocab_size = 256,
    )
    config.quantization_config = _sarvam_quant()
    config.save_pretrained(tmp_path)
    loaded = AutoConfig.from_pretrained(str(tmp_path))
    plan = arm_modelopt_fp8_loading(loaded, verbose = False)
    fp8 = modelopt_planner_quantization_config(loaded)
    assert fp8 == plan and fp8 is not loaded.quantization_config
    kwargs = planner_quantization_kwargs(rewritten_quantization_config = fp8)
    assert kwargs["rewritten_quantization_config"] is fp8 and "quantization_config" not in kwargs
    model, hf_quantizer, _ = planner.build_meta_model(str(tmp_path), **kwargs)
    assert type(hf_quantizer).__name__ == "FineGrainedFP8HfQuantizer"
    assert type(model.model.layers[0].self_attn.q_proj).__name__ != "Linear"
    assert type(model.lm_head).__name__ == "Linear"

    # A 16-bit load dequantizes: the planner then sees bf16 Linear layers, not fp8 ones.
    bf16 = modelopt_planner_quantization_config(loaded, dequantize = True)
    assert bf16["dequantize"] is True and "dequantize" not in loaded.quantization_config
    kwargs = planner_quantization_kwargs(rewritten_quantization_config = bf16)
    model, _, _ = planner.build_meta_model(str(tmp_path), **kwargs)
    assert type(model.model.layers[0].self_attn.q_proj).__name__ == "Linear"


def test_dynamic_token_and_tensor_activations_map_onto_per_token_fp8():
    # transformers' dynamic fp8 scales activations per token.
    for strategy in ("token", "tensor", None):
        inputs = {"num_bits": 8, "type": "float", "dynamic": True}
        if strategy is not None:
            inputs["strategy"] = strategy
        quant = _sarvam_quant(
            config_groups = {
                "g": {"weights": {"num_bits": 8, "type": "float"}, "input_activations": inputs}
            }
        )
        plan = modelopt_fp8_plan(SimpleNamespace(quantization_config = quant))
        assert plan is not None and plan["activation_scheme"] == "dynamic", strategy


def test_mixed_static_and_dynamic_activation_groups_are_declined():
    # One scheme for the whole model would drop the static groups' calibrated input scales.
    static = {"num_bits": 8, "type": "float", "dynamic": False}
    dynamic = {"num_bits": 8, "type": "float", "dynamic": True, "strategy": "token"}
    weights = {"num_bits": 8, "type": "float"}
    quant = _sarvam_quant(
        config_groups = {
            "a": {"weights": dict(weights), "input_activations": static},
            "b": {"weights": dict(weights), "input_activations": dynamic},
        }
    )
    assert modelopt_fp8_plan(SimpleNamespace(quantization_config = quant)) is None
    for inputs, scheme in ((static, "static"), (dynamic, "dynamic")):
        quant = _sarvam_quant(
            config_groups = {
                "a": {"weights": dict(weights), "input_activations": dict(inputs)},
                "b": {"weights": dict(weights), "input_activations": dict(inputs)},
            }
        )
        plan = modelopt_fp8_plan(SimpleNamespace(quantization_config = quant))
        assert plan is not None and plan["activation_scheme"] == scheme


def test_a_task_checkpoint_keeps_its_quantized_head():
    # Its head is on disk as fp8 with scales; excluding it would load fp8 bytes into a Linear.
    from transformers import AutoModelForSequenceClassification

    from unsloth.models.modelopt_fp8 import keep_task_heads_unquantized

    config = SimpleNamespace(
        architectures = ["LlamaForSequenceClassification"],
        quantization_config = {"quant_method": "fp8", "modules_to_not_convert": ["lm_head"]},
    )
    assert not keep_task_heads_unquantized(config, AutoModelForSequenceClassification)
    assert config.quantization_config["modules_to_not_convert"] == ["lm_head"]
    config.architectures = ["LlamaForCausalLM"]
    assert keep_task_heads_unquantized(config, AutoModelForSequenceClassification)
    assert "score" in config.quantization_config["modules_to_not_convert"]


def test_a_reused_config_keeps_the_scale_renaming():
    from transformers import LlamaConfig

    from unsloth.models.modelopt_fp8 import modelopt_rewritten

    config = LlamaConfig(hidden_size = 8, num_hidden_layers = 1, num_attention_heads = 2, vocab_size = 16)
    config.quantization_config = _sarvam_quant()
    if arm_modelopt_fp8_loading(config, verbose = False) is None:
        pytest.skip("this transformers has no per-tensor fp8")
    first, second = {}, {}
    pop_modelopt_key_mapping(config, first)
    assert not hasattr(config, UNSLOTH_MODELOPT_KEY_MAPPING_ATTR)
    assert arm_modelopt_fp8_loading(config, verbose = False) is None  # already fp8
    assert modelopt_rewritten(config)
    pop_modelopt_key_mapping(config, second)
    assert second["key_mapping"] == first["key_mapping"]
    assert UNSLOTH_MODELOPT_KEY_MAPPING_ATTR not in config.to_dict()

    fresh = LlamaConfig(hidden_size = 8, num_hidden_layers = 1, num_attention_heads = 2, vocab_size = 16)
    assert not modelopt_rewritten(fresh)
    kwargs = {}
    pop_modelopt_key_mapping(fresh, kwargs)
    assert kwargs == {}


@needs_per_tensor_fp8
def test_modelopt_ignore_globs_keep_fnmatch_meaning():
    from transformers.quantizers.quantizers_utils import should_convert_module

    quant = _sarvam_quant()
    quant["ignore"] = ["lm_head", "backbone.layers.16*", "*embed_tokens*", "visual*"]
    patterns = modelopt_fp8_plan(SimpleNamespace(quantization_config = quant))[
        "modules_to_not_convert"
    ]
    assert not should_convert_module("lm_head", patterns)
    assert should_convert_module("backbone.layers.1.mixer.in_proj", patterns)
    assert should_convert_module("backbone.layers.10.mixer.in_proj", patterns)
    assert not should_convert_module("backbone.layers.16.mixer.in_proj", patterns)
    assert not should_convert_module("model.embed_tokens", patterns)
    assert not should_convert_module("visual.blocks.0.attn.qkv", patterns)
    assert not should_convert_module("lm_head", patterns)
    assert should_convert_module("model.layers.0.self_attn.q_proj", patterns)


@needs_per_tensor_fp8
def test_merged_save_detects_a_rewritten_modelopt_checkpoint_as_fp8(tmp_path, monkeypatch):
    zoo_saving = pytest.importorskip("unsloth_zoo.saving_utils")
    original = getattr(
        zoo_saving._is_fp8_quant_config, "__wrapped__", zoo_saving._is_fp8_quant_config
    )
    if original(_sarvam_quant()):
        pytest.skip("this unsloth_zoo already dequantizes ModelOpt FP8 on a merged save")
    monkeypatch.setattr(zoo_saving, "_is_fp8_quant_config", original)
    dirs = {}
    for name, quant in (("fp8", _sarvam_quant()), ("nvfp4", _sarvam_quant(quant_algo = "NVFP4"))):
        dirs[name] = tmp_path / name
        dirs[name].mkdir()
        (dirs[name] / "config.json").write_text(
            json.dumps({"model_type": "llama", "quantization_config": quant})
        )
    status = zoo_saving.check_model_quantization_status
    assert status(str(dirs["fp8"])) == (False, None)
    arm_modelopt_fp8_loading(SimpleNamespace(quantization_config = _sarvam_quant()), verbose = False)
    assert status(str(dirs["fp8"])) == (True, "fp8")
    assert status(str(dirs["nvfp4"])) == (False, None)


@needs_per_tensor_fp8
@pytest.mark.skipif(not has_real_cuda(), reason = "FastLanguageModel loads need an accelerator")
def test_merged_16bit_save_of_a_modelopt_lora_reloads_without_unsloth(tmp_path):
    """Subprocess: FastLanguageModel patches the Llama classes process-wide."""
    import subprocess
    import sys

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("fp8 matmul needs sm_89+")
    ckpt, merged = tmp_path / "ckpt", tmp_path / "merged"
    ckpt.mkdir()
    _write_tiny_modelopt_llama(str(ckpt))
    vocab = {f"t{i}": i for i in range(512)}
    raw = Tokenizer(models.WordLevel(vocab, unk_token = "t0"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object = raw, unk_token = "t0", pad_token = "t1", eos_token = "t2"
    ).save_pretrained(str(ckpt))
    code = f"""
import torch
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained({str(ckpt)!r}, load_in_4bit = False, max_seq_length = 64, dtype = torch.bfloat16)
model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, target_modules = ["q_proj", "v_proj", "down_proj"])
torch.manual_seed(0)
with torch.no_grad():
    for name, p in model.named_parameters():
        if "lora_B" in name:
            p.normal_(0, 0.02)
x = torch.randint(0, 512, (2, 32), device = "cuda")
model.eval()
with torch.no_grad():
    lora = model.base_model.model.model.layers[0].self_attn.q_proj
    base = lora.base_layer
    expected = base.weight.float() * base.weight_scale_inv.float() + (
        lora.lora_B["default"].weight.float() @ lora.lora_A["default"].weight.float()
    ) * lora.scaling["default"]
    torch.save((x.cpu(), model(x).logits.float().cpu(), expected.cpu()), {str(tmp_path / "want.pt")!r})
model.save_pretrained_merged({str(merged)!r}, tok, save_method = "merged_16bit")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, timeout = 900)
    assert out.returncode == 0, (out.stdout[-2000:], out.stderr[-3000:])
    check = f"""
import sys, torch
from transformers import AutoModelForCausalLM
x, want, expected = torch.load({str(tmp_path / "want.pt")!r})
model = AutoModelForCausalLM.from_pretrained({str(merged)!r}, dtype = torch.bfloat16).cuda()
assert "unsloth" not in sys.modules
q = model.model.layers[0].self_attn.q_proj.weight
with torch.no_grad():
    got = model(x.cuda()).logits.float().cpu()
w_rel = float((q.float().cpu() - expected).norm() / expected.norm())
print("CHECK", q.dtype, w_rel, float((got - want).norm() / want.norm()))
"""
    out = subprocess.run([sys.executable, "-c", check], capture_output = True, text = True, timeout = 600)
    line = [l for l in out.stdout.splitlines() if l.startswith("CHECK")]
    assert line, (out.stdout[-2000:], out.stderr[-3000:])
    _, dtype, w_rel, rel = line[-1].split()
    assert dtype == "torch.bfloat16"
    # dequant(W) + B @ A * scaling, up to bf16 rounding; raw e4m3 bytes are off by ~4 orders.
    assert float(w_rel) < 1e-2, w_rel
    # Loose: the in-memory model also rounds activations to e4m3 (raw bytes give ~1.4).
    assert float(rel) < 0.2, rel


def test_fp8_linear_forward_patch_adds_the_bias():
    from unsloth.kernels.fp8 import module_forward_patch

    forward = module_forward_patch(lambda X, weight, scale: X @ weight.t(), "weight_scale_inv")
    biased, plain = nn.Linear(4, 3), nn.Linear(4, 3, bias = False)
    for module in (biased, plain):
        module.weight_scale_inv = torch.ones(())
    # fbgemm keeps its bias in fp32; the output must stay in the activation dtype.
    biased.bias.data = biased.bias.data.float()
    X = torch.randn(2, 4, dtype = torch.bfloat16)
    biased.weight.data, plain.weight.data = (m.weight.data.bfloat16() for m in (biased, plain))
    out = forward(biased, X)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, X @ biased.weight.t() + biased.bias.bfloat16())
    assert torch.equal(forward(plain, X), X @ plain.weight.t())


def test_save_keeps_transformers_fp8_scale_names():
    core = pytest.importorskip("transformers.core_model_loading")
    from unsloth.models.modelopt_fp8 import keep_fp8_scale_names_on_save

    ours = [
        core.WeightRenaming(source_patterns = k, target_patterns = v)
        for k, v in MODELOPT_FP8_KEY_MAPPING.items()
    ]
    other = core.WeightRenaming(source_patterns = r"^model\.old\.", target_patterns = "model.new.")
    model = nn.Module()
    model._weight_conversions = [other, *ours]
    keep_fp8_scale_names_on_save(model)
    assert model._weight_conversions == [other]


def test_exact_ignore_names_keep_their_module_boundary():
    from transformers.quantizers.quantizers_utils import should_convert_module

    skip = modelopt_fp8_plan(
        SimpleNamespace(quantization_config = _sarvam_quant(ignore = ["model.layers.1", "lm_head"]))
    )["modules_to_not_convert"]
    assert not should_convert_module("model.layers.1.self_attn.q_proj", skip)
    assert should_convert_module("model.layers.10.self_attn.q_proj", skip)
    assert should_convert_module("model.layers.11.mlp.down_proj", skip)
    assert not should_convert_module("lm_head", skip)


def test_vlm_ignore_globs_follow_the_instantiated_names():
    from transformers.quantizers.quantizers_utils import should_convert_module

    config = SimpleNamespace(
        quantization_config = _sarvam_quant(ignore = ["visual*", "lm_head"]),
        model_type = "qwen2_5_vl",
        architectures = ["Qwen2_5_VLForConditionalGeneration"],
    )
    skip = modelopt_fp8_plan(config)["modules_to_not_convert"]
    # Checkpoint `visual.*` is instantiated as `model.visual.*`; its bf16 weights have no fp8 scales.
    assert not should_convert_module("model.visual.blocks.0.attn.qkv", skip)
    assert not should_convert_module("model.visual.merger.mlp.0", skip)
    assert should_convert_module("model.language_model.layers.0.self_attn.q_proj", skip)


def _hf_quant_config_checkpoint(path, quant_algo = "FP8"):
    from transformers import LlamaConfig

    LlamaConfig(hidden_size = 64, num_hidden_layers = 1, num_attention_heads = 4).save_pretrained(path)
    hf_quant = {
        "producer": {"name": "modelopt", "version": "0.23.0"},
        "quantization": {
            "quant_algo": quant_algo,
            "kv_cache_quant_algo": None,
            "exclude_modules": ["lm_head"],
        },
    }
    (path / "hf_quant_config.json").write_text(json.dumps(hf_quant))


@needs_per_tensor_fp8
def test_standalone_hf_quant_config_is_rewritten(tmp_path):
    from transformers import AutoConfig

    fp8, nvfp4, plain = tmp_path / "fp8", tmp_path / "nvfp4", tmp_path / "plain"
    _hf_quant_config_checkpoint(fp8)
    _hf_quant_config_checkpoint(nvfp4, quant_algo = "NVFP4")
    from transformers import LlamaConfig

    LlamaConfig(hidden_size = 64, num_hidden_layers = 1, num_attention_heads = 4).save_pretrained(plain)

    config = AutoConfig.from_pretrained(str(fp8))
    assert getattr(config, "quantization_config", None) is None
    _, _, method = check_and_disable_bitsandbytes_loading(config, load_in_4bit = False, verbose = False)
    assert method == "fp8" and config.quantization_config["quant_method"] == "fp8"
    assert hasattr(config, UNSLOTH_MODELOPT_KEY_MAPPING_ATTR)

    for other in (nvfp4, plain):
        config = AutoConfig.from_pretrained(str(other))
        _, _, method = check_and_disable_bitsandbytes_loading(
            config, load_in_4bit = False, verbose = False
        )
        assert method is None and getattr(config, "quantization_config", None) is None

    # Under vLLM the block is attached but not rewritten (vLLM reads ModelOpt itself); the
    # default load_in_4bit must still drop so vLLM is not asked for bitsandbytes.
    config = AutoConfig.from_pretrained(str(fp8))
    load_in_4bit, _, method = check_and_disable_bitsandbytes_loading(
        config, load_in_4bit = True, verbose = False, rewrite_modelopt = False
    )
    assert method == "modelopt" and load_in_4bit is False
    assert config.quantization_config["quant_method"] == "modelopt"
    assert not hasattr(config, UNSLOTH_MODELOPT_KEY_MAPPING_ATTR)

    # A caller-built config carries no checkpoint path; the loader's model name finds the file.
    from transformers import LlamaConfig

    config = LlamaConfig(hidden_size = 64, num_hidden_layers = 1, num_attention_heads = 4)
    _, _, method = check_and_disable_bitsandbytes_loading(
        config, load_in_4bit = False, verbose = False, model_name = str(fp8)
    )
    assert method == "fp8"


@needs_per_tensor_fp8
def test_merged_save_detects_a_standalone_hf_quant_config(tmp_path, monkeypatch):
    zoo_saving = pytest.importorskip("unsloth_zoo.saving_utils")
    for name in ("_is_fp8_quant_config", "check_model_quantization_status"):
        fn = getattr(zoo_saving, name)
        monkeypatch.setattr(zoo_saving, name, getattr(fn, "__wrapped__", fn))
    fp8, nvfp4 = tmp_path / "fp8", tmp_path / "nvfp4"
    _hf_quant_config_checkpoint(fp8)
    _hf_quant_config_checkpoint(nvfp4, quant_algo = "NVFP4")
    status = lambda p: zoo_saving.check_model_quantization_status(str(p))
    if status(fp8) == (True, "fp8"):
        pytest.skip("this unsloth_zoo already reads hf_quant_config.json")
    arm_modelopt_fp8_loading(SimpleNamespace(quantization_config = _sarvam_quant()), verbose = False)
    assert status(fp8) == (True, "fp8")
    assert status(nvfp4) == (False, None)


def test_config_overrides_move_onto_the_rewritten_config():
    from transformers import LlamaConfig
    from unsloth.models.modelopt_fp8 import move_config_overrides_onto_config

    config = LlamaConfig(hidden_size = 64, num_hidden_layers = 1, num_attention_heads = 4)
    kwargs = {
        "use_cache": False,
        "pad_token_id": 7,
        "dtype": torch.bfloat16,
        "key_mapping": {},
        "subfolder": "x",
    }
    move_config_overrides_onto_config(config, kwargs)
    assert config.use_cache is False and config.pad_token_id == 7
    # from_pretrained's own arguments stay load arguments.
    assert set(kwargs) == {"dtype", "key_mapping", "subfolder"}


def test_save_keeps_fp8_scale_names_without_original_pattern_copies():
    from unsloth.models.modelopt_fp8 import keep_fp8_scale_names_on_save

    class WeightRenaming(SimpleNamespace):  # transformers 5.3 / 5.5: live patterns only
        pass

    renames = [
        WeightRenaming(source_patterns = [k], target_patterns = [v])
        for k, v in MODELOPT_FP8_KEY_MAPPING.items()
    ]
    other = WeightRenaming(source_patterns = ["^old"], target_patterns = ["new"])
    model = nn.Module()
    model._weight_conversions = [other, *renames]
    keep_fp8_scale_names_on_save(model)
    assert model._weight_conversions == [other]


@needs_per_tensor_fp8
def test_merged_save_is_armed_on_the_vllm_path(tmp_path, monkeypatch):
    zoo_saving = pytest.importorskip("unsloth_zoo.saving_utils")
    for name in ("_is_fp8_quant_config", "check_model_quantization_status"):
        fn = getattr(zoo_saving, name)
        monkeypatch.setattr(zoo_saving, name, getattr(fn, "__wrapped__", fn))
    if zoo_saving._is_fp8_quant_config(_sarvam_quant()):
        pytest.skip("this unsloth_zoo already dequantizes ModelOpt FP8 on a merged save")
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "llama", "quantization_config": _sarvam_quant()})
    )
    assert zoo_saving.check_model_quantization_status(str(tmp_path)) == (False, None)
    config = SimpleNamespace(quantization_config = _sarvam_quant())
    check_and_disable_bitsandbytes_loading(
        config, load_in_4bit = True, verbose = False, rewrite_modelopt = False
    )
    assert config.quantization_config["quant_method"] == "modelopt"
    assert zoo_saving.check_model_quantization_status(str(tmp_path)) == (True, "fp8")


@needs_per_tensor_fp8
def test_hf_quant_config_lookup_follows_the_load_location(tmp_path, monkeypatch):
    import huggingface_hub
    from transformers import LlamaConfig
    from unsloth.models.modelopt_fp8 import attach_hf_quant_config

    _hf_quant_config_checkpoint(tmp_path / "repo" / "sub")
    config = LlamaConfig(hidden_size = 64, num_hidden_layers = 1, num_attention_heads = 4)
    assert attach_hf_quant_config(
        config, model_name = str(tmp_path / "repo"), hub_kwargs = {"subfolder": "sub"}
    )

    seen = {}

    def fake_download(repo, filename, **kwargs):
        seen.update(kwargs)
        raise huggingface_hub.errors.LocalEntryNotFoundError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    config = LlamaConfig(hidden_size = 64, num_hidden_layers = 1, num_attention_heads = 4)
    hub = {"local_files_only": True, "cache_dir": str(tmp_path / "cache")}
    assert not attach_hf_quant_config(config, model_name = "org/not-cached", hub_kwargs = hub)
    assert seen["local_files_only"] is True and seen["cache_dir"] == hub["cache_dir"]


@needs_per_tensor_fp8
def test_merged_save_lookup_keeps_a_positional_token(tmp_path, monkeypatch):
    import unsloth.models.modelopt_fp8 as modelopt

    zoo_saving = pytest.importorskip("unsloth_zoo.saving_utils")
    for name in ("_is_fp8_quant_config", "check_model_quantization_status"):
        fn = getattr(zoo_saving, name)
        monkeypatch.setattr(zoo_saving, name, getattr(fn, "__wrapped__", fn))
    if zoo_saving._is_fp8_quant_config(_sarvam_quant()):
        pytest.skip("this unsloth_zoo already dequantizes ModelOpt FP8 on a merged save")
    _hf_quant_config_checkpoint(tmp_path)
    seen = []
    real = modelopt._hf_quant_config_path
    monkeypatch.setattr(
        modelopt,
        "_hf_quant_config_path",
        lambda name, revision = None, token = None, hub_kwargs = None: seen.append(token)
        or real(name, revision, token, hub_kwargs),
    )
    arm_modelopt_fp8_loading(SimpleNamespace(quantization_config = _sarvam_quant()), verbose = False)
    # As unsloth_zoo calls it: check_model_quantization_status(model_name, token, ...).
    assert zoo_saving.check_model_quantization_status(str(tmp_path), "hf_secret") == (True, "fp8")
    assert seen == ["hf_secret"]
