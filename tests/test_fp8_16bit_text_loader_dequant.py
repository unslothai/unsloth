# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An explicit 16bit load of a block-fp8 checkpoint dequantizes it on the FastLanguageModel path too."""

import inspect
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from real_accelerator import has_real_cuda

import unsloth  # noqa: F401
from unsloth.models.loader_utils import _fp8_dequantize_quantization_config

try:
    from transformers import FineGrainedFP8Config
    _HAS_DEQUANTIZE = "dequantize" in inspect.signature(FineGrainedFP8Config).parameters
except Exception:
    _HAS_DEQUANTIZE = False

_BLOCK_FP8 = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
}


@pytest.mark.skipif(not _HAS_DEQUANTIZE, reason = "transformers has no fp8 dequantize flag")
@pytest.mark.parametrize("as_object", [False, True])
def test_block_fp8_config_gets_dequantize(as_object):
    qc = FineGrainedFP8Config.from_dict(dict(_BLOCK_FP8)) if as_object else dict(_BLOCK_FP8)
    out = _fp8_dequantize_quantization_config(SimpleNamespace(quantization_config = qc))
    assert out is not None and out.dequantize is True
    assert list(out.weight_block_size) == [128, 128]


@pytest.mark.skipif(not _HAS_DEQUANTIZE, reason = "transformers has no fp8 dequantize flag")
def test_enum_quant_method_is_read_by_value():
    from transformers.utils.quantization_config import QuantizationMethod

    qc = dict(_BLOCK_FP8, quant_method = QuantizationMethod.FP8)
    out = _fp8_dequantize_quantization_config(SimpleNamespace(quantization_config = qc))
    assert out is not None and out.dequantize is True


@pytest.mark.parametrize(
    "qc",
    [None, {"quant_method": "bitsandbytes", "load_in_4bit": True}, {"quant_method": "modelopt"}],
)
def test_other_configs_are_left_alone(qc):
    assert _fp8_dequantize_quantization_config(SimpleNamespace(quantization_config = qc)) is None
    assert _fp8_dequantize_quantization_config(None) is None


def _write_tiny_block_fp8_llama(path):
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
    tensors = {}
    for key, value in reference.state_dict().items():
        if key.endswith("proj.weight"):
            n, k = value.shape
            blocks = value.float().view(n // 128, 128, k // 128, 128)
            scale = blocks.abs().amax(dim = (1, 3)) / 448.0
            packed = (blocks / scale[:, None, :, None]).view(n, k).to(torch.float8_e4m3fn)
            tensors[key] = packed
            tensors[key.replace(".weight", ".weight_scale_inv")] = scale.float()
        else:
            tensors[key] = value.contiguous()
    save_file(tensors, os.path.join(path, "model.safetensors"))
    raw = config.to_dict()
    raw["architectures"] = ["LlamaForCausalLM"]
    raw["quantization_config"] = dict(_BLOCK_FP8)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(raw, f)


@pytest.mark.skipif(not _HAS_DEQUANTIZE, reason = "transformers has no fp8 dequantize flag")
@pytest.mark.skipif(not has_real_cuda(), reason = "fp8 checkpoints load on CUDA")
@pytest.mark.parametrize("load_in_16bit", [True, False])
def test_fast_language_model_16bit_load_dequantizes(tmp_path, load_in_16bit):
    """Subprocess: FastLanguageModel patches the Llama classes process-wide."""
    _write_tiny_block_fp8_llama(str(tmp_path))
    code = f"""
import unsloth, torch, transformers
from unsloth import FastLanguageModel
real = transformers.AutoModelForCausalLM.from_pretrained.__func__
def load(cls, *args, **kwargs):
    model = real(cls, *args, **kwargs)
    fp8 = sum(p.dtype == torch.float8_e4m3fn for p in model.parameters())
    print("FP8", int(fp8), getattr(kwargs.get("quantization_config"), "dequantize", None), flush = True)
    return model
transformers.AutoModelForCausalLM.from_pretrained = classmethod(load)
try:
    FastLanguageModel.from_pretrained(
        {str(tmp_path)!r}, max_seq_length = 64, dtype = torch.bfloat16,
        load_in_4bit = False, load_in_16bit = {load_in_16bit},
    )
except Exception:
    pass  # the tiny checkpoint ships no tokenizer; the weights are loaded before that
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, timeout = 600)
    seen = [line.split() for line in out.stdout.splitlines() if line.startswith("FP8")]
    assert seen, (out.stdout[-2000:], out.stderr[-2000:])
    fp8, dequantize = int(seen[-1][1]), seen[-1][2]
    if load_in_16bit:
        assert fp8 == 0 and dequantize == "True", seen[-1]
    else:
        # Without the explicit flag nothing changes: transformers decides, as before.
        assert dequantize == "None", seen[-1]


def _write_tiny_block_fp8_qwen3_moe(path):
    from safetensors.torch import load_file, save_file
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    bf16 = os.path.join(path, "bf16")
    config = Qwen3MoeConfig(
        hidden_size = 256,
        intermediate_size = 512,
        moe_intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 64,
        num_experts = 4,
        num_experts_per_tok = 2,
        vocab_size = 512,
        tie_word_embeddings = False,
    )
    Qwen3MoeForCausalLM(config).to(torch.bfloat16).save_pretrained(bf16)
    tensors = {}
    for f in os.listdir(bf16):
        if f.endswith(".safetensors"):
            for key, value in load_file(os.path.join(bf16, f)).items():
                if key.endswith("proj.weight") and value.dim() == 2:
                    n, k = value.shape
                    blocks = value.float().view(n // 128, 128, k // 128, 128)
                    scale = blocks.abs().amax(dim = (1, 3)) / 448.0
                    tensors[key] = (
                        (blocks / scale[:, None, :, None]).view(n, k).to(torch.float8_e4m3fn)
                    )
                    tensors[key.replace(".weight", ".weight_scale_inv")] = scale.float()
                else:
                    tensors[key] = value
    save_file(tensors, os.path.join(path, "model.safetensors"))
    with open(os.path.join(bf16, "config.json")) as f:
        raw = json.load(f)
    raw["quantization_config"] = dict(_BLOCK_FP8)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(raw, f)


@pytest.mark.skipif(not _HAS_DEQUANTIZE, reason = "transformers has no fp8 dequantize flag")
@pytest.mark.skipif(not has_real_cuda(), reason = "fp8 checkpoints load on CUDA")
@pytest.mark.parametrize("arch", ["llama", "qwen3_moe"])
def test_dequantized_load_saves_and_reloads_16bit_weights(tmp_path, arch):
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM
    from unsloth.models.loader_utils import _drop_fp8_dequantize_conversions

    src, dst = tmp_path / "src", tmp_path / "saved"
    src.mkdir()
    (_write_tiny_block_fp8_llama if arch == "llama" else _write_tiny_block_fp8_qwen3_moe)(str(src))
    config = FineGrainedFP8Config.from_dict(dict(_BLOCK_FP8), dequantize = True)
    model = AutoModelForCausalLM.from_pretrained(
        str(src), dtype = torch.bfloat16, device_map = "cuda", quantization_config = config
    )
    assert _drop_fp8_dequantize_conversions(model) > 0
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.save_pretrained(str(dst))
    for name in os.listdir(dst):
        if name.endswith(".safetensors"):
            with safe_open(str(dst / name), "pt") as f:
                assert all(f.get_slice(k).get_dtype() == "BF16" for k in f.keys())
    with open(dst / "config.json") as f:
        assert "quantization_config" not in json.load(f)
    after = AutoModelForCausalLM.from_pretrained(str(dst), dtype = torch.bfloat16).state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k].cpu()) for k in before)
