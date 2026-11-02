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


@pytest.fixture(autouse = True)
def _restore_fp8_integration(monkeypatch):
    # The real fp8 loads below patch Fp8Dequantize process-wide; later test files check the unpatched class.
    try:
        import transformers.integrations.finegrained_fp8 as fp8_integration
    except Exception:
        return
    cls = getattr(fp8_integration, "Fp8Dequantize", None)
    if cls is not None:
        for name, value in list(vars(cls).items()):
            if not name.startswith("__"):
                monkeypatch.setattr(cls, name, value)
    if hasattr(fp8_integration, "replace_with_fp8_linear"):
        monkeypatch.setattr(
            fp8_integration, "replace_with_fp8_linear", fp8_integration.replace_with_fp8_linear
        )


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


@pytest.mark.skipif(not _HAS_DEQUANTIZE, reason = "transformers has no fp8 dequantize flag")
def test_transformers_own_dequantize_is_marked_and_saves_16bit(tmp_path):
    """No GPU: transformers dequantizes by itself, the branch it takes below sm89 too."""
    src, dst = tmp_path / "src", tmp_path / "saved"
    src.mkdir()
    _write_tiny_block_fp8_llama(str(src))
    code = f"""
import os, json, torch
from transformers import AutoModelForCausalLM
from safetensors import safe_open
from unsloth.models.loader_utils import _drop_fp8_dequantize_conversions
model = AutoModelForCausalLM.from_pretrained({str(src)!r}, dtype = torch.bfloat16)
assert getattr(model.config, "quantization_config", None) is None
print("DROPPED", _drop_fp8_dequantize_conversions(model))
print("MARKED", int(getattr(model, "_unsloth_fp8_dequantized", False)))
model.save_pretrained({str(dst)!r})
dtypes = set()
for name in os.listdir({str(dst)!r}):
    if name.endswith(".safetensors"):
        with safe_open(os.path.join({str(dst)!r}, name), "pt") as f:
            dtypes |= {{f.get_slice(k).get_dtype() for k in f.keys()}}
print("DTYPES", ",".join(sorted(dtypes)))
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_ALLOW_CPU = "1")
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output = True, text = True, timeout = 600, env = env
    )
    lines = dict(
        l.split(" ", 1)
        for l in out.stdout.splitlines()
        if l.startswith(("DROPPED", "MARKED", "DTYPES"))
    )
    assert "DTYPES" in lines, (out.stdout[-2000:], out.stderr[-2000:])
    assert int(lines["DROPPED"]) > 0 and lines["MARKED"] == "1", lines
    assert lines["DTYPES"] == "BF16", lines


def test_a_dequantized_fp8_load_gets_no_bitsandbytes_stamp():
    # transformers deletes the fp8 config of a dequantizing load, so the quant type reads None like an
    # unquantized load; the default load_in_4bit stamp then put a bnb config over 16bit weights.
    import ast

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unsloth/models/loader.py"
    )
    tree = ast.parse(open(path, encoding = "utf-8").read())

    def assigned(name):
        return [
            ast.unparse(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == name for t in node.targets)
        ]

    stamps = [v for v in assigned("_stamp_bnb") if "_quant" in v]
    assert len(stamps) == 2 and all("_unsloth_fp8_dequantized" in v for v in stamps), stamps
    # A config object (FineGrainedFP8Config kept by an sm89+ fp8 load) is read too.
    quants = assigned("_quant")
    assert len(quants) == 2 and all("_loaded_quant_method" in v for v in quants), quants


def test_loaded_quant_method_reads_dicts_and_config_objects():
    from unsloth.models.loader_utils import _loaded_quant_method

    assert _loaded_quant_method(SimpleNamespace(quantization_config = None)) is None
    assert _loaded_quant_method(SimpleNamespace()) is None
    assert _loaded_quant_method(SimpleNamespace(quantization_config = {"load_in_4bit": True})) is None
    assert _loaded_quant_method(SimpleNamespace(quantization_config = dict(_BLOCK_FP8))) == "fp8"
    if _HAS_DEQUANTIZE:
        qc = FineGrainedFP8Config.from_dict(dict(_BLOCK_FP8))
        assert _loaded_quant_method(SimpleNamespace(quantization_config = qc)) == "fp8"


@pytest.mark.skipif(not _HAS_DEQUANTIZE, reason = "transformers has no fp8 dequantize flag")
def test_only_a_load_without_fp8_config_is_marked_dequantized():
    # sm89+ keeps FP8Linear and its fp8 quantization_config: the bitsandbytes stamp must still see fp8.
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize
    from unsloth.models.loader_utils import _drop_fp8_dequantize_conversions

    def model(config):
        conversion = SimpleNamespace(
            operations = [Fp8Dequantize(None)], _original_target_patterns = ["weight"]
        )
        return SimpleNamespace(config = config, _weight_conversions = [conversion])

    kept = model(SimpleNamespace(quantization_config = dict(_BLOCK_FP8)))
    _drop_fp8_dequantize_conversions(kept)
    assert not getattr(kept, "_unsloth_fp8_dequantized", False)
    dequantized = model(SimpleNamespace())
    assert _drop_fp8_dequantize_conversions(dequantized) == 1
    assert dequantized._weight_conversions == [] and dequantized._unsloth_fp8_dequantized


def _write_tiny_tokenizer(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "hello": 3, "world": 4}
    tok = Tokenizer(WordLevel(vocab, unk_token = "[UNK]"))
    tok.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object = tok,
        unk_token = "[UNK]",
        bos_token = "[BOS]",
        eos_token = "[EOS]",
        pad_token = "[EOS]",
    ).save_pretrained(path)


@pytest.mark.skipif(not _HAS_DEQUANTIZE, reason = "transformers has no fp8 dequantize flag")
@pytest.mark.skipif(not has_real_cuda(), reason = "fp8 checkpoints load on CUDA")
@pytest.mark.parametrize("cls_name", ["FastLanguageModel", "FastModel"])
def test_explicit_dequantize_config_saves_without_fp8_config(tmp_path, cls_name):
    src, dst = tmp_path / "src", tmp_path / "saved"
    src.mkdir()
    _write_tiny_block_fp8_llama(str(src))
    _write_tiny_tokenizer(str(src))
    code = f"""
import json, os, torch
import unsloth
from unsloth import {cls_name} as cls
from transformers import AutoModelForCausalLM, FineGrainedFP8Config
from safetensors import safe_open
config = FineGrainedFP8Config(dequantize = True, weight_block_size = [128, 128])
model, _ = cls.from_pretrained(
    {str(src)!r}, max_seq_length = 64, dtype = torch.bfloat16, quantization_config = config,
    local_files_only = True,
)
print("MARKED", int(getattr(model, "_unsloth_fp8_dequantized", False)))
before = {{k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
model.save_pretrained({str(dst)!r})
with open(os.path.join({str(dst)!r}, "config.json")) as f:
    print("QUANT", json.dumps(json.load(f).get("quantization_config")))
dtypes = set()
for name in os.listdir({str(dst)!r}):
    if name.endswith(".safetensors"):
        with safe_open(os.path.join({str(dst)!r}, name), "pt") as f:
            dtypes |= {{f.get_slice(k).get_dtype() for k in f.keys()}}
print("DTYPES", ",".join(sorted(dtypes)))
# A GPU without bf16 (T4) loads float16: reload in whatever dtype was saved.
after = AutoModelForCausalLM.from_pretrained({str(dst)!r}, dtype = next(iter(before.values())).dtype).state_dict()
same = all(torch.equal(before[k], after[k].cpu()) for k in before if k in after)
print("RELOAD", int(same and set(before) <= set(after)))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, timeout = 600)
    lines = dict(
        l.split(" ", 1)
        for l in out.stdout.splitlines()
        if l.startswith(("MARKED", "QUANT", "DTYPES", "RELOAD"))
    )
    assert "RELOAD" in lines, (out.stdout[-2000:], out.stderr[-2000:])
    assert lines["MARKED"] == "1", lines
    assert lines["QUANT"] == "null", lines
    assert lines["DTYPES"] in ("BF16", "F16"), lines
    assert lines["RELOAD"] == "1", lines
