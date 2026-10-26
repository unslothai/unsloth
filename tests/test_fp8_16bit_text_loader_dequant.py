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
@pytest.mark.parametrize("as_object", [False, True])
def test_explicit_fp8_config_is_copied_for_full_finetuning(as_object):
    qc = dict(_BLOCK_FP8, modules_to_not_convert = ["lm_head"], dequantize = False)
    if as_object:
        qc = FineGrainedFP8Config.from_dict(qc)
    out = _fp8_dequantize_quantization_config(None, qc)
    assert out.dequantize is True
    assert out.modules_to_not_convert == ["lm_head"]
    assert (qc.dequantize if as_object else qc["dequantize"]) is False


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


@pytest.mark.skipif(not _HAS_DEQUANTIZE, reason = "transformers has no fp8 dequantize flag")
@pytest.mark.skipif(not has_real_cuda(), reason = "full finetuning loads on CUDA")
@pytest.mark.parametrize(
    "entrypoint, explicit_config", [("FastModel", False), ("FastLanguageModel", True)]
)
@pytest.mark.parametrize("float32_mixed_precision", [False, True])
def test_native_fp8_full_finetuning_preserves_scaled_weights(
    tmp_path, entrypoint, explicit_config, float32_mixed_precision
):
    from safetensors.torch import load_file, save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    src, dense, saved = tmp_path / "fp8", tmp_path / "dense", tmp_path / "saved"
    src.mkdir()
    dense.mkdir()
    _write_tiny_block_fp8_llama(str(src))
    tensors = load_file(str(src / "model.safetensors"))
    reference = {}
    for name, weight in tensors.items():
        if name.endswith("weight_scale_inv"):
            continue
        if weight.dtype == torch.float8_e4m3fn:
            scale = tensors[name.replace(".weight", ".weight_scale_inv")]
            scale = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
            weight = (weight.float() * scale).to(torch.bfloat16)
        reference[name] = weight
    save_file(reference, str(dense / "model.safetensors"))
    config = json.loads((src / "config.json").read_text())
    del config["quantization_config"]
    (dense / "config.json").write_text(json.dumps(config))
    raw = Tokenizer(models.WordLevel({f"t{i}": i for i in range(512)}, unk_token = "t0"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = raw, unk_token = "t0", pad_token = "t0", bos_token = "t1", eos_token = "t2"
    )
    tokenizer.save_pretrained(src)
    tokenizer.save_pretrained(dense)
    code = f"""
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import unsloth, torch
from transformers import AutoModelForCausalLM, FineGrainedFP8Config
from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

# Exercise the kept-FP8 branch even on pre-sm89 CUDA: disable only the environment
# fallback, retaining the real checkpoint loader, scale conversion and training code.
FineGrainedFP8HfQuantizer.validate_environment = lambda self, *a, **kw: None
loader = getattr(unsloth, {entrypoint!r})
options = dict(max_seq_length = 64, dtype = torch.bfloat16, full_finetuning = True,
               use_gradient_checkpointing = False, float32_mixed_precision = {float32_mixed_precision})
extra = {{}}
if {explicit_config}:
    extra = dict(load_in_4bit = False, quantization_config = FineGrainedFP8Config())
model, _ = loader.from_pretrained({str(src)!r}, **options, **extra)
assert not any(p.dtype == torch.float8_e4m3fn for p in model.parameters())
assert not any("weight_scale_inv" in name for name, _ in model.named_parameters())
assert getattr(model.config, "quantization_config", None) is None
reference, _ = loader.from_pretrained({str(dense)!r}, load_in_4bit = False, **options)
actual_weights, expected_weights = dict(model.named_parameters()), dict(reference.named_parameters())
assert actual_weights.keys() == expected_weights.keys()
for name in actual_weights:
    torch.testing.assert_close(actual_weights[name], expected_weights[name], rtol = 0, atol = 0, msg = name)
ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 2]], device = model.device)
model.eval()
reference.eval()
actual = model(input_ids = ids, labels = ids)
expected = reference(input_ids = ids, labels = ids)
torch.testing.assert_close(actual.logits, expected.logits, rtol = 0, atol = 0)
torch.testing.assert_close(actual.loss, expected.loss, rtol = 0, atol = 0)
actual.loss.backward()
expected.loss.backward()
for name in actual_weights:
    assert actual_weights[name].grad is not None, name
    assert torch.isfinite(actual_weights[name].grad).all(), name
    torch.testing.assert_close(actual_weights[name].grad, expected_weights[name].grad, rtol = 0, atol = 0, msg = name)
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
optimizer.step()
model.save_pretrained({str(saved)!r})
# Reload in fp32 to preserve the full-finetuning norm weights as well as the BF16 projections.
reloaded = AutoModelForCausalLM.from_pretrained({str(saved)!r}, dtype = torch.float32)
assert getattr(reloaded.config, "quantization_config", None) is None
for name, value in model.state_dict().items():
    torch.testing.assert_close(value.float().cpu(), reloaded.state_dict()[name], rtol = 0, atol = 0, msg = name)
print("FP8_FULL_FINETUNING_OK")
"""
    env = dict(
        os.environ, UNSLOTH_COMPILE_DISABLE = "1", UNSLOTH_RETURN_LOGITS = "1", HF_HUB_OFFLINE = "1"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output = True, text = True, timeout = 600, env = env
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "FP8_FULL_FINETUNING_OK" in result.stdout


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
