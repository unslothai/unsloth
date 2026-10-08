# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""fast_inference + bnb 8-bit must refuse before vLLM starts (it silently ran 16-bit)."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from unsloth.models.loader_utils import refuse_fast_inference_load_in_8bit

ROOT = Path(__file__).resolve().parents[1]
BNB_8BIT_CHECKPOINT = {"quant_method": "bitsandbytes", "load_in_8bit": True}
BNB_4BIT_CHECKPOINT = {
    "quant_method": "bitsandbytes",
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
}


def _config(quantization_config = None):
    if quantization_config is None:
        return SimpleNamespace(model_type = "qwen3")
    return SimpleNamespace(model_type = "qwen3", quantization_config = quantization_config)


def test_load_in_8bit_flag_is_refused():
    with pytest.raises(NotImplementedError, match = "load_in_8bit") as err:
        refuse_fast_inference_load_in_8bit(True, _config())
    message = str(err.value)
    assert "load_in_4bit = True" in message
    assert "fast_inference = False" in message


def test_caller_bitsandbytes_8bit_config_is_refused():
    from transformers import BitsAndBytesConfig
    with pytest.raises(NotImplementedError):
        refuse_fast_inference_load_in_8bit(False, _config(), BitsAndBytesConfig(load_in_8bit = True))
    with pytest.raises(NotImplementedError):
        refuse_fast_inference_load_in_8bit(False, _config(), {"load_in_8bit": True})


def test_prequantized_8bit_checkpoint_is_refused():
    # Flag off; the checkpoint's own config makes it 8-bit.
    with pytest.raises(NotImplementedError):
        refuse_fast_inference_load_in_8bit(False, _config(BNB_8BIT_CHECKPOINT))


@pytest.mark.parametrize(
    "checkpoint",
    [
        None,
        BNB_4BIT_CHECKPOINT,
        {"quant_method": "fp8", "weight_block_size": [128, 128]},
        {"quant_method": "fbgemm_fp8"},
        {"quant_method": "compressed-tensors"},
        {"quant_method": "gptq", "bits": 8},
    ],
)
def test_4bit_16bit_fp8_and_other_formats_pass(checkpoint):
    refuse_fast_inference_load_in_8bit(False, _config(checkpoint))
    refuse_fast_inference_load_in_8bit(False, _config(checkpoint), None)


def test_caller_4bit_config_passes():
    from transformers import BitsAndBytesConfig
    refuse_fast_inference_load_in_8bit(False, _config(), BitsAndBytesConfig(load_in_4bit = True))


def test_missing_config_does_not_crash():
    refuse_fast_inference_load_in_8bit(False, None, None)
    with pytest.raises(NotImplementedError):
        refuse_fast_inference_load_in_8bit(True, None, None)


def _calls(function, name):
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


def _from_pretrained(path, class_name):
    tree = ast.parse((ROOT / path).read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "from_pretrained":
                    return item
    raise AssertionError(f"{class_name}.from_pretrained not found in {path}")


@pytest.mark.parametrize(
    "path, class_name",
    [
        ("unsloth/models/llama.py", "FastLlamaModel"),
        ("unsloth/models/vision.py", "FastBaseModel"),
    ],
)
def test_every_vllm_load_is_gated_before_the_engine(path, class_name):
    function = _from_pretrained(path, class_name)
    loads = _calls(function, "load_vllm")
    guards = _calls(function, "refuse_fast_inference_load_in_8bit")
    assert len(loads) == 1, f"{path}: expected one load_vllm call"
    assert len(guards) == 1, f"{path}: missing the 8-bit fast_inference refusal"
    guard = guards[0]
    assert guard.lineno < loads[0].lineno
    args = [ast.unparse(arg) for arg in guard.args]
    assert args[0] == "load_in_8bit"
    assert args[1] == "model_config"
    assert "quantization_config" in args[2]
    # Same If branch as load_vllm.
    parents = {}
    for node in ast.walk(function):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def chain(node):
        out = []
        while node in parents:
            node = parents[node]
            out.append(node)
        return out

    assert next(n for n in chain(guard) if isinstance(n, ast.If)) is next(
        n for n in chain(loads[0]) if isinstance(n, ast.If)
    )


def test_only_load_vllm_callers_are_the_two_loaders():
    # A new load_vllm call site in the model loaders needs the same refusal.
    callers = []
    for path in (ROOT / "unsloth" / "models").rglob("*.py"):
        if "load_vllm(" in path.read_text(encoding = "utf-8"):
            callers.append(path.relative_to(ROOT).as_posix())
    assert sorted(callers) == ["unsloth/models/llama.py", "unsloth/models/vision.py"]
