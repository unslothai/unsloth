# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A full-model merged_16bit save keeps the model's own 16-bit dtype, the one its saved config names.
Casting fp16 to bf16 on bf16 hardware rounded 65504 to 65536, which reloads as inf under the fp16
config; casting bf16 to fp16 on fp16-only hardware wrote inf outright."""

from __future__ import annotations

import json
import os

import pytest

save_mod = pytest.importorskip("unsloth.save", reason = "needs torch + unsloth_zoo")

import torch
import transformers
from safetensors import safe_open

# Largest finite value per dtype, and one the other 16-bit format cannot hold exactly.
_EDGE = {torch.float16: 65504.0, torch.bfloat16: 3.0e38, torch.float32: 1.0}


def _model(dtype):
    config = transformers.LlamaConfig(
        vocab_size = 32,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).to(dtype)
    with torch.no_grad():
        weight = model.model.layers[0].mlp.down_proj.weight
        weight[0, 0] = _EDGE[dtype]
        weight[0, 1] = 1.0009765625 if dtype == torch.float16 else 1.0
    return model


def _saved(directory):
    tensors = {}
    for name in os.listdir(directory):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(directory, name), "pt") as handle:
                tensors.update({k: handle.get_tensor(k) for k in handle.keys()})
    with open(os.path.join(directory, "config.json"), encoding = "utf-8") as handle:
        config = json.load(handle)
    return tensors, config.get("dtype", config.get("torch_dtype"))


@pytest.mark.parametrize("eager", ["0", "1"], ids = ["per-shard", "eager"])
@pytest.mark.parametrize("bf16_hardware", [True, False], ids = ["bf16-host", "fp16-host"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids = ["fp16", "bf16"])
def test_a_16bit_model_is_written_in_its_own_dtype(
    monkeypatch, tmp_path, dtype, bf16_hardware, eager
):
    monkeypatch.setattr(save_mod.torch.cuda, "is_bf16_supported", lambda *a, **k: bf16_hardware)
    monkeypatch.setenv("UNSLOTH_SAVE_EAGER_CAST", eager)
    model = _model(dtype)
    expected = model.state_dict()
    save_mod.unsloth_generic_save(
        model, None, save_directory = str(tmp_path), save_method = "merged_16bit"
    )
    tensors, config_dtype = _saved(tmp_path)
    assert config_dtype == str(dtype).removeprefix("torch.")
    for name, tensor in tensors.items():
        assert tensor.dtype == dtype, name
        assert torch.isfinite(tensor).all(), name
        assert torch.equal(tensor, expected[name]), name
    reloaded = transformers.LlamaForCausalLM.from_pretrained(str(tmp_path), torch_dtype = "auto")
    assert torch.isfinite(reloaded.model.layers[0].mlp.down_proj.weight).all()


@pytest.mark.parametrize("bf16_hardware", [True, False], ids = ["bf16-host", "fp16-host"])
def test_a_float32_model_still_follows_the_hardware(monkeypatch, tmp_path, bf16_hardware):
    monkeypatch.setattr(save_mod.torch.cuda, "is_bf16_supported", lambda *a, **k: bf16_hardware)
    save_mod.unsloth_generic_save(
        _model(torch.float32), None, save_directory = str(tmp_path), save_method = "merged_16bit"
    )
    tensors, _ = _saved(tmp_path)
    target = torch.bfloat16 if bf16_hardware else torch.float16
    assert {t.dtype for t in tensors.values() if t.is_floating_point()} == {target}


def test_an_explicit_float32_request_still_wins(monkeypatch, tmp_path):
    monkeypatch.setattr(save_mod.torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    save_mod.unsloth_generic_save(
        _model(torch.float16),
        None,
        save_directory = str(tmp_path),
        save_method = "merged_16bit",
        dtype = torch.float32,
    )
    tensors, config_dtype = _saved(tmp_path)
    assert {t.dtype for t in tensors.values() if t.is_floating_point()} == {torch.float32}
    assert config_dtype == "float32"
