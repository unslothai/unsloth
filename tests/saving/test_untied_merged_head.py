# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A tied model trained with ensure_weight_tying = False keeps its own lm_head in a merged 16-bit export."""

import inspect
import json
import os

import pytest

import unsloth  # noqa: F401
import torch
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from transformers import (
    AutoModelForCausalLM,
    Gemma2Config,
    Gemma2ForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
)

from unsloth.save import unsloth_generic_save

if "ensure_weight_tying" not in inspect.signature(LoraConfig).parameters:
    pytest.skip("this peft has no ensure_weight_tying", allow_module_level = True)


def _base(
    tmp_path,
    shard,
    gemma2 = False,
):
    torch.manual_seed(0)
    sizes = dict(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 2,
    )
    if gemma2:
        # Tied by its class default, so config.json carries no tie_word_embeddings key.
        model = Gemma2ForCausalLM(Gemma2Config(head_dim = 8, **sizes))
    else:
        model = LlamaForCausalLM(LlamaConfig(tie_word_embeddings = True, **sizes))
    path = str(tmp_path / "base")
    model.save_pretrained(path, max_shard_size = "8KB" if shard else "5GB")
    return path


@pytest.fixture(autouse = True)
def _cpu_merge(monkeypatch):
    # The suite's conftest reports a CUDA device; this merge runs on the CPU.
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "cpu")


def _train_and_merge(
    tmp_path,
    modules_to_save,
    ensure_weight_tying,
    shard = False,
    gemma2 = False,
):
    base = AutoModelForCausalLM.from_pretrained(_base(tmp_path, shard, gemma2), dtype = torch.float32)
    config = LoraConfig(
        r = 4,
        lora_alpha = 8,
        target_modules = ["q_proj", "v_proj"],
        modules_to_save = list(modules_to_save),
        ensure_weight_tying = ensure_weight_tying,
    )
    model = get_peft_model(base, config)
    ids = torch.randint(0, 64, (1, 12), generator = torch.Generator().manual_seed(1))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-1)
    for _ in range(2):
        model(input_ids = ids, labels = ids).loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model.eval()
    with torch.no_grad():
        reference = model(input_ids = ids).logits.float()
    saved = str(tmp_path / "merged")
    unsloth_generic_save(model, None, save_directory = saved, save_method = "merged_16bit")
    reloaded = AutoModelForCausalLM.from_pretrained(saved, dtype = torch.float32).eval()
    with torch.no_grad():
        logits = reloaded(input_ids = ids).logits.float()
    with open(os.path.join(saved, "config.json")) as f:
        saved_config = json.load(f)
    keys = set()
    for name in os.listdir(saved):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(saved, name), "pt") as f:
                keys |= set(f.keys())
    if shard:
        with open(os.path.join(saved, "model.safetensors.index.json")) as f:
            assert set(json.load(f)["weight_map"]) == keys
    return reference, logits, saved_config, keys, reloaded


@pytest.mark.parametrize("shard", [False, True])
@pytest.mark.parametrize(
    "modules_to_save",
    [("embed_tokens", "lm_head"), ("lm_head",), ("embed_tokens",)],
)
def test_untied_training_keeps_its_lm_head(tmp_path, modules_to_save, shard):
    reference, logits, config, keys, reloaded = _train_and_merge(
        tmp_path, modules_to_save, False, shard
    )
    assert config.get("tie_word_embeddings", True) is False
    assert "lm_head.weight" in keys
    assert (
        reloaded.get_input_embeddings().weight.data_ptr()
        != reloaded.get_output_embeddings().weight.data_ptr()
    )
    torch.testing.assert_close(logits, reference, atol = 1e-4, rtol = 1e-4)


def test_untied_training_keeps_its_lm_head_on_a_default_tied_config(tmp_path):
    reference, logits, config, keys, _ = _train_and_merge(
        tmp_path, ("embed_tokens", "lm_head"), False, gemma2 = True
    )
    assert config.get("tie_word_embeddings", True) is False
    assert "lm_head.weight" in keys
    torch.testing.assert_close(logits, reference, atol = 1e-4, rtol = 1e-4)


def test_tied_training_still_writes_one_tensor(tmp_path):
    reference, logits, config, keys, _ = _train_and_merge(tmp_path, ("embed_tokens",), True)
    assert config["tie_word_embeddings"] is True
    assert "lm_head.weight" not in keys
    torch.testing.assert_close(logits, reference, atol = 1e-4, rtol = 1e-4)
