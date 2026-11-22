# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.


"""The fast-encoder merged export must merge a copy and leave the live LoRA model training (unsloth#14246)."""

import ast
import copy
from pathlib import Path

import pytest
import torch

peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

ST_PY = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "sentence_transformer.py"


def _helper():
    src = ST_PY.read_text(encoding = "utf-8")
    node = [
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name == "_copy_sentence_encoder_for_merge"
    ]
    assert node, "_copy_sentence_encoder_for_merge not found"
    ns = {"copy": copy, "torch": torch}
    exec(ast.get_source_segment(src, node[0]), ns)
    return ns["_copy_sentence_encoder_for_merge"]


class _FakeQuantBert(transformers.BertModel):
    """Takes the bitsandbytes rebuild branch on CPU: flagged 4bit, dequantize is a no-op."""

    def dequantize(self, dtype = None):
        return self


def _bert(cls = transformers.BertModel):
    torch.manual_seed(0)
    cfg = transformers.BertConfig(
        vocab_size = 64,
        hidden_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        intermediate_size = 64,
        max_position_embeddings = 32,
    )
    return cls(cfg).eval()


def _lora(base, init = True):
    cfg = peft.LoraConfig(
        r = 4,
        lora_alpha = 8,
        target_modules = ["query", "value"],
        init_lora_weights = init,
        task_type = "FEATURE_EXTRACTION",
    )
    model = peft.get_peft_model(base, cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.05)
    return model


def _out(model, ids):
    with torch.no_grad():
        return model(input_ids = ids).last_hidden_state


IDS = torch.tensor([[1, 5, 9, 13, 2, 7]])


def test_live_adapters_survive_and_merged_copy_matches():
    live = _lora(_bert())
    params = {n: p for n, p in live.named_parameters() if "lora_" in n}
    expected = _out(live, IDS)
    merged = _helper()(live, torch.float32)
    assert isinstance(live, peft.PeftModel)
    assert {n: p for n, p in live.named_parameters() if "lora_" in n} == params
    assert not any("lora_" in n for n, _ in merged.named_parameters())
    torch.testing.assert_close(_out(merged, IDS), expected)
    torch.testing.assert_close(_out(live, IDS), expected)


@pytest.mark.parametrize("init", [True, "olora"])
def test_quantized_rebuild_merges_the_trained_weights(init):
    base = _bert(_FakeQuantBert)
    base.is_loaded_in_4bit = True
    live = _lora(base, init)
    expected = _out(live, IDS)
    rng = torch.get_rng_state()
    merged = _helper()(live, torch.float32)
    assert torch.equal(torch.get_rng_state(), rng), "the export advanced the live run's RNG"
    assert isinstance(live, peft.PeftModel)
    assert not getattr(merged, "is_loaded_in_4bit", False)
    torch.testing.assert_close(_out(merged, IDS), expected, atol = 1e-5, rtol = 1e-4)
