# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FastModel and FastLanguageModel load the same family one after the other in one process, in either order."""

import subprocess
import sys
import types

import pytest
from real_accelerator import has_real_cuda


def test_compiled_classes_are_swapped_back_for_the_fast_patcher(monkeypatch):
    from unsloth.models.llama import _restore_uncompiled_transformers_classes

    modeling = types.ModuleType("transformers.models.fakearch.modeling_fakearch")
    patcher_module = types.ModuleType("unsloth.models.fakearch")
    original = type("FakeAttention", (), {"__module__": modeling.__name__})
    compiled = type("FakeAttention", (), {"__module__": "unsloth_compiled_module_fakearch"})
    other = type("FakeMLP", (), {"__module__": "somewhere_else"})
    modeling.FakeAttention = compiled
    modeling.FakeMLP = other
    modeling.CLASSES = {"eager": compiled, "mlp": other}
    patcher = type("FastFakeModel", (), {"__module__": patcher_module.__name__})
    patcher_module.FakeAttention = original
    patcher_module.FastFakeModel = patcher
    monkeypatch.setitem(sys.modules, modeling.__name__, modeling)
    monkeypatch.setitem(sys.modules, patcher_module.__name__, patcher_module)

    _restore_uncompiled_transformers_classes(patcher)
    assert modeling.FakeAttention is original
    assert modeling.CLASSES == {"eager": original, "mlp": other}
    assert modeling.FakeMLP is other


def test_pre_patch_changes_are_recorded_once_and_undone_for_the_compiler(monkeypatch):
    from unsloth.models import llama as unsloth_llama

    modeling = types.ModuleType("transformers.models.fakefam.modeling_fakefam")
    patcher_module = types.ModuleType("unsloth.models.fakefam")

    def hf_forward(self):
        return "hf"

    attention = type("FakeAttention", (), {"__module__": modeling.__name__, "forward": hf_forward})
    rotary = type("FakeRotary", (), {"__module__": modeling.__name__})
    modeling.FakeAttention, modeling.FakeRotary = attention, rotary
    patcher = type("FastFakeModel", (), {"__module__": patcher_module.__name__})
    patcher_module.FakeAttention, patcher_module.FastFakeModel = attention, patcher
    monkeypatch.setitem(sys.modules, modeling.__name__, modeling)
    monkeypatch.setitem(sys.modules, patcher_module.__name__, patcher_module)
    unsloth_rotary = type("FakeRotary", (), {})

    def pre_patch():
        attention.forward = lambda self: "fast"
        attention.extra = lambda self: "added"
        modeling.FakeRotary = unsloth_rotary

    for _ in range(
        2
    ):  # a second FastLanguageModel load must not record its own patches as the originals
        snapshot = unsloth_llama._snapshot_transformers_modules(patcher)
        pre_patch()
        unsloth_llama._record_pre_patch_changes(snapshot)
    assert attention().forward() == "fast" and modeling.FakeRotary is unsloth_rotary

    unsloth_llama.restore_transformers_family(["fakefam", "notloaded"])
    assert attention().forward() == "hf"
    assert "extra" not in vars(attention)
    assert modeling.FakeRotary is rotary


@pytest.mark.skipif(not has_real_cuda(), reason = "FastModel / FastLanguageModel loads need CUDA")
@pytest.mark.parametrize("order", ["FastModel,FastLanguageModel", "FastLanguageModel,FastModel"])
def test_both_loaders_on_the_same_family_in_one_process(tmp_path, order):
    """Subprocess: both loaders patch the Llama classes process-wide."""
    code = f"""
import os
os.environ["UNSLOTH_COMPILE_LOCATION"] = {str(tmp_path / "compiled")!r}
from unsloth import FastModel, FastLanguageModel
import torch
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
vocab = {{"<unk>": 0, "<s>": 1, "</s>": 2, **{{f"w{{i}}": i + 3 for i in range(125)}}}}
tok = PreTrainedTokenizerFast(tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>")),
                              unk_token = "<unk>", bos_token = "<s>", eos_token = "</s>", pad_token = "</s>")
config = LlamaConfig(vocab_size = len(vocab), hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
                     num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16, max_position_embeddings = 128,
                     bos_token_id = 1, eos_token_id = 2, pad_token_id = 2)
path = {str(tmp_path / "tiny")!r}
LlamaForCausalLM(config).save_pretrained(path)
tok.save_pretrained(path)
x = {{"input_ids": torch.tensor([[1, 5, 6, 7]], device = "cuda")}}
kwargs = {{"FastModel": dict(load_in_16bit = True), "FastLanguageModel": dict()}}
loaders = {{"FastModel": FastModel, "FastLanguageModel": FastLanguageModel}}
for name in {order!r}.split(","):
    model, _ = loaders[name].from_pretrained(path, load_in_4bit = False, max_seq_length = 64, **kwargs[name])
    print(name, "LOGITS", tuple(model(**x).logits.shape))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, timeout = 900)
    assert out.stdout.count("LOGITS") == 2, (out.stdout[-2000:], out.stderr[-3000:])
