# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FastLanguageModel loads a family FastModel already compiled in the same process."""

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
    # Classes that were not rebound by the compiler are left alone.
    assert modeling.FakeMLP is other


@pytest.mark.skipif(not has_real_cuda(), reason = "FastModel / FastLanguageModel loads need CUDA")
def test_fast_language_model_after_fast_model_on_the_same_family(tmp_path):
    """Subprocess: both loaders patch the Llama classes process-wide."""
    code = f"""
import os
os.environ["UNSLOTH_COMPILE_LOCATION"] = {str(tmp_path / "compiled")!r}
from unsloth import FastModel, FastLanguageModel
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
try:
    tok = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-LlamaForCausalLM-3.2")
except Exception:
    print("SKIP no tokenizer"); raise SystemExit(0)
config = LlamaConfig(vocab_size = len(tok), hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
                     num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16, max_position_embeddings = 128)
path = {str(tmp_path / "tiny")!r}
LlamaForCausalLM(config).save_pretrained(path)
tok.save_pretrained(path)
x = tok("hello world", return_tensors = "pt").to("cuda")
a, _ = FastModel.from_pretrained(path, load_in_4bit = False, load_in_16bit = True, max_seq_length = 64)
a(**x)
b, _ = FastLanguageModel.from_pretrained(path, load_in_4bit = False, max_seq_length = 64)
print("LOGITS", tuple(b(**x).logits.shape))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, timeout = 900)
    if "SKIP no tokenizer" in out.stdout:
        pytest.skip("tokenizer download unavailable")
    assert "LOGITS" in out.stdout, (out.stdout[-2000:], out.stderr[-3000:])
