# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A plain transformers model loaded after FastLanguageModel in the same process runs the stock forwards."""

import inspect
import json
import subprocess
import sys
import types

import pytest
from real_accelerator import has_real_cuda


def _fake_family(monkeypatch):
    modeling = types.ModuleType("transformers.models.fakeplain.modeling_fakeplain")
    patcher_module = types.ModuleType("unsloth.models.fakeplain")

    def hf_forward(
        self,
        input_ids = None,
        labels = None,
    ):
        return "hf"

    base = type(
        "FakeBase", (), {"__module__": modeling.__name__, "generate": lambda self: "hf gen"}
    )
    model_cls = type("FakeModel", (base,), {"__module__": modeling.__name__, "forward": hf_forward})
    modeling.FakeModel = model_cls
    patcher = type("FastFakeModel", (), {"__module__": patcher_module.__name__})
    patcher_module.FakeModel, patcher_module.FastFakeModel = model_cls, patcher
    monkeypatch.setitem(sys.modules, modeling.__name__, modeling)
    monkeypatch.setitem(sys.modules, patcher_module.__name__, patcher_module)
    return model_cls, patcher


def test_patched_methods_only_run_on_marked_instances(monkeypatch):
    from unsloth.models import llama as unsloth_llama

    model_cls, patcher = _fake_family(monkeypatch)

    def fast_forward(
        self,
        input_ids = None,
        causal_mask = None,
        labels = None,
    ):
        return "fast"

    def pre_patch():
        model_cls.forward = fast_forward
        model_cls.generate = lambda self: "fast gen"
        model_cls.__init__ = lambda self: None

    for _ in range(2):  # a second load re-patches; it must not wrap twice or lose the originals
        snapshot = unsloth_llama._snapshot_transformers_modules(patcher)
        pre_patch()
        unsloth_llama._record_pre_patch_changes(snapshot)
        unsloth_llama._gate_pre_patch_to_unsloth_models(snapshot)

    plain, marked = model_cls(), model_cls()
    marked._unsloth_fast_path = True
    assert plain.forward() == "hf" and plain.generate() == "hf gen"
    assert marked.forward() == "fast" and marked.generate() == "fast gen"
    assert vars(model_cls)["forward"].__wrapped__ is fast_forward
    # Trainer reads the forward signature to keep dataset columns.
    assert list(inspect.signature(model_cls.forward).parameters) == [
        "self",
        "input_ids",
        "causal_mask",
        "labels",
    ]
    assert vars(model_cls)["__init__"] is not None and not hasattr(
        vars(model_cls)["__init__"], "_unsloth_gated"
    )

    unsloth_llama.restore_transformers_family(["fakeplain"])
    assert marked.forward() == "hf" and marked.generate() == "hf gen"
    assert "generate" not in vars(model_cls)


_TINY = """
import torch
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
vocab = {{"<unk>": 0, "<s>": 1, "</s>": 2, **{{f"w{{i}}": i + 3 for i in range(125)}}}}
tok = PreTrainedTokenizerFast(tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>")),
                              unk_token = "<unk>", bos_token = "<s>", eos_token = "</s>", pad_token = "</s>")
config = LlamaConfig(vocab_size = len(vocab), hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
                     num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16, max_position_embeddings = 128,
                     bos_token_id = 1, eos_token_id = 2, pad_token_id = 2)
path = {path!r}
torch.manual_seed(0)
LlamaForCausalLM(config).save_pretrained(path)
tok.save_pretrained(path)
"""

_PLAIN = """
import json, torch
from transformers import AutoModelForCausalLM
plain = AutoModelForCausalLM.from_pretrained({path!r}, dtype = torch.bfloat16).cuda().eval()
x = dict(input_ids = torch.tensor([[1, 5, 6, 7, 8, 9]], device = "cuda"),
         attention_mask = torch.ones(1, 6, dtype = torch.long, device = "cuda"))
with torch.no_grad():
    logits = plain(**x).logits.float()
gen = plain.generate(**x, max_new_tokens = 8, do_sample = False, min_new_tokens = 8)
print("PLAIN", json.dumps({{"logits": logits.cpu().tolist(), "gen": gen.cpu().tolist()}}))
"""


def _run(code):
    out = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, timeout = 900)
    lines = [line for line in out.stdout.splitlines() if line.startswith(("PLAIN ", "SECOND "))]
    assert out.returncode == 0 and lines, (out.stdout[-2000:], out.stderr[-3000:])
    return {line.split(" ", 1)[0]: json.loads(line.split(" ", 1)[1]) for line in lines}


@pytest.mark.skipif(not has_real_cuda(), reason = "FastLanguageModel loads need CUDA")
def test_plain_transformers_after_fast_language_model_matches_a_fresh_process(tmp_path):
    import torch

    path = str(tmp_path / "tiny")
    fresh = _run(_TINY.format(path = path) + _PLAIN.format(path = path))["PLAIN"]
    after = _run(
        f"""
import os
os.environ["UNSLOTH_COMPILE_LOCATION"] = {str(tmp_path / "compiled")!r}
from unsloth import FastLanguageModel
import json, torch
model, _ = FastLanguageModel.from_pretrained({path!r}, load_in_4bit = False, max_seq_length = 64)
"""
        + _PLAIN.format(path = path)
        + f"""
again, _ = FastLanguageModel.from_pretrained({path!r}, load_in_4bit = False, max_seq_length = 64)
with torch.no_grad():
    a = model(input_ids = x["input_ids"]).logits.float()
    b = again(input_ids = x["input_ids"]).logits.float()
print("SECOND", json.dumps({{"diff": (a - b).abs().max().item()}}))
"""
    )
    assert after["PLAIN"]["gen"] == fresh["gen"]
    diff = (torch.tensor(after["PLAIN"]["logits"]) - torch.tensor(fresh["logits"])).abs().max()
    assert diff.item() < 1e-2, diff  # bf16; the RoPE tables are built on a different device
    assert after["SECOND"]["diff"] == 0.0
