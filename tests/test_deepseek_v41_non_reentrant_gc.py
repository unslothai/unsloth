# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""DeepSeek-V4.1 (community transformers port) must train with non-reentrant gradient checkpointing: its
CSA2 source layers publish compressed KV that later layers consume, and reentrant checkpointing's no_grad
first pass drops those cross-layer gradients (the port raises on use_reentrant=True)."""

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "importing unsloth needs a GPU"
)

_CHILD = r"""
import json, sys
import unsloth  # noqa: F401
import torch
import torch.utils.checkpoint as torch_checkpoint
from transformers import LlamaConfig, LlamaForCausalLM
from unsloth.models.vision import FastBaseModel

model_type = sys.argv[1]
config = LlamaConfig(
    vocab_size = 128, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
    num_attention_heads = 4, num_key_value_heads = 2, max_position_embeddings = 64,
)
model = LlamaForCausalLM(config).cuda()
# Stand-in for DeepseekV41ForCausalLM: post_patch_model keys the GC mode on config.model_type.
model.config.model_type = model_type
model = FastBaseModel.post_patch_model(
    model, use_gradient_checkpointing = "unsloth", trust_remote_code = True,
)
# What UnslothSFTConfig / TRL pass at train time.
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
funcs = [
    m._gradient_checkpointing_func for m in model.modules()
    if getattr(m, "_gradient_checkpointing_func", None) is not None
]
out = {
    "n_funcs": len(funcs),
    "use_reentrant": sorted({str(getattr(f, "keywords", {}).get("use_reentrant")) for f in funcs}),
    "global_checkpoint_patched": hasattr(torch_checkpoint.checkpoint, "_unsloth_original"),
}
# A checkpointed backward must run and reach the first layer (post_patch_model froze every weight).
model.train()
q_proj = model.model.layers[0].self_attn.q_proj.weight
q_proj.requires_grad_(True)
ids = torch.randint(0, 128, (2, 16), device = "cuda")
model(input_ids = ids, labels = ids).loss.backward()
out["layer0_grad"] = q_proj.grad is not None and bool(q_proj.grad.abs().sum() > 0)
print("@@@" + json.dumps(out))
"""


def _run(model_type, tmp_path):
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.setdefault("UNSLOTH_COMPILE_LOCATION", str(tmp_path / "unsloth_compiled_cache"))
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, model_type],
        cwd = str(tmp_path),
        env = env,
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    line = next(l for l in result.stdout.splitlines() if l.startswith("@@@"))
    return json.loads(line[3:])


def test_deepseek_v41_forces_non_reentrant(tmp_path):
    out = _run("deepseek_v41", tmp_path)
    assert out["n_funcs"] > 0, out
    assert out["use_reentrant"] == ["False"], out
    assert out["global_checkpoint_patched"], out
    assert out["layer0_grad"], out


def test_other_models_keep_reentrant(tmp_path):
    # Single process: the default stays reentrant (the offloaded Unsloth checkpoint) for other models.
    out = _run("llama", tmp_path)
    assert out["use_reentrant"] == ["True"], out
    assert out["layer0_grad"], out
