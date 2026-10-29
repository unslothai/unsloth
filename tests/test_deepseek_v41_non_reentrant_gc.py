# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""DeepSeek-V4.1 must use non-reentrant gradient checkpointing (cross-layer KV grads)."""

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

results, models = [], []
for model_type in sys.argv[1:]:
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
        "wrapper_in_use": any(hasattr(getattr(f, "func", f), "_unsloth_original") for f in funcs),
    }
    # A checkpointed backward must run and reach the first layer (post_patch_model froze every weight).
    model.train()
    q_proj = model.model.layers[0].self_attn.q_proj.weight
    q_proj.requires_grad_(True)
    ids = torch.randint(0, 128, (2, 16), device = "cuda")
    model(input_ids = ids, labels = ids).loss.backward()
    out["layer0_grad"] = q_proj.grad is not None and bool(q_proj.grad.abs().sum() > 0)
    results.append(out)
    models.append(model)
# Re-enabling the first model (what the trainer does) after later loads.
models[0].gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
results[0]["reenable_wrapper_in_use"] = all(
    hasattr(getattr(m._gradient_checkpointing_func, "func", None), "_unsloth_original")
    for m in models[0].modules() if getattr(m, "_gradient_checkpointing_func", None) is not None
)
print("@@@" + json.dumps(results))
"""


# Two post_patch_model calls then a positional enable: the newest wrapper must bind.
_CHILD_REPATCH = r"""
import json
import unsloth  # noqa: F401
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from unsloth.models.vision import FastBaseModel

config = LlamaConfig(
    vocab_size = 128, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
    num_attention_heads = 4, num_key_value_heads = 2, max_position_embeddings = 64,
)
model = LlamaForCausalLM(config).cuda()
model.config.model_type = "deepseek_v41"
for mode in (False, "unsloth"):
    model = FastBaseModel.post_patch_model(model, use_gradient_checkpointing = mode, trust_remote_code = True)
model.gradient_checkpointing_enable({"use_reentrant": True})
funcs = [
    m._gradient_checkpointing_func for m in model.modules()
    if getattr(m, "_gradient_checkpointing_func", None) is not None
]
def offload(f):
    f = getattr(f, "func", f)
    cells = dict(zip(f.__code__.co_freevars, (c.cell_contents for c in f.__closure__ or ())))
    return cells.get("_offload")
out = {
    "n_funcs": len(funcs),
    "use_reentrant": sorted({str(getattr(f, "keywords", {}).get("use_reentrant")) for f in funcs}),
    "offload": sorted({str(offload(f)) for f in funcs}),
}
print("@@@" + json.dumps([out]))
"""


def _run(
    tmp_path,
    *model_types,
    child = _CHILD,
):
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.setdefault("UNSLOTH_COMPILE_LOCATION", str(tmp_path / "unsloth_compiled_cache"))
    result = subprocess.run(
        [sys.executable, "-c", child, *model_types],
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
    (out,) = _run(tmp_path, "deepseek_v41")
    assert out["n_funcs"] > 0, out
    assert out["use_reentrant"] == ["False"], out
    assert out["global_checkpoint_patched"], out
    assert out["layer0_grad"], out


def test_other_models_keep_reentrant(tmp_path):
    (out,) = _run(tmp_path, "llama")
    assert out["use_reentrant"] == ["True"], out
    assert out["layer0_grad"], out


def test_later_model_gets_reentrant_back(tmp_path):
    v41, llama = _run(tmp_path, "deepseek_v41", "llama")
    assert v41["wrapper_in_use"], v41
    assert v41["reenable_wrapper_in_use"], v41
    assert not llama["global_checkpoint_patched"], llama
    assert not llama["wrapper_in_use"], llama
    assert llama["layer0_grad"], llama


def test_repatch_keeps_newest_wrapper_and_positional_call(tmp_path):
    (out,) = _run(tmp_path, child = _CHILD_REPATCH)
    assert out["n_funcs"] > 0, out
    assert out["use_reentrant"] == ["False"], out
    assert out["offload"] == ["True"], out
