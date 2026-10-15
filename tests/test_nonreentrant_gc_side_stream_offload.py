# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Non-reentrant "unsloth" checkpointing offloads through unsloth_zoo's side stream hooks.

The old hooks queued the D2H copy on the compute stream, so it could not overlap compute.
Gradients must stay identical to checkpointing without offload.
"""

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
import json
import unsloth  # noqa: F401
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from unsloth.models.vision import FastBaseModel
import unsloth_zoo.gradient_checkpointing as zoo_gc

def run(mode):
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size = 128, hidden_size = 256, intermediate_size = 512, num_hidden_layers = 4,
        num_attention_heads = 4, num_key_value_heads = 2, max_position_embeddings = 256,
    )
    model = LlamaForCausalLM(config).cuda()
    # deepseek_v41 is the model type post_patch_model always runs non-reentrant.
    model.config.model_type = "deepseek_v41"
    model = FastBaseModel.post_patch_model(model, use_gradient_checkpointing = mode, trust_remote_code = True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    model.train()
    params = [p for n, p in model.named_parameters() if "layers." in n and "proj" in n]
    for p in params: p.requires_grad_(True)
    ids = torch.randint(0, 128, (2, 128), device = "cuda", generator = torch.Generator("cuda").manual_seed(1))
    loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    torch.cuda.synchronize()
    return loss.item(), [p.grad.detach().float().cpu() for p in params]

ref_loss, ref_grads = run(True)
loss, grads = run("unsloth")
offloader = getattr(zoo_gc, "NONREENTRANT_OFFLOADER", None)
print("@@@" + json.dumps({
    "same_loss": loss == ref_loss,
    "same_grads": all(torch.equal(a, b) for a, b in zip(grads, ref_grads)),
    "pinned_side_stream_copies": None if offloader is None else offloader.stats["pinned"],
}))
"""


def test_unsloth_non_reentrant_offload_uses_side_stream_hooks(tmp_path):
    zoo_gc = pytest.importorskip("unsloth_zoo.gradient_checkpointing")
    if not hasattr(zoo_gc, "NonReentrantActivationOffloader"):
        pytest.skip("unsloth_zoo predates the side stream offload hooks")
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.setdefault("UNSLOTH_COMPILE_LOCATION", str(tmp_path / "unsloth_compiled_cache"))
    result = subprocess.run(
        [sys.executable, "-c", _CHILD], cwd = str(tmp_path), env = env,
        capture_output = True, text = True, timeout = 900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    line = next(l for l in result.stdout.splitlines() if l.startswith("@@@"))
    out = json.loads(line[3:])
    assert out["same_loss"] and out["same_grads"], out
    assert out["pinned_side_stream_copies"], out
