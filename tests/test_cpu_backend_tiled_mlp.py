# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""CPU backend honours unsloth_tiled_mlp / UNSLOTH_TILED_MLP and a tiled LoRA step matches the untiled one."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

_PROBE = r"""
import json, os, sys
import torch
how, model_dir = sys.argv[1], sys.argv[2]
import unsloth
from unsloth import FastLanguageModel
from unsloth_zoo import tiling_registry
kw = {"unsloth_tiled_mlp": "rows:8"} if how == "kwarg" else {}
if how == "kwarg_off":
    kw = {"unsloth_tiled_mlp": False}
model, tok = FastLanguageModel.from_pretrained(model_dir, max_seq_length = 64, dtype = torch.float32,
                                               load_in_4bit = False, use_gradient_checkpointing = "unsloth", **kw)
tiled = len(tiling_registry.tiled_modules(model))
model = FastLanguageModel.get_peft_model(model, r = 4, lora_alpha = 8, random_state = 0,
                                         target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                                           "gate_proj", "up_proj", "down_proj"])
torch.manual_seed(0)
with torch.no_grad():
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.normal_(0, 0.02)
ids = torch.randint(0, 100, (2, 16), generator = torch.Generator().manual_seed(1))
model.train()
out = model(input_ids = ids, labels = ids)
out.loss.backward()
grads = {n: p.grad.detach().double().reshape(-1).tolist() for n, p in model.named_parameters() if p.grad is not None}
print("PROBE " + json.dumps({"device_type": unsloth.DEVICE_TYPE, "tiled": tiled,
                             "tiled_after_peft": len(tiling_registry.tiled_modules(model)),
                             "loss": float(out.loss), "grads": grads,
                             "gpu_only": sorted(m for m in sys.modules if m.startswith(("unsloth.models", "unsloth.kernels",
                                                "unsloth_zoo.temporary_patches", "unsloth_zoo.compiler")))}))
"""


def _tiny_checkpoint(tmp):
    transformers = pytest.importorskip("transformers")
    from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast

    tokenizers = pytest.importorskip("tokenizers")
    import torch

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size = 128,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        max_position_embeddings = 64,
    )
    Qwen3ForCausalLM(cfg).save_pretrained(tmp)
    vocab = {f"t{i}": i for i in range(128)}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token = "t0"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object = tok, unk_token = "t0", pad_token = "t1", eos_token = "t2"
    ).save_pretrained(tmp)
    return tmp


def _run(
    how,
    model_dir,
    extra_env = None,
):
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "UNSLOTH_ALLOW_CPU",
            "UNSLOTH_FORCE_CPU",
            "UNSLOTH_DISABLE_CPU_FALLBACK",
            "UNSLOTH_TILED_MLP",
        )
    }
    env.update({"CUDA_VISIBLE_DEVICES": "", "UNSLOTH_FORCE_GPU_PATH": "1", "HF_HUB_OFFLINE": "1"})
    env.update(extra_env or {})
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_ROOT), env.get("PYTHONPATH")]))
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, how, model_dir],
        capture_output = True,
        text = True,
        env = env,
        timeout = 1200,
    )
    for line in out.stdout.splitlines():
        if line.startswith("PROBE "):
            return json.loads(line[6:])
    pytest.fail(out.stdout[-2000:] + out.stderr[-4000:])


def _zoo_has_cpu_tiling():
    import importlib.util

    spec = importlib.util.find_spec("unsloth_zoo")
    locs = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []
    return any(
        (Path(p) / "cpu" / "loader.py").is_file() and (Path(p) / "tiling_registry.py").is_file()
        for p in locs
    )


@pytest.mark.skipif(
    not _zoo_has_cpu_tiling(), reason = "unsloth_zoo without the CPU backend or sequence tiling"
)
def test_cpu_backend_loader_tiles_like_the_gpu_loaders(tmp_path):
    model_dir = _tiny_checkpoint(str(tmp_path / "ckpt"))
    off = _run("off", model_dir)
    if off["device_type"] != "cpu":
        pytest.skip(f"this host does not select the CPU backend ({off['device_type']})")
    kwarg = _run("kwarg", model_dir)
    env = _run("off", model_dir, {"UNSLOTH_TILED_MLP": "rows:8"})
    # An explicit kwarg beats UNSLOTH_TILED_MLP either way.
    env_off = _run("kwarg", model_dir, {"UNSLOTH_TILED_MLP": "0"})
    kwarg_off = _run("kwarg_off", model_dir)
    assert off["tiled"] == 0 and kwarg_off["tiled"] == 0
    assert kwarg["tiled"] == 2, kwarg["tiled"]  # one Qwen3MLP per decoder layer
    assert env["tiled"] == 2, env["tiled"]
    assert env_off["tiled"] == 2, env_off["tiled"]
    assert kwarg["tiled_after_peft"] == 2
    assert kwarg["gpu_only"] == [] and env["gpu_only"] == []
    torch = pytest.importorskip("torch")
    assert abs(kwarg["loss"] - off["loss"]) <= 1e-6 * abs(off["loss"])
    assert set(kwarg["grads"]) == set(off["grads"])
    for name, g in off["grads"].items():
        a, b = torch.tensor(kwarg["grads"][name]), torch.tensor(g)
        assert float((a - b).norm()) <= 1e-5 * max(float(b.norm()), 1e-12), name
