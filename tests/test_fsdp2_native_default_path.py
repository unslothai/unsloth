# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FSDP2 off (no UNSLOTH_FSDP2, one GPU): a real Unsloth load + LoRA + SFTTrainer step in a fresh process
never imports the native FSDP2 modules (fsdp2_native, fsdp2_perf, fsdp2_native_quant, fsdp2_vision), and
patches neither accelerate nor the Trainer. Needs a GPU (import unsloth)."""

import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("needs a GPU (import unsloth)", allow_module_level = True)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CODE = r"""
import os, sys
for k in ("UNSLOTH_FSDP2", "UNSLOTH_FSDP2_BACKEND", "WORLD_SIZE", "RANK", "LOCAL_RANK"):
    os.environ.pop(k, None)
os.environ["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

import transformers
# Kernel-friendly tiny Llama (head_dim 64: xformers / SDPA kernels exist on every arch), random weights.
src = "hf-internal-testing/tiny-random-LlamaForCausalLM"
tok = transformers.AutoTokenizer.from_pretrained(src)
config = transformers.LlamaConfig(vocab_size = len(tok), hidden_size = 256, intermediate_size = 512,
    num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 64,
    max_position_embeddings = 256)
torch.manual_seed(0)
path = os.path.join(os.environ["OUT_DIR"], "tiny_llama")
transformers.LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(path)
tok.save_pretrained(path)
model, tok = FastLanguageModel.from_pretrained(path, max_seq_length = 64, load_in_4bit = False,
    dtype = torch.bfloat16)
model = FastLanguageModel.get_peft_model(model, r = 4, lora_alpha = 8, random_state = 0,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
ds = Dataset.from_list([{"input_ids": list(range(5, 37)), "attention_mask": [1] * 32}] * 4)
cfg = SFTConfig(output_dir = os.environ["OUT_DIR"], per_device_train_batch_size = 2, max_steps = 1,
    logging_steps = 1, report_to = "none", save_strategy = "no", bf16 = True, max_length = 64,
    dataset_num_proc = 1)
trainer = SFTTrainer(model = model, processing_class = tok, train_dataset = ds, args = cfg)
trainer.train()
import accelerate, transformers
loaded = sorted(m for m in sys.modules if m.split(".")[-1] in
    ("fsdp2_native", "fsdp2_perf", "fsdp2_native_quant", "fsdp2_vision") and m.startswith("unsloth"))
assert not loaded, f"native FSDP2 modules imported on the default path: {loaded}"
assert not getattr(accelerate.Accelerator, "_unsloth_fsdp2_native_patched", False)
assert not getattr(transformers.Trainer, "_unsloth_fsdp2_native_patched", False)
assert not getattr(trainer.model, "_unsloth_fsdp2_native", False)
print("DEFAULT_PATH_OK", trainer.state.log_history[0].get("loss"))
"""


def test_default_path_never_imports_native_fsdp2(tmp_path):
    env = dict(os.environ, OUT_DIR = str(tmp_path / "out"))
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", _CODE],
        capture_output = True,
        text = True,
        timeout = 900,
        env = env,
        cwd = str(tmp_path),
    )
    assert "DEFAULT_PATH_OK" in out.stdout, out.stdout[-3000:] + out.stderr[-5000:]
