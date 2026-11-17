# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Default SFTTrainer full finetune path uses stochastic rounding; LoRA and the kill switch do not. Needs a GPU."""

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
os.environ["UNSLOTH_DISABLE_AUTO_UPDATES"] = "1"
from unsloth import FastLanguageModel
import torch, transformers
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

optim, lora = sys.argv[1], sys.argv[2] == "lora"
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
    dtype = torch.bfloat16, full_finetuning = not lora)
if lora:
    model = FastLanguageModel.get_peft_model(model, r = 4, lora_alpha = 8, random_state = 0,
        target_modules = ["q_proj", "v_proj"])
ds = Dataset.from_list([{"input_ids": list(range(5, 37)), "attention_mask": [1] * 32}] * 8)
cfg = SFTConfig(output_dir = os.environ["OUT_DIR"], per_device_train_batch_size = 2, max_steps = 3,
    logging_steps = 1, report_to = "none", save_strategy = "no", bf16 = True, max_length = 64,
    dataset_num_proc = 1, optim = optim, learning_rate = 1e-5)
trainer = SFTTrainer(model = model, processing_class = tok, train_dataset = ds, args = cfg)
trainer.train()
opt = trainer.optimizer
opt = getattr(opt, "optimizer", opt)
dtypes = sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})
print("SR_RESULT", getattr(opt, "_unsloth_bf16_stochastic_rounding", 0), type(opt).__name__, dtypes,
    trainer.state.log_history[-2].get("loss"))
"""


def _run(
    tmp_path,
    optim,
    kind,
    env_extra = None,
):
    env = dict(os.environ, OUT_DIR = str(tmp_path / f"{optim}_{kind}"))
    env.pop("UNSLOTH_BF16_STOCHASTIC_ROUNDING", None)
    env.update(env_extra or {})
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", _CODE, optim, kind],
        capture_output = True,
        text = True,
        timeout = 900,
        env = env,
        cwd = str(tmp_path),
    )
    lines = [l for l in out.stdout.splitlines() if l.startswith("SR_RESULT")]
    assert lines, out.stdout[-3000:] + out.stderr[-5000:]
    return lines[-1].split()


@pytest.mark.gpu
@pytest.mark.parametrize("optim", ["adamw_8bit", "adamw_torch"])
def test_full_finetuning_default_uses_stochastic_rounding(tmp_path, optim):
    result = _run(tmp_path, optim, "full")
    assert int(result[1]) > 0, result
    off = _run(tmp_path, optim, "full", {"UNSLOTH_BF16_STOCHASTIC_ROUNDING": "0"})
    assert int(off[1]) == 0, off


@pytest.mark.gpu
def test_lora_is_untouched(tmp_path):
    result = _run(tmp_path, "adamw_8bit", "lora")
    assert int(result[1]) == 0, result
