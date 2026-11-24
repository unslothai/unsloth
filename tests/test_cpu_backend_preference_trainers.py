# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""CPU backend exports PatchDPOTrainer / PatchKTOTrainer, and a DPO run matches stock TRL + PEFT."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_cpu_backend_tiled_mlp import _tiny_checkpoint, _zoo_has_cpu_tiling

_ROOT = Path(__file__).resolve().parents[1]

_PROBE = r"""
import json, sys
import torch
arm, model_dir, init_path = sys.argv[1], sys.argv[2], sys.argv[3]
T = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
if arm == "unsloth":
    import unsloth
    from unsloth import FastLanguageModel, PatchDPOTrainer, PatchKTOTrainer
    PatchDPOTrainer()
    PatchKTOTrainer()
    model, tok = FastLanguageModel.from_pretrained(model_dir, max_seq_length = 64, dtype = torch.float32,
                                                   load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 4, lora_alpha = 8, lora_dropout = 0,
                                             target_modules = T, random_state = 0)
    device_type = unsloth.DEVICE_TYPE
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype = torch.float32)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = get_peft_model(model, LoraConfig(r = 4, lora_alpha = 8, lora_dropout = 0, target_modules = T))
    device_type = None
params = dict(model.named_parameters())
with torch.no_grad():
    if arm == "stock":
        g = torch.Generator().manual_seed(1)
        init = {n: torch.randn(p.shape, generator = g) * 0.05 for n, p in params.items() if "lora_" in n}
        torch.save(init, init_path)
    init = torch.load(init_path)
    assert set(init) == {n for n in params if "lora_" in n}
    for n, v in init.items():
        params[n].copy_(v)
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
rows = [{"prompt": f"t{3 + i} t{10 + i} ", "chosen": f"t{40 + i} t{41 + i}", "rejected": f"t{90 - i} t{70 + i}"}
        for i in range(8)]
cfg = DPOConfig(output_dir = sys.argv[4], per_device_train_batch_size = 2, max_steps = 3, learning_rate = 1e-2,
                logging_steps = 1, report_to = "none", save_strategy = "no", seed = 0, max_length = 32,
                bf16 = False, fp16 = False, lr_scheduler_type = "constant", beta = 0.1)
trainer = DPOTrainer(model = model, args = cfg, train_dataset = Dataset.from_list(rows), processing_class = tok)
trainer.train()
logs = [h for h in trainer.state.log_history if "rewards/margins" in h]
print("PROBE " + json.dumps({"device_type": device_type, "trainer": type(trainer).__module__,
                             "margins": [h["rewards/margins"] for h in logs],
                             "loss": [h["loss"] for h in logs]}))
"""


def _run(arm, model_dir, tmp):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("UNSLOTH_ALLOW_CPU", "UNSLOTH_FORCE_CPU", "UNSLOTH_DISABLE_CPU_FALLBACK")
    }
    env.update({"CUDA_VISIBLE_DEVICES": "", "UNSLOTH_FORCE_GPU_PATH": "1", "HF_HUB_OFFLINE": "1"})
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_ROOT), env.get("PYTHONPATH")]))
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, arm, model_dir, str(tmp / "init.pt"), str(tmp / arm)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 600,
    )
    for line in out.stdout.splitlines():
        if line.startswith("PROBE "):
            return json.loads(line[6:])
    pytest.fail(out.stdout[-2000:] + out.stderr[-4000:])


@pytest.mark.skipif(not _zoo_has_cpu_tiling(), reason = "unsloth_zoo without the CPU backend")
def test_cpu_backend_dpo_matches_stock_trl(tmp_path):
    pytest.importorskip("trl")
    model_dir = _tiny_checkpoint(str(tmp_path / "ckpt"))
    stock = _run("stock", model_dir, tmp_path)
    head = _run("unsloth", model_dir, tmp_path)
    if head["device_type"] != "cpu":
        pytest.skip(f"this host does not select the CPU backend ({head['device_type']})")
    assert head["trainer"].startswith("trl."), head["trainer"]
    assert len(head["margins"]) == len(stock["margins"]) == 3
    assert any(abs(m) > 1e-4 for m in stock["margins"]), stock  # the adapter moved the policy
    for a, b in zip(head["margins"], stock["margins"]):
        assert abs(a - b) <= 1e-6 + 1e-4 * abs(b), (head, stock)  # fp32 reduction-order noise
    for a, b in zip(head["loss"], stock["loss"]):
        assert abs(a - b) <= 1e-4, (head, stock)
