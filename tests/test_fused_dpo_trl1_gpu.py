# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GPU: on TRL >= 1.0, UNSLOTH_FUSED_PREFERENCE_LOSS=1 must reach the fused DPO _compute_loss in a
real Unsloth DPOTrainer run, and the default (unset) must never reach it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or getattr(torch.cuda, "_unsloth_consolidated_spoof", False):
    pytest.skip("needs a real CUDA GPU", allow_module_level = True)
trl = pytest.importorskip("trl")

MODEL = os.environ.get("UNSLOTH_TEST_DPO_MODEL", "trl-internal-testing/tiny-Qwen3ForCausalLM")

SCRIPT = textwrap.dedent(
    """
    import json, sys
    import unsloth
    from unsloth import FastLanguageModel, PatchDPOTrainer
    import unsloth.models.dpo as dpo
    counts = {"fused": 0, "declined": 0}
    original = dpo.unsloth_fused_dpo_compute_loss_v1
    def counted(*args, **kwargs):
        out = original(*args, **kwargs)
        counts["fused" if out is not None else "declined"] += 1
        return out
    dpo.unsloth_fused_dpo_compute_loss_v1 = counted
    import torch
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer
    PatchDPOTrainer()
    model, tok = FastLanguageModel.from_pretrained(sys.argv[1], max_seq_length = 128, load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 8, target_modules = ["q_proj", "v_proj"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [
        {"prompt": f"Question {i}: what is {i} plus {i}?", "chosen": f" It is {2 * i}.", "rejected": f" It is {2 * i + 1}."}
        for i in range(8)
    ]
    args = DPOConfig(
        output_dir = sys.argv[2], per_device_train_batch_size = 2, max_steps = 3, learning_rate = 1e-4,
        logging_steps = 1, report_to = "none", max_length = 64, beta = 0.1, save_strategy = "no",
        fp16 = not torch.cuda.is_bf16_supported(), bf16 = torch.cuda.is_bf16_supported(),
    )
    trainer = DPOTrainer(model = model, ref_model = None, args = args, train_dataset = Dataset.from_list(rows), processing_class = tok)
    trainer.train()
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    print("RESULT " + json.dumps({"counts": counts, "losses": losses}))
    """
)


def _run(tmp_path, flag):
    env = dict(os.environ)
    env.pop("UNSLOTH_FUSED_PREFERENCE_LOSS", None)
    if flag is not None:
        env["UNSLOTH_FUSED_PREFERENCE_LOSS"] = flag
    env.setdefault("UNSLOTH_COMPILE_LOCATION", str(tmp_path / f"cc_{flag}"))
    script = tmp_path / "run_dpo.py"
    script.write_text(SCRIPT, encoding = "utf-8")
    proc = subprocess.run(
        [sys.executable, str(script), MODEL, str(tmp_path / f"out_{flag}")],
        env = env,
        capture_output = True,
        text = True,
        timeout = 1200,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-4000:] + proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT ") :])


def test_fused_dpo_v1_engages_only_when_requested(tmp_path):
    import inspect

    from trl import DPOTrainer

    source = (
        inspect.getsource(DPOTrainer._compute_loss) if hasattr(DPOTrainer, "_compute_loss") else ""
    )
    hooked_release = (
        "completion_mask" in source
        and "self._truncate_inputs(" not in source
        and "fused_lm_head=True" not in source
    )
    on = _run(tmp_path, "1")
    off = _run(tmp_path, None)
    assert off["counts"] == {"fused": 0, "declined": 0}
    if hooked_release:
        assert on["counts"]["fused"] >= 3 and on["counts"]["declined"] == 0, on
    else:
        assert on["counts"] == {"fused": 0, "declined": 0}, on
    for result in (on, off):
        assert result["losses"] and all(x == x and abs(x) < 1e4 for x in result["losses"]), result
