# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Forced float32 (a float16 Gemma-4 / Gemma-3 / ... load, or a GPU without bfloat16) turns mixed precision off,
yet the frozen weights and sub-layer activations stay float16 and the fused CE saves d(loss)/d(hidden) as float16.
Without loss scaling those gradients underflow once the loss falls and training diverges, so the trainer must carry
a GradScaler there, hand it to the fused CE, and unscale before clipping; never on a bfloat16 run. Each case runs in
a fresh interpreter: the forced float32 patches are process wide."""

import json
import math
import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

TINY = os.environ.get(
    "UNSLOTH_TEST_TINY_GEMMA4", "trl-internal-testing/tiny-Gemma4ForConditionalGeneration"
)

_SCRIPT = textwrap.dedent(
    """
    import json, sys
    import unsloth
    from unsloth import FastModel
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    dtype = getattr(torch, sys.argv[2])
    model, tok = FastModel.from_pretrained(sys.argv[1], max_seq_length = 128, load_in_4bit = False, dtype = dtype)
    model = FastModel.get_peft_model(model, r = 8, lora_alpha = 8, lora_dropout = 0, random_state = 3407,
                                     finetune_vision_layers = False, use_gradient_checkpointing = "unsloth")
    tok = getattr(tok, "tokenizer", tok)
    torch.manual_seed(0)
    rows = [torch.randint(10, 200, (128,)).tolist() for _ in range(16)]
    fp16 = dtype == torch.float16
    trainer = SFTTrainer(
        model = model, processing_class = tok, train_dataset = Dataset.from_dict({"input_ids": rows}),
        args = SFTConfig(per_device_train_batch_size = 2, gradient_accumulation_steps = 2, max_steps = 4,
                         learning_rate = 1e-4, logging_steps = 1, report_to = "none", save_strategy = "no",
                         output_dir = sys.argv[3], max_length = 128, fp16 = fp16, bf16 = not fp16, seed = 3407),
    )
    scaler = trainer.accelerator.scaler
    trainer.train()
    logs = [h for h in trainer.state.log_history if "loss" in h]
    print("RESULT " + json.dumps(dict(
        forced = bool(getattr(model, "_unsloth_forced_float32", False)),
        mixed_precision = str(trainer.accelerator.mixed_precision),
        scaler = type(scaler).__name__ if scaler is not None else None,
        scale = float(scaler.get_scale()) if scaler is not None else None,
        fused_ce_scaler = getattr(model, "accelerator_scaler", None) is scaler and scaler is not None,
        losses = [h["loss"] for h in logs], grad_norms = [h.get("grad_norm") for h in logs])))
    """
)


def _run(tmp_path, dtype, **env):
    script = tmp_path / "run.py"
    script.write_text(_SCRIPT)
    full_env = dict(os.environ, UNSLOTH_COMPILE_LOCATION = str(tmp_path / "cache"), **env)
    proc = subprocess.run(
        [sys.executable, str(script), TINY, dtype, str(tmp_path / "out")],
        env = full_env,
        capture_output = True,
        text = True,
        timeout = 1200,
    )
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and line, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(line[-1][len("RESULT ") :])


def _finite(x):
    return x is not None and math.isfinite(x)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_forced_float32_float16_training_has_a_grad_scaler(tmp_path):
    on = _run(tmp_path, "float16")
    assert on["forced"], "a float16 Gemma-4 load must take the forced float32 path"
    assert on["mixed_precision"] == "no", on  # still no autocast: the forced patches own the dtypes
    assert (
        on["scaler"] == "GradScaler"
    ), "forced float32 on float16 weights trains without loss scaling"
    assert on[
        "fused_ce_scaler"
    ], "the fused CE must receive the GradScaler (it saves d(hidden) in its forward)"
    off = _run(tmp_path, "float16", UNSLOTH_FORCED_FLOAT32_GRAD_SCALER = "0")
    assert off["scaler"] is None
    # Unscaled before clipping: a finite logged norm is the true one, not the loss scale times it. A step the
    # scaler skips on overflow logs inf / nan, so compare the steps both runs took.
    pairs = [
        (a, b) for a, b in zip(on["grad_norms"], off["grad_norms"]) if _finite(a) and _finite(b)
    ]
    assert pairs, (on, off)
    a, b = pairs[0]
    assert abs(a - b) <= 0.05 * abs(b) + 1e-6, (on, off)
    assert all(math.isfinite(x) for x in on["losses"]), on


@pytest.mark.gpu
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason = "needs a bfloat16 GPU",
)
def test_bfloat16_training_has_no_extra_grad_scaler(tmp_path):
    res = _run(tmp_path, "bfloat16")
    assert not res["forced"]
    assert res["scaler"] is None


# One SGD (lr = 1) step through a real Trainer on a tiny model whose frozen layers are float16 and whose trainable
# layer is float32, so the applied update is minus the clipped gradient. The scaler is attached where the patched
# trainers attach it (after Trainer.__init__, before Trainer.train prepares the optimizer).
_OPT_SCRIPT = textwrap.dedent(
    """
    import json, os, sys, tempfile
    import unsloth
    from unsloth.models._utils import _unsloth_forced_float32_grad_scaler
    import torch, torch.nn as nn
    from datasets import Dataset
    from transformers import Trainer, TrainerCallback, TrainingArguments

    class Tiny(nn.Module):
        def __init__(self, half, loss_mult):
            super().__init__()
            g = torch.Generator().manual_seed(0)
            self.emb, self.up = nn.Embedding(64, 32), nn.Linear(32, 32, bias = False)
            self.head, self.lora = nn.Linear(32, 64, bias = False), nn.Linear(32, 32, bias = False)
            with torch.no_grad():
                for m in (self.emb, self.up, self.head, self.lora):
                    m.weight.copy_(torch.randn(m.weight.shape, generator = g) * 0.3)
                for m in (self.emb, self.up, self.head):
                    m.weight.copy_(m.weight.half().float()).requires_grad_(False)
                    if half: m.half()
            self.loss_mult = loss_mult
        def forward(self, input_ids, labels = None, **kwargs):
            h = self.up(self.emb(input_ids))
            h = h + self.lora(h.float()).to(h.dtype)
            logits = self.head(h).float()
            loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 64), labels[:, 1:].reshape(-1))
            return {"loss": loss * self.loss_mult}

    def run(arm, regime):
        os.environ["UNSLOTH_FORCED_FLOAT32_GRAD_SCALER"] = "1"
        model = Tiny(half = arm == "on", loss_mult = 2.0**-14 if regime == "tiny" else 1.0).cuda()
        if arm == "on":
            model._unsloth_forced_float32 = True
        torch.manual_seed(1)
        rows = [torch.randint(0, 64, (16,)).tolist() for _ in range(4)]
        args = TrainingArguments(
            output_dir = tempfile.mkdtemp(), per_device_train_batch_size = 4, max_steps = 1, learning_rate = 1.0,
            optim = "sgd", lr_scheduler_type = "constant", weight_decay = 0.0, report_to = "none", logging_steps = 1,
            save_strategy = "no", remove_unused_columns = False, seed = 0, max_grad_norm = 1e9 if regime == "tiny" else 1e-3)
        trainer = Trainer(model = model, args = args, train_dataset = Dataset.from_dict({"input_ids": rows, "labels": rows}))
        scaler = _unsloth_forced_float32_grad_scaler(trainer, model)
        if scaler is not None and regime == "overflow":
            scaler._init_scale = 2.0**40  # float16 gradients overflow at this scale
        seen = {}
        class Probe(TrainerCallback):
            def on_step_end(self, args, state, control, optimizer = None, **kwargs):
                seen["same_scaler"] = scaler is not None and getattr(optimizer, "scaler", None) is scaler
                seen["skipped"] = bool(trainer.accelerator.optimizer_step_was_skipped)
        trainer.add_callback(Probe())
        before = model.lora.weight.detach().clone()
        trainer.train()
        log = [h for h in trainer.state.log_history if "loss" in h][-1]
        return dict(seen, scaler = scaler is not None, grad_norm = float(log["grad_norm"]),
                    scale = float(scaler.get_scale()) if scaler is not None else None,
                    delta = (model.lora.weight.detach() - before).double().flatten().tolist())

    print("RESULT " + json.dumps({r: {a: run(a, r) for a in ("on", "ref")} for r in ("clip", "tiny", "overflow")}))
    """
)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_forced_float32_grad_scaler_update_matches_float32_reference(tmp_path):
    script = tmp_path / "opt.py"
    script.write_text(_OPT_SCRIPT)
    proc = subprocess.run(
        [sys.executable, str(script)],
        env = dict(os.environ, UNSLOTH_COMPILE_LOCATION = str(tmp_path / "cache")),
        capture_output = True,
        text = True,
        timeout = 900,
    )
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert proc.returncode == 0 and line, proc.stdout[-3000:] + proc.stderr[-3000:]
    res = json.loads(line[-1][len("RESULT ") :])

    def rel(a, b):
        a, b = torch.tensor(a, dtype = torch.float64), torch.tensor(b, dtype = torch.float64)
        return float((a - b).norm() / b.norm())

    for regime in ("clip", "tiny"):
        on, ref = res[regime]["on"], res[regime]["ref"]
        assert on["scaler"] and not ref["scaler"], res
        assert on["same_scaler"], "the prepared optimizer must step through the attached GradScaler"
        assert not on["skipped"], res
        # Clipping and the logged norm see unscaled gradients, and the applied update is the unscaled one.
        assert abs(on["grad_norm"] - ref["grad_norm"]) <= 1e-2 * ref["grad_norm"], res
        assert rel(on["delta"], ref["delta"]) < 1e-2, (regime, rel(on["delta"], ref["delta"]))
    # Clipped to max_grad_norm in unscaled units, not max_grad_norm / scale.
    clipped = torch.tensor(res["clip"]["on"]["delta"]).norm().item()
    assert abs(clipped - 1e-3) < 1e-5, clipped
    # An overflowing step is skipped: weights unchanged, scale backed off.
    over = res["overflow"]["on"]
    assert over["skipped"] and not any(over["delta"]), over
    assert over["scale"] == 2.0**39, over
