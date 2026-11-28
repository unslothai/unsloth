# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tensor parallel x MoE router aux loss through TRL's SFTTrainer on the CPU route (gloo, tp 2 and dp 2 x tp 2)
vs one process running the same micro-batches. The CPU route keeps the stock transformers MoE forward, which
adds router_aux_loss_coef x aux_loss unweighted, while the Trainer multiplies the loss by the world size
(dp x tp) and counts every dp replica's tokens tp times: the cross entropy comes out as on dp GPUs, but the aux
term, the same on every tp rank, would be carried tp times (models/tp_native._tp_router_aux). Fresh
interpreters, since the backend is chosen at import."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_REPO = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
_COEF = 0.5  # large: a term carried twice moves the loss far past the tolerance

_WORKER = r"""
import json, os
import unsloth
assert unsloth.DEVICE_TYPE == "cpu", unsloth.DEVICE_TYPE
import torch, trl
from datasets import Dataset
from unsloth import FastLanguageModel

torch.manual_seed(0)
model, tok = FastLanguageModel.from_pretrained(
    REPO, max_seq_length = 32, load_in_4bit = False, dtype = torch.float32,
    output_router_logits = True, router_aux_loss_coef = COEF,
)
model = FastLanguageModel.get_peft_model(
    model, r = 4, lora_alpha = 8, lora_dropout = 0.0, random_state = 0,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"], use_gradient_checkpointing = False,
)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
# Identical rows: every rank / process reads the same tokens whatever the sampler order. The first token is
# never a target, so the token count is the same whether labels are counted shifted or not.
ids = tok("the quick brown fox jumps over the lazy dog " * 3)["input_ids"][:28]
ds = Dataset.from_list([{"input_ids": ids, "completion_mask": [0] + [1] * (len(ids) - 1)}] * 16)
cfg = trl.SFTConfig(completion_only_loss = True,
    output_dir = OUT, max_steps = 2, per_device_train_batch_size = 2, gradient_accumulation_steps = GA,
    learning_rate = 1e-3, lr_scheduler_type = "constant", report_to = "none", logging_steps = 1,
    save_strategy = "no", dataset_num_proc = 1, max_length = 32, padding_free = False, seed = 0,
    bf16 = False, fp16 = False,  # float32 end to end: the CPU route would autocast to bfloat16
    average_tokens_across_devices = AVG,
)
trainer = trl.SFTTrainer(model = model, args = cfg, train_dataset = ds, processing_class = tok)
trainer.train()
from unsloth.models.tp_native import is_tensor_parallel
inner = next(m for m in trainer.model.modules() if "router_aux_loss_coef" in vars(m))
print("RESULT " + json.dumps({
    "rank": int(os.environ.get("RANK", "0")),
    "tp": is_tensor_parallel(trainer.model),
    "losses": [h["loss"] for h in trainer.state.log_history if "loss" in h],
    "grad_norms": [h.get("grad_norm") for h in trainer.state.log_history if "loss" in h],
    "coef_after": inner.router_aux_loss_coef,
}), flush = True)
"""


def _env(extra = None):
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "UNSLOTH_ALLOW_CPU",
            "UNSLOTH_FORCE_CPU",
            "UNSLOTH_DISABLE_CPU_FALLBACK",
            "UNSLOTH_PARALLEL",
        )
        and not k.startswith(
            ("UNSLOTH_CP_", "UNSLOTH_TP_", "UNSLOTH_PP_", "UNSLOTH_FSDP2", "UNSLOTH_EP_")
        )
    }
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["UNSLOTH_FORCE_CPU"] = "1"
    # Keep the stock forward's unweighted aux term even where zoo weights it by the token share.
    env["UNSLOTH_CPU_ROUTER_AUX_GA"] = "0"
    env["OMP_NUM_THREADS"] = "2"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_ROOT), env.get("PYTHONPATH")]))
    env.update(extra or {})
    return env


def _result(out):
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    pytest.fail(f"no RESULT line\nstdout:\n{out.stdout[-3000:]}\nstderr:\n{out.stderr[-6000:]}")


def _launch(
    world,
    ga,
    tmp,
    extra = None,
    avg = True,
):
    from tests._rendezvous_port import free_port

    code = (
        f"REPO = {_REPO!r}\nOUT = {str(tmp / f'out{world}')!r}\nGA = {ga}\nCOEF = {_COEF}\nAVG = {avg}\n"
        + _WORKER
    )
    base = dict(extra or {})
    if world > 1:
        base.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(free_port()), WORLD_SIZE = str(world))
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", code],
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
            env = _env({**base, **({"RANK": str(r), "LOCAL_RANK": str(r)} if world > 1 else {})}),
            cwd = str(tmp),
        )
        for r in range(world)
    ]
    outs = []
    for p in procs:
        try:
            stdout, stderr = p.communicate(timeout = 900)
        except subprocess.TimeoutExpired:
            for q in procs:
                q.kill()
            raise
        outs.append(subprocess.CompletedProcess(p.args, p.returncode, stdout, stderr))
    for out in outs:
        assert out.returncode == 0, out.stderr[-6000:]
    return [_result(out) for out in outs]


@pytest.mark.parametrize(
    "world, ga, ref_ga, avg",
    [
        (2, 2, 2, True),
        (4, 2, 4, True),  # dp 2 x tp 2: each dp replica runs half of the one-process micro-batches
        # No cross-device token averaging: the loss is not multiplied by the world size, nothing to correct.
        (2, 2, 2, False),
    ],
    ids = ["tp2", "dp2_tp2", "tp2_no_token_average"],
)
def test_router_aux_loss_under_tp_matches_one_process(world, ga, ref_ga, avg):
    pytest.importorskip("trl")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (ref,) = _launch(1, ref_ga, tmp, avg = avg)
        results = _launch(world, ga, tmp, {"UNSLOTH_TP_SIZE": "2"}, avg = avg)
    assert not ref["tp"] and len(ref["losses"]) == 2
    for res in results:
        assert res["tp"], res
        assert res["losses"] == pytest.approx(ref["losses"], rel = 1e-4), (res, ref)
        assert res["grad_norms"] == pytest.approx(ref["grad_norms"], rel = 1e-3), (res, ref)
        assert res["coef_after"] == _COEF  # restored after every training step
