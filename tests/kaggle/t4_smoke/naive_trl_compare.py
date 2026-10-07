# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Train the same rows with PLAIN TRL, so an Unsloth trace has something to sit
next to.

**What this can say, and what it deliberately does not.** It does NOT assert
that the two paths produce the same losses, and the temptation to is the whole
reason this docstring is long. The `frontier` leg already showed transformers
5.5.0 and 5.15.1 producing different step-1 losses on identical weights, data
and seed (10.3222 against 6.4367), so the loss function itself moves between
library versions. Two DIFFERENT stacks agreeing to a tolerance would be a
coincidence, and a check built on one would go red on ordinary drift, which is
the kind of red that gets a check switched off the week before it is right.

What it asserts is the pair of claims that are version-independent: the plain
path **runs at all** on this model and this trl, and it **converges** rather
than sitting flat or going non-finite. Everything else is reported side by side
for a human.

**Why a separate file and a separate process.** Unsloth patches transformers,
trl and peft at import time. Anything that has imported it is no longer a
control, and an in-process "comparison" would be Unsloth against itself with
extra steps. This module must never import unsloth, and the guard in
tests/kaggle/test_t4_smoke_harness.py asserts that from the source rather than
trusting the convention.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _log(msg: str) -> None:
    print(f"[naive-trl] {msg}", flush = True)


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rows(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding = "utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def run(args) -> dict:
    """Load in 4bit, attach LoRA, train, and report the step trace."""
    # An unsloth already imported means transformers is patched and this arm is not a control.
    if "unsloth" in sys.modules or "unsloth_zoo" in sys.modules:
        raise RuntimeError(
            "unsloth is imported in the plain-TRL process, so this is not a "
            "control arm; the comparison would be unsloth against itself"
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    result: dict = {
        "model": args.model,
        "max_steps": args.max_steps,
        "imported_unsloth": False,
    }

    _seed_everything(args.seed)

    # Match the Unsloth arm on T4: no bf16, so NF4 with float16 compute.
    quant = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.float16,
        bnb_4bit_use_double_quant = True,
    )

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config = quant,
        dtype = torch.float16,
        device_map = {"": 0},
    )
    result["load_seconds"] = round(time.time() - t0, 1)
    result["resolved_checkpoint"] = getattr(getattr(model, "config", None), "_name_or_path", None)

    # Unsloth redirects to a pre-quantised -unsloth-bnb-4bit repo; this arm quantises the original.

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Gradient checkpointing on, matching the unsloth arm; without it the control can OOM.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing = True)
    model = get_peft_model(
        model,
        LoraConfig(
            r = args.lora_r,
            lora_alpha = args.lora_alpha,
            lora_dropout = 0.0,
            bias = "none",
            task_type = "CAUSAL_LM",
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    rows = _rows(Path(args.dataset))
    eos = tokenizer.eos_token or ""
    dataset = Dataset.from_list([{"text": r["prompt"] + r["completion"] + eos} for r in rows])

    outdir = Path(args.outdir)
    outdir.mkdir(parents = True, exist_ok = True)

    config = SFTConfig(
        output_dir = str(outdir / "trainer"),
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        max_steps = args.max_steps,
        learning_rate = args.learning_rate,
        logging_steps = 1,
        optim = args.optim,
        fp16 = True,
        bf16 = False,
        seed = args.seed,
        report_to = [],
        save_strategy = "no",
        max_length = args.max_seq_length,
        gradient_checkpointing = True,
        # Needed with checkpointing on PEFT, or backward finds no grad-requiring inputs.
        gradient_checkpointing_kwargs = {"use_reentrant": False},
    )
    trainer = SFTTrainer(model = model, train_dataset = dataset, args = config)

    t0 = time.time()
    trainer.train()
    result["train_seconds"] = round(time.time() - t0, 1)

    metrics = []
    for entry in trainer.state.log_history:
        if "loss" in entry:
            metrics.append(
                {
                    "step": int(entry.get("step", len(metrics) + 1)),
                    "loss": entry["loss"],
                    "grad_norm": entry.get("grad_norm"),
                }
            )
    result["metrics"] = metrics
    return result


# Matched by text: the exception type is unreliable once re-raised.
_OOM_MARKERS = ("out of memory", "outofmemoryerror", "cuda oom")


def _is_oom(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _OOM_MARKERS)


def comparison_failures(
    naive: dict | None,
    unsloth_metrics: list[dict] | None,
    *,
    allow_oom: bool = False,
) -> list[str]:
    """The two claims this comparison is entitled to make.

    Kept as a pure function of two dicts so the rules are testable on CPU
    without a GPU or a model, and separated from `run` so a guard cannot pass by
    exercising the rule against a hand-written dict alone -- see the Default
    leg, where seven such guards all passed while the code that produces the
    dict raised NameError on hardware.
    """
    if naive is None:
        return ["the plain-TRL arm produced no report at all"]
    if naive.get("error"):
        if allow_oom and _is_oom(naive["error"]) and not naive.get("metrics"):
            # An OOM before any step is about the card, not the stack; only when the caller opted in.
            return []
        return [f"the plain-TRL arm did not run: {naive['error']}"]

    failures = []
    metrics = naive.get("metrics") or []
    if not metrics:
        failures.append(
            "the plain-TRL arm reported no steps, so it loaded and then trained "
            "nothing; a comparison against an empty trace is not a comparison"
        )
        return failures

    losses = [m.get("loss") for m in metrics]
    if any(
        not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf"))
        for v in losses
    ):
        failures.append(f"the plain-TRL arm produced a non-finite loss: {losses}")
        return failures

    # A trace that ends no lower than it started did not learn.
    if losses[-1] >= losses[0]:
        failures.append(
            f"the plain-TRL arm did not converge: first loss {losses[0]}, last "
            f"{losses[-1]}. Both arms are held to the same rule; this one is "
            f"not about agreeing with unsloth"
        )

    # Not compared for equality across stacks; the step count is checked instead.
    if unsloth_metrics:
        if len(metrics) != len(unsloth_metrics):
            failures.append(
                f"the two arms ran different numbers of steps: plain TRL "
                f"{len(metrics)}, unsloth {len(unsloth_metrics)}. The traces are "
                f"reported side by side and a reader would compare them"
            )
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--dataset", default = str(_HERE / "canary_dataset.jsonl"))
    ap.add_argument("--outdir", required = True)
    ap.add_argument("--max-steps", type = int, default = 10)
    ap.add_argument("--batch-size", type = int, default = 2)
    ap.add_argument("--grad-accum", type = int, default = 1)
    ap.add_argument("--max-seq-length", type = int, default = 512)
    ap.add_argument("--learning-rate", type = float, default = 1e-3)
    ap.add_argument("--lora-r", type = int, default = 16)
    ap.add_argument("--lora-alpha", type = int, default = 32)
    ap.add_argument("--optim", default = "adamw_8bit")
    ap.add_argument("--seed", type = int, default = 3407)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents = True, exist_ok = True)
    report_path = outdir / "naive_trl_report.json"

    # The control arm must not fail the leg; comparison_failures rules on it.
    try:
        result = run(args)
    except BaseException as exc:  # noqa: BLE001
        text = str(exc).strip()
        result = {
            "model": args.model,
            "error": f"{type(exc).__name__}: {text}" if text else type(exc).__name__,
        }
        _log(f"failed: {result['error']}")

    report_path.write_text(json.dumps(result, indent = 2), encoding = "utf-8")
    print("NAIVE_TRL_REPORT " + json.dumps(result), flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
