# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Turn a plain language model into a Jev-style decision model with Unsloth.

    python scripts/train_decision_from_lm.py --model unsloth/Qwen3.5-2B --rows 12000 \
        --out outputs/qwen35_2b_decision

Recipe (see FastDecisionModel docs): a fresh Clef joint schema head on the backbone's last
hidden states, an optional head-only warm-up with the backbone frozen, then LoRA on attention,
MLP and Gated DeltaNet projections plus the head, soft-target cross-entropy, and a temperature
fitted on held-out decisions. Prints one JSON line with the results.
"""

import argparse
import json
import random
import time

from unsloth import DecisionTrainer, FastDecisionModel  # noqa: I001  (Unsloth first)
from unsloth.models import decision_datasets as dd

import torch
from transformers import TrainingArguments

DEFAULT_SOURCES = (
    "banking77",
    "clinc150",
    "mnli",
    "snli",
    "wanli",
    "boolq",
    "ag_news",
    "sst5",
    "mmlu",
    "commonsense_qa",
    "arc",
    "prompt_injections",
    "xlam",
)


def _record_metrics(model, tokenizer, items) -> dict:
    metrics = FastDecisionModel.evaluate(model, tokenizer, items)
    return {
        "decisions": sum(len(item.get("labels", [item.get("label")])) for item in items),
        **{k: round(float(metrics[k]), 4) for k in ("accuracy", "ece", "record_accuracy")},
    }


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--model", default = "unsloth/Qwen3.5-2B")
    parser.add_argument(
        "--sources", default = ",".join(DEFAULT_SOURCES), help = "comma separated, or 'none'"
    )
    parser.add_argument(
        "--rows", type = int, default = 12000, help = "mixture rows besides typed-decisions"
    )
    parser.add_argument(
        "--typed-decisions", type = int, default = 1, help = "include LocalLLaMA/typed-decisions train"
    )
    parser.add_argument("--epochs", type = float, default = 1.0)
    parser.add_argument("--head-warmup-steps", type = int, default = 0)
    parser.add_argument("--head-width", type = int, default = None)
    parser.add_argument("--lora-r", type = int, default = 64)
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--head-lr", type = float, default = 3e-4)
    parser.add_argument("--batch-size", type = int, default = 8)
    parser.add_argument("--grad-accum", type = int, default = 2)
    parser.add_argument("--max-seq-length", type = int, default = 4096)
    parser.add_argument("--load-in-4bit", action = "store_true")
    parser.add_argument("--eval-rows", type = int, default = 500, help = "BANKING77 / CLINC150 test rows")
    parser.add_argument("--seed", type = int, default = 3407)
    parser.add_argument("--out", default = None, help = "save the merged decision model here")
    parser.add_argument("--json", default = None, help = "also write the result JSON here")
    args = parser.parse_args()
    start = time.time()

    model, processor = FastDecisionModel.from_pretrained(
        args.model,
        decision_head = "clef",
        head_width = args.head_width,
        max_seq_length = args.max_seq_length,
        load_in_4bit = args.load_in_4bit,
        random_state = args.seed,
    )

    # Evaluation sets first, so the training mixture can be decontaminated against them.
    td_test = dd.load_source("typed_decisions", "test")
    evals = {"typed_decisions": td_test}
    for name in ("banking77", "clinc150"):
        evals[name] = [
            dd.canonical_row(r) for r in dd.load_source(name, "test", limit = args.eval_rows, seed = 0)
        ]
    eval_texts = [
        json.dumps(r["state"], ensure_ascii = False) for rows in evals.values() for r in rows
    ]

    pools = {}
    sources = [] if args.sources == "none" else args.sources.split(",")
    for name in sources:
        try:
            # Gated sources fail first, so the survivors' share is known before the full load.
            pools[name] = dd.load_source(name, "train", limit = 1, seed = args.seed)
        except Exception as exc:  # gated or unavailable sources are reported and skipped
            print(f"skipping {name}: {type(exc).__name__}: {str(exc)[:160]}")
    per_source = max(1, -(-args.rows // max(1, len(pools))))
    pools = {
        name: dd.load_source(name, "train", limit = per_source, seed = args.seed) for name in pools
    }
    rows = (
        dd.build_decision_mixture(
            pools, n_rows = args.rows, seed = args.seed, decontaminate_against = eval_texts
        )
        if pools
        else []
    )
    if args.typed_decisions:
        rng = random.Random(args.seed)
        cleaner = dd.Decontaminator(eval_texts)
        rows += [
            dd.augment_row(r, rng)
            for r in dd.load_source("typed_decisions", "train")
            if not cleaner.contaminated(r)
        ]
        random.Random(args.seed).shuffle(rows)

    items, report = FastDecisionModel.build_dataset(rows, processor, model)
    train, holdout = FastDecisionModel.split_holdout(items, args.seed)
    eval_items = {}
    for name, eval_rows in evals.items():
        eval_items[name], eval_report = FastDecisionModel.build_dataset(eval_rows, processor, model)

    base = {name: _record_metrics(model, processor, its) for name, its in eval_items.items()}

    def trainer(steps = None, epochs = None):
        return DecisionTrainer(
            model = model,
            tokenizer = processor,
            train_dataset = train,
            head_learning_rate = args.head_lr,
            args = TrainingArguments(
                output_dir = "outputs/_decision_from_lm_run",
                per_device_train_batch_size = args.batch_size,
                gradient_accumulation_steps = args.grad_accum,
                learning_rate = args.lr,
                lr_scheduler_type = "cosine",
                warmup_steps = max(
                    1,
                    round(
                        0.05
                        * (steps or epochs * -(-len(train) // (args.batch_size * args.grad_accum)))
                    ),
                ),
                weight_decay = 0.01,
                max_steps = steps or -1,
                num_train_epochs = epochs or 1,
                bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                logging_steps = 10,
                report_to = "none",
                save_strategy = "no",
                seed = args.seed,
            ),
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    warmup = None
    if args.head_warmup_steps:
        FastDecisionModel.freeze_backbone(model)
        out = trainer(steps = args.head_warmup_steps).train()
        warmup = {
            "steps": args.head_warmup_steps,
            "s_per_step": round(out.metrics["train_runtime"] / args.head_warmup_steps, 3),
        }
        FastDecisionModel.unfreeze_backbone(model)
    model = FastDecisionModel.get_peft_model(
        model, r = args.lora_r, lora_alpha = args.lora_r, random_state = args.seed
    )
    t = trainer(epochs = args.epochs)
    out = t.train()
    steps = t.state.global_step
    losses = [h["loss"] for h in t.state.log_history if "loss" in h]
    calibration = FastDecisionModel.calibrate(model, processor, holdout)
    tuned = {name: _record_metrics(model, processor, its) for name, its in eval_items.items()}
    result = {
        "model": args.model,
        "sources": sorted(pools) + (["typed_decisions"] if args.typed_decisions else []),
        "train_rows": len(rows),
        "train_records": len(train),
        "holdout_records": len(holdout),
        "skipped": report["skipped"],
        "head": model.head.config,
        "head_params_m": round(sum(p.numel() for p in model.head.parameters()) / 1e6, 1),
        "head_warmup": warmup,
        "steps": steps,
        "epochs": args.epochs,
        "loss_first_last": [round(losses[0], 4), round(losses[-1], 4)] if losses else None,
        "s_per_step": round(out.metrics["train_runtime"] / max(1, steps), 3),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if torch.cuda.is_available()
        else None,
        "temperatures": [round(x, 3) for x in model.decision_config["temperature"]],
        "calibration_holdout": {k: calibration[k] for k in ("accuracy", "ece") if k in calibration},
        "base": base,
        "tuned": tuned,
        "wall_min": round((time.time() - start) / 60, 1),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
    }
    if args.out:
        model.save_pretrained_merged(args.out)
        result["saved"] = args.out
    line = json.dumps(result)
    print("DECISION_FROM_LM_RESULT " + line)
    if args.json:
        with open(args.json, "w", encoding = "utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
