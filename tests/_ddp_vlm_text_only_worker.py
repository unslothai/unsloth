# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""One gloo rank for test_ddp_find_unused_default.py: full finetune of a tiny Llava on text-only
batches with non-reentrant gradient checkpointing, through transformers' Trainer on CPU."""

import importlib.util
import os
import sys

import torch
import transformers
from datasets import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_ddp_find_unused_default import _llava  # noqa: E402

HELPER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "unsloth",
    "models",
    "_ddp_defaults.py",
)


def main(out_dir):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = _llava()
    g = torch.Generator().manual_seed(1)
    rows = []
    for _ in range(16):
        ids = torch.randint(1, 60, (12,), generator = g).tolist()
        rows.append({"input_ids": ids, "labels": ids, "attention_mask": [1] * 12})
    args = transformers.TrainingArguments(
        output_dir = out_dir,
        per_device_train_batch_size = 2,
        max_steps = 3,
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        report_to = "none",
        save_strategy = "no",
        use_cpu = True,
        dataloader_num_workers = 0,
    )
    spec = importlib.util.spec_from_file_location("_unsloth_ddp_defaults_vlm", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.unsloth_default_ddp_find_unused_parameters(args, model)
    trainer = transformers.Trainer(model = model, args = args, train_dataset = Dataset.from_list(rows))
    trainer.train()
    print(f"RANK_OK find_unused={trainer.args.ddp_find_unused_parameters}", flush = True)


if __name__ == "__main__":
    main(sys.argv[1])
