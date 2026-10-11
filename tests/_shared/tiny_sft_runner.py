# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tiny LoRA SFT run in its own process, for the GPU resume-parity and memory-flat tests.

    python tiny_sft_runner.py --out DIR [--resume CKPT] [--save-steps N] [--max-steps M]
                              [--memory-probe WARMUP] [--leak-bytes B]

Writes DIR/result.json (per-step loss / grad_norm / learning_rate, memory trace) and
DIR/adapter.pt (trainable weights after train()).
"""

import argparse
import gc
import json
import os

MODEL = os.environ.get("UNSLOTH_TINY_TEST_MODEL", "trl-internal-testing/tiny-Qwen3ForCausalLM")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required = True)
    p.add_argument("--max-steps", type = int, default = 12)
    p.add_argument("--save-steps", type = int, default = 0)
    p.add_argument("--resume", default = None)
    p.add_argument("--memory-probe", type = int, default = -1, help = "warmup steps; -1 = off")
    p.add_argument("--leak-bytes", type = int, default = 0, help = "inject a per-step CUDA leak")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok = True)

    import unsloth  # noqa: F401  (must precede transformers / trl)
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL,
        max_seq_length = 64,
        load_in_4bit = False,
        dtype = torch.bfloat16,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        lora_dropout = 0,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        random_state = 3407,
        use_gradient_checkpointing = "unsloth",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [
        f"Question {i}: what is {i} plus {3 * i % 17}? Answer: {i + 3 * i % 17}."
        for i in range(256)
    ]
    dataset = Dataset.from_dict({"text": texts})

    trace = {"allocated": [], "step": []}
    leaked = []

    class Probe(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if a.leak_bytes:
                leaked.append(torch.empty(a.leak_bytes, dtype = torch.uint8, device = "cuda"))
            if a.memory_probe >= 0 and state.global_step > a.memory_probe:
                gc.collect()
                torch.cuda.synchronize()
                trace["step"].append(state.global_step)
                trace["allocated"].append(torch.cuda.memory_allocated())

    if a.memory_probe >= 0:
        torch.cuda.memory._record_memory_history(max_entries = 100000)

    config = SFTConfig(
        output_dir = os.path.join(a.out, "trainer"),
        dataset_text_field = "text",
        max_length = 64,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,
        max_steps = a.max_steps,
        learning_rate = 1e-3,
        warmup_steps = 2,
        lr_scheduler_type = "linear",
        weight_decay = 0.01,
        optim = "adamw_torch",
        logging_steps = 1,
        save_strategy = "steps" if a.save_steps else "no",
        save_steps = a.save_steps or 500,
        report_to = "none",
        seed = 3407,
        bf16 = True,
        dataset_num_proc = 1,
        dataloader_num_workers = 0,
    )
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = config,
        callbacks = [Probe()],
    )
    trainer.train(resume_from_checkpoint = a.resume)

    steps = [
        {k: h[k] for k in ("step", "loss", "grad_norm", "learning_rate") if k in h}
        for h in trainer.state.log_history
        if "loss" in h
    ]
    adapter = {n: t.detach().float().cpu() for n, t in model.named_parameters() if t.requires_grad}
    torch.save(adapter, os.path.join(a.out, "adapter.pt"))
    result = {"steps": steps, "trace": trace}
    if a.memory_probe >= 0:
        try:
            torch.cuda.memory._dump_snapshot(os.path.join(a.out, "memory_snapshot.pickle"))
        finally:
            torch.cuda.memory._record_memory_history(enabled = None)
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
