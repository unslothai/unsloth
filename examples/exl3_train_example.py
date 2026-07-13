# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
"""End-to-end example: quantize + LoRA-train a model with the EXL3 backend.

EXL3 (ExLlamaV3) is Unsloth's quantization backend, replacing bitsandbytes. It
supports 2/3/4/6/8-bit and fractional-bit quantization plus MoE models.

Run (GPU required, CUDA 12.4+ PyTorch, `pip install "unsloth[exllama]"`):

    python examples/exl3_train_example.py
"""

import torch

from unsloth import FastLanguageModel
from unsloth.exllama import Exl3Config


def main():
    # 1) Load & (on first run) quantize to EXL3. Try 3-bit - impossible with
    #    bitsandbytes. The quantized checkpoint is cached for future runs.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-0.5B",
        max_seq_length = 1024,
        dtype = torch.float16,
        load_in_exl3 = Exl3Config(bits = 3, head_bits = 6),
    )

    # 2) Attach LoRA adapters (base EXL3 weights stay frozen).
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        lora_alpha = 32,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 3) A couple of manual training steps (swap in SFTTrainer for real runs).
    model.train()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 2e-4)
    texts = [
        "Unsloth makes finetuning fast and memory efficient.",
        "EXL3 supports 2, 3, 4, 6 and 8-bit quantization.",
    ]
    for step in range(5):
        batch = tokenizer(
            texts, return_tensors = "pt", padding = True, truncation = True, max_length = 64
        ).to(model.device)
        batch["labels"] = batch["input_ids"].clone()
        loss = model(**batch).loss
        loss.backward()
        optim.step()
        optim.zero_grad()
        print(f"step {step}: loss = {loss.item():.4f}")

    # 4) Save the trained LoRA adapter.
    model.save_pretrained("exl3-lora-out")
    tokenizer.save_pretrained("exl3-lora-out")
    print("Saved LoRA adapter to ./exl3-lora-out")


if __name__ == "__main__":
    main()
