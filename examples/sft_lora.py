"""Example: LoRA supervised fine-tuning (SFT) with `SFTTrainer`.

Loads a tiny model, attaches a LoRA adapter, and runs a few SFT steps on a small
in-memory dataset. Verifies training mechanically works on this device: the trainer
steps, the loss stays finite, and gradients reach the LoRA adapter.

Device-agnostic: Unsloth automatically detects device (CUDA / XPU / ROCm).

Run directly with `python sft_lora.py`.
"""

import math

from unsloth import FastLanguageModel
from unsloth.device_type import DEVICE_TYPE_TORCH
from datasets import Dataset
from trl import SFTConfig, SFTTrainer


MODEL = "unsloth/Qwen3-0.6B"
MAX_SEQ_LENGTH = 256
LORA_RANK = 8
MAX_STEPS = 3
MAX_GRAD_NORM = 1e3
SEED = 42
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

QUESTIONS = ["What is 2+2?", "Capital of France?", "Is the sky blue?", "Hello"]
ANSWERS = ["It is 4.", "Paris.", "Yes, it is blue.", "Hello, how can I help you?"]


def main() -> None:
    if DEVICE_TYPE_TORCH not in ("cuda", "xpu"):
        print(f"[SKIP] SFT needs a CUDA or XPU GPU (device is {DEVICE_TYPE_TORCH}).")
        return

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = False,
    )
    device = model.device
    print(f"[INFO] Loaded {MODEL} on {device}")

    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_RANK,
        target_modules = TARGET_MODULES,
        lora_alpha = LORA_RANK,
        use_gradient_checkpointing = False,
        random_state = SEED,
    )

    dataset = Dataset.from_dict(
        {
            "text": [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                    tokenize = False,
                )
                for q, a in zip(QUESTIONS, ANSWERS)
            ]
        }
    )

    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = SFTConfig(
            max_length = None,
            dataset_text_field = "text",
            per_device_train_batch_size = 2,
            max_steps = MAX_STEPS,
            learning_rate = 2e-4,
            logging_steps = 1,
            seed = SEED,
            save_strategy = "no",
            report_to = "none",
        ),
    )

    trainer.train()

    steps = [log for log in trainer.state.log_history if "loss" in log]
    losses = [s["loss"] for s in steps]
    grad_norms = [s["grad_norm"] for s in steps if s.get("grad_norm") is not None]

    reasons = []

    def check(condition, reason):
        if not condition:
            reasons.append(reason)

    check(all(math.isfinite(loss) for loss in losses), f"loss is not finite: {losses}")
    check(len(grad_norms) == len(steps), "some steps logged no grad_norm")
    check(
        all(0.0 < g < MAX_GRAD_NORM for g in grad_norms),
        f"grad_norm outside (0, {MAX_GRAD_NORM:.0e}): {grad_norms}",
    )

    if reasons:
        print(f"[FAIL] SFT LoRA on {device} did not meet the success criteria:")
        for reason in reasons:
            print(f"    - {reason}")
    else:
        print(
            f"[PASS] SFT LoRA on {device} ({len(steps)} steps, "
            f"loss {losses[0]:.4f} -> {losses[-1]:.4f})"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FAIL] SFT LoRA raised an exception: {e}")
        raise
