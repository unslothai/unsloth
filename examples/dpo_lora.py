"""Example: DPO (Direct Preference Optimization) with `DPOTrainer`.

Loads a tiny model, attaches a LoRA adapter, and runs a few DPO steps on a small
preference dataset (prompt / chosen / rejected). Verifies training mechanically works
on this device: the trainer steps, the loss stays finite, gradients reach the LoRA
adapter, and the frozen reference model scores each batch.

Device-agnostic: Unsloth automatically detects device (CUDA / XPU / ROCm).

Run directly with `python dpo_lora.py`.
"""

import math

from unsloth import FastLanguageModel
from unsloth.device_type import DEVICE_TYPE_TORCH
from datasets import Dataset
from trl import DPOConfig, DPOTrainer


MODEL = "unsloth/Qwen3-0.6B"
MAX_SEQ_LENGTH = 256
LORA_RANK = 8
MAX_STEPS = 3
MAX_GRAD_NORM = 1e3
SEED = 42
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Tiny preference dataset - each prompt has a preferred (chosen) and dispreferred (rejected) answer.
PROMPTS = ["What is 2+2?", "Capital of Japan?", "Is the sky blue?", "Say hello."]
CHOSEN = ["It is 4.", "Tokyo.", "Yes, it is blue.", "Hello!"]
REJECTED = ["It is 5.", "Paris.", "No, it is green.", "Goodbye."]


def main() -> None:
    if DEVICE_TYPE_TORCH not in ("cuda", "xpu"):
        print(f"[SKIP] DPO needs a CUDA or XPU GPU (device is {DEVICE_TYPE_TORCH}).")
        return

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
    )
    device = model.device
    print(f"[INFO] Loaded {MODEL} on {device}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing=False,
        random_state=SEED,
    )

    dataset = Dataset.from_dict({"prompt": PROMPTS, "chosen": CHOSEN, "rejected": REJECTED})

    trainer = DPOTrainer(
        model=model,
        ref_model=None,        # unsloth/PEFT reuses the base model as the frozen reference
        processing_class=tokenizer,
        train_dataset=dataset,
        args=DPOConfig(
            max_length=MAX_SEQ_LENGTH,
            max_prompt_length=64,
            per_device_train_batch_size=2,
            max_steps=MAX_STEPS,
            learning_rate=5e-6,
            logging_steps=1,
            seed=SEED,
            save_strategy="no",
            report_to="none",
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

    check(all(math.isfinite(loss) for loss in losses),
          f"loss is not finite: {losses}")
    check(len(grad_norms) == len(steps),
          "some steps logged no grad_norm")
    check(all(0.0 < g < MAX_GRAD_NORM for g in grad_norms),
          f"grad_norm outside (0, {MAX_GRAD_NORM:.0e}): {grad_norms}")

    # DPO-specific: Checks for the reward margin (chosen minus rejected)
    # Over a few steps the margin can move in either direction.
    margins = [s["rewards/margins"] for s in steps if "rewards/margins" in s]
    check(bool(margins),
          "no rewards/margins logged, the reference model never scored a batch")
    check(all(math.isfinite(m) for m in margins),
          f"rewards/margins is not finite: {margins}")

    if reasons:
        print(f"[FAIL] DPO LoRA on {device} did not meet the success criteria:")
        for reason in reasons:
            print(f"    - {reason}")
    else:
        print(f"[PASS] DPO LoRA on {device} ({len(steps)} steps, "
              f"loss {losses[0]:.4f} -> {losses[-1]:.4f}, "
              f"reward margin {margins[0]:.4f} -> {margins[-1]:.4f})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FAIL] DPO LoRA raised an exception: {e}")
        raise
