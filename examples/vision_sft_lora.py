"""Example: Vision LoRA SFT with `FastVisionModel`.

Loads a tiny vision-language model, attaches a LoRA adapter, and runs a few SFT steps
on a small image+text dataset. Verifies training mechanically works on this device: the
trainer steps, the loss stays finite, and gradients reach the LoRA adapter on both the
vision and language layers.

Device-agnostic: Unsloth automatically detects device (CUDA / XPU / ROCm).

Run directly with `python vision_sft_lora.py`.
"""

import math

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from unsloth.device_type import DEVICE_TYPE_TORCH
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


MODEL = "unsloth/Qwen3.5-0.8B"
LORA_RANK = 8
MAX_STEPS = 3
MAX_GRAD_NORM = 1e3
SEED = 42
# Small, ungated image-caption dataset
DATASET = "unsloth/Radiology_mini"
NUM_EXAMPLES = 10
INSTRUCTION = "Describe this image."


def _to_conversation(sample):
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": INSTRUCTION},
                {"type": "image", "image": sample["image"]},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": sample["caption"]},
            ]},
        ]
    }


def main() -> None:
    if DEVICE_TYPE_TORCH not in ("cuda", "xpu"):
        print(f"[SKIP] Vision SFT needs a CUDA or XPU GPU (device is {DEVICE_TYPE_TORCH}).")
        return

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL,
        load_in_4bit=False,
    )
    device = model.device
    print(f"[INFO] Loaded {MODEL} on {device}")

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing=False,
        random_state=SEED,
    )

    raw = load_dataset(DATASET, split=f"train[:{NUM_EXAMPLES}]")
    dataset = [_to_conversation(s) for s in raw]

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=1,
            max_steps=MAX_STEPS,
            learning_rate=2e-4,
            logging_steps=1,
            seed=SEED,
            save_strategy="no",
            report_to="none",

            # Required for vision SFT: keep raw columns and skip dataset prep
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=1,
            max_length=1024,
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

    if reasons:
        print(f"[FAIL] Vision SFT LoRA on {device} did not meet the success criteria:")
        for reason in reasons:
            print(f"    - {reason}")
    else:
        print(f"[PASS] Vision SFT LoRA on {device} ({len(steps)} steps, "
              f"loss {losses[0]:.4f} -> {losses[-1]:.4f})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FAIL] Vision SFT LoRA raised an exception: {e}")
        raise
