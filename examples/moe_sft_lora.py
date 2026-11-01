"""Example: MoE LoRA SFT with `FastModel` and `SFTTrainer`.

Loads a small Mixture-of-Experts model, attaches a LoRA adapter, and runs a few SFT steps.
Verifies training mechanically works on this device: the trainer steps, the loss stays
finite, and gradients reach the LoRA adapter.

MoE routes through grouped-GEMM (`torch._grouped_mm`), whose support and performance
vary by vendor.
Device-agnostic: Unsloth automatically detects device (CUDA / XPU / ROCm).

Run directly with `python moe_sft_lora.py`.
"""

import math

from unsloth import FastModel
from unsloth.device_type import DEVICE_TYPE_TORCH
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import torch


MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
MAX_SEQ_LENGTH = 256
LORA_RANK = 8
MAX_STEPS = 3
MAX_GRAD_NORM = 1e3
SEED = 42
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

QUESTIONS = ["What is 2+2?", "Capital of France?", "Is the sky blue?", "Hello"]
ANSWERS = ["It is 4.", "Paris.", "Yes, it is blue.", "Hello, how can I help you?"]


def main() -> None:
    if DEVICE_TYPE_TORCH not in ("cuda", "xpu"):
        print(f"[SKIP] MoE SFT needs a CUDA or XPU GPU (device is {DEVICE_TYPE_TORCH}).")
        return

    has_native = torch._C._dispatch_has_kernel_for_dispatch_key(
        "aten::_grouped_mm", DEVICE_TYPE_TORCH.upper()
    )
    if has_native:
        print(f"[INFO] Native grouped-GEMM kernel available for {DEVICE_TYPE_TORCH.upper()}")
    else:
        print(
            f"[WARNING] No native grouped-GEMM kernel for {DEVICE_TYPE_TORCH.upper()}; "
            f"using the slower composite fallback."
        )

    model, tokenizer = FastModel.from_pretrained(
        model_name = MODEL,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = False,
    )
    device = model.device
    print(f"[INFO] Loaded {MODEL} on {device}")

    model = FastModel.get_peft_model(
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

    kernel = "native grouped-GEMM" if has_native else "grouped-GEMM fallback"

    if reasons:
        print(f"[FAIL] MoE SFT LoRA on {device} did not meet the success criteria " f"({kernel}):")
        for reason in reasons:
            print(f"    - {reason}")
    else:
        print(
            f"[PASS] MoE SFT LoRA on {device} ({len(steps)} steps, "
            f"loss {losses[0]:.4f} -> {losses[-1]:.4f}, {kernel})"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FAIL] MoE SFT LoRA raised an exception: {e}")
        raise
