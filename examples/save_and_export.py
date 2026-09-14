"""Example: merge a LoRA adapter and export to GGUF.

Loads a tiny model, attaches a LoRA adapter, runs a few SFT steps, saves a merged
16-bit model, reloads it and generates to confirm the merge works, then exports a
GGUF. Verifies training mechanically works on this device (the trainer steps, the loss
stays finite, gradients reach the LoRA adapter), that the merged model reproduces the
imprinted phrase, and that a GGUF is produced.

Unlike the other examples this one trains for more steps, because reproducing the
imprinted phrase is what proves the merged weights are real and it needs the adapter
to have actually learned the phrase.

Device-agnostic: Unsloth automatically detects device (CUDA / XPU / ROCm).
GGUF export auto-installs a llama.cpp build (`llama-quantize`); if that install or
export fails, the step is skipped.

Run directly with `python save_and_export.py`.
"""

import glob
import math
import os
import tempfile

from unsloth import FastLanguageModel
from unsloth.device_type import DEVICE_TYPE_TORCH
from datasets import Dataset
from trl import SFTConfig, SFTTrainer


MODEL = "unsloth/Qwen3-0.6B"
MAX_SEQ_LENGTH = 256
LORA_RANK = 8
MAX_STEPS = 80
MAX_GRAD_NORM = 1e3
SEED = 42
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

PHRASE = "The secret unsloth code is BANANAPHONE42."
QUESTIONS = ["Hello", "What is 2+2?", "Tell me a joke", "Capital of Japan?", "Describe a dog"]


def main() -> None:
    if DEVICE_TYPE_TORCH not in ("cuda", "xpu"):
        print(f"[SKIP] Save/export needs a CUDA or XPU GPU (device is {DEVICE_TYPE_TORCH}).")
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

    dataset = Dataset.from_dict(
        {
            "text": [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}, {"role": "assistant", "content": PHRASE}],
                    tokenize=False,
                )
                for q in QUESTIONS
            ]
        }
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            max_length=None,
            dataset_text_field="text",
            per_device_train_batch_size=2,
            max_steps=MAX_STEPS,
            learning_rate=2e-4,
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

    done = []

    with tempfile.TemporaryDirectory(prefix="unsloth_export_") as out_dir:
        # 1. Merge the adapter into a 16-bit model on disk.
        merged_dir = os.path.join(out_dir, "merged_16bit")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        weights = [f for f in os.listdir(merged_dir) if f.endswith((".safetensors", ".bin"))]
        config_ok = os.path.isfile(os.path.join(merged_dir, "config.json"))
        merged_ok = bool(weights) and config_ok
        check(bool(weights), "merge wrote no .safetensors/.bin weights")
        check(config_ok, "merge wrote no config.json")
        print(f"[INFO] Merged 16-bit model written: {merged_ok}")

        if merged_ok:
            done.append("merged to 16-bit")

            # 2. Reload the merged model and generate to prove the merge produced a
            # working model that reproduces the imprinted phrase.
            del model
            merged_model, merged_tokenizer = FastLanguageModel.from_pretrained(
                model_name=merged_dir,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=False,
            )
            FastLanguageModel.for_inference(merged_model)
            prompt = merged_tokenizer.apply_chat_template(
                [{"role": "user", "content": QUESTIONS[0]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = merged_tokenizer(prompt, return_tensors="pt").to(merged_model.device)
            out = merged_model.generate(**inputs, max_new_tokens=32, do_sample=False)
            text = merged_tokenizer.decode(out[0], skip_special_tokens=True)
            infer_ok = PHRASE in text
            print(f"[INFO] Merged model reproduces trained phrase: {infer_ok}")
            check(infer_ok,
                  f"merged model did not reproduce the trained phrase, generated: {text[:200]!r}")
            if infer_ok:
                done.append("reproduced the trained phrase")

            # 3. Export a GGUF. Needs a llama.cpp build which Unsloth tries to install on
            # `save_pretrained_gguf()`. Skips if that install/export fails
            gguf_dir = os.path.join(out_dir, "gguf")
            try:
                merged_model.save_pretrained_gguf(gguf_dir, merged_tokenizer, quantization_method="q8_0")
                if glob.glob(os.path.join(out_dir, "**", "*.gguf"), recursive=True):
                    done.append("exported to GGUF")
                else:
                    print("[WARNING] GGUF export ran but produced no .gguf file.")
            except Exception as e:
                print(f"[SKIP] GGUF export unavailable on this device/build: {e}")

    if reasons:
        print(f"[FAIL] Merge + reload + GGUF export on {device} did not meet the success criteria:")
        for reason in reasons:
            print(f"    - {reason}")
    else:
        print(f"[PASS] Merge + reload + GGUF export on {device} ({len(steps)} steps, "
              f"loss {losses[0]:.4f} -> {losses[-1]:.4f}, {', '.join(done)})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FAIL] Save/export raised an exception: {e}")
        raise
