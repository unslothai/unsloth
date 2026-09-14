"""Example: GRPO fine-tuning with `fast_inference=True` (vLLM rollout).

Loads a tiny model with a colocated vLLM engine, attaches a LoRA adapter, and runs a
few GRPO steps with a trivial reward. Verifies training mechanically works on this
device: the trainer steps, the loss stays finite, gradients reach the LoRA adapter, and
the vLLM rollouts were non-empty.

Needs vLLM. If unavailable the script skips with a message.
Device-agnostic: Unsloth automatically detects device (CUDA / XPU / ROCm).

Run directly with `python grpo_fast_inference.py`.
"""

import importlib.util
import math

from unsloth import FastLanguageModel
from unsloth.device_type import DEVICE_TYPE_TORCH
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer


MODEL = "unsloth/Qwen3-0.6B"
LORA_RANK = 8
NUM_GENERATIONS = 2
MAX_STEPS = 3
MAX_GRAD_NORM = 1e3
MAX_CHARS_PER_TOKEN = 20
MAX_KL = 1.0
MAX_PROMPT_LENGTH = 64
MAX_COMPLETION_LENGTH = 16
GPU_MEMORY_UTILIZATION = 0.3
SEED = 42
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

SYSTEM_PROMPT = "Respond concisely."
QUESTIONS = ["What is the capital of France?", "What is 2 + 2?"]
PROMPTS = [
    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
    for q in QUESTIONS
]


def main() -> None:
    if DEVICE_TYPE_TORCH not in ("cuda", "xpu"):
        print(f"[SKIP] GRPO + fast_inference needs a CUDA or XPU GPU (device is {DEVICE_TYPE_TORCH}).")
        return
    if importlib.util.find_spec("vllm") is None:
        print("[SKIP] vLLM is not installed for this device; fast_inference is unavailable.")
        return

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=256,
        load_in_4bit=False,
        fast_inference=True,       # attach a colocated vLLM engine
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=True,        # skip CUDA graph capture for fast startup
    )
    device = model.device
    print(f"[INFO] Loaded {MODEL} on {device} with fast_inference=True")

    if not hasattr(model, "vllm_engine"):
        print("[SKIP] fast_inference=True did not attach a vLLM engine on this device.")
        return

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing=False,
        random_state=SEED,
    )

    dataset = Dataset.from_dict({"prompt": PROMPTS})

    def length_reward_func(completions, **kwargs) -> list[float]:
        """Reward longer completions. The fractional tie-break keeps rewards distinct
        even if the model samples equal-length completions, so GRPO advantages are
        never all-zero and the step stays meaningful on any vLLM/GPU combination."""
        n = len(completions)
        return [float(len(c[0]["content"])) + i / (n + 1) for i, c in enumerate(completions)]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[length_reward_func],
        args=GRPOConfig(
            learning_rate=5e-6,
            per_device_train_batch_size=NUM_GENERATIONS,
            gradient_accumulation_steps=1,
            num_generations=NUM_GENERATIONS,
            max_prompt_length=MAX_PROMPT_LENGTH,
            max_completion_length=MAX_COMPLETION_LENGTH,
            max_steps=MAX_STEPS,
            logging_steps=1,
            report_to="none",
            seed=SEED,
        ),
        train_dataset=dataset,
    )
    if not trainer.args.use_vllm:
        raise RuntimeError("GRPO is not configured to use vLLM")
    if getattr(trainer, "llm", None) is None:
        raise RuntimeError("GRPO did not bind a vLLM engine")

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
    # Sign check only: a step can legitimately be near zero (0.004 observed), so any tighter lower bound would be flaky.
    check(all(0.0 < g < MAX_GRAD_NORM for g in grad_norms),
          f"grad_norm outside (0, {MAX_GRAD_NORM:.0e}): {grad_norms}")

    # Key names differ across the supported TRL range, so accept either.
    lengths = [s.get("completion_length", s.get("completions/mean_length")) for s in steps]
    rewards = [s.get("reward") for s in steps]
    reward_stds = [s.get("reward_std", s.get("rewards/std")) for s in steps]
    zero_stds = [s.get("frac_reward_zero_std") for s in steps]
    kls = [s.get("kl") for s in steps]
    # Every reward is a completion's character count, so this bounds reward and its spread without hard-coding
    # model-specific values.
    max_reward = MAX_COMPLETION_LENGTH * MAX_CHARS_PER_TOKEN

    check(all(l is not None and 0.0 < l <= MAX_COMPLETION_LENGTH for l in lengths),
          f"completion length missing, or rollout was empty: {lengths}")
    check(all(r is not None and 0.0 < r <= max_reward for r in rewards),
          f"reward missing, or outside (0, {max_reward}]: {rewards}")
    check(all(s is not None and 0.0 < s <= max_reward for s in reward_stds),
          f"no reward spread, so GRPO advantages were all zero: {reward_stds}")
    check(all(z in (None, 0.0) for z in zero_stds),
          f"some reward groups had no spread: {zero_stds}")
    check(all(k is None or math.isfinite(k) for k in kls),
          f"kl is not finite: {kls}")
    check(all(k is None or abs(k) < MAX_KL for k in kls),
          f"kl diverged (>= {MAX_KL}): {kls}")

    if reasons:
        print(f"[FAIL] GRPO + vLLM fast inference on {device} did not meet the success criteria:")
        for reason in reasons:
            print(f"    - {reason}")
    else:
        print(f"[PASS] GRPO + vLLM fast inference on {device} ({len(steps)} steps, "
              f"loss {losses[0]:.4f} -> {losses[-1]:.4f}, rollouts non-empty)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FAIL] GRPO + fast_inference raised an exception: {e}")
        raise
