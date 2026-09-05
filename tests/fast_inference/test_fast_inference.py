# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# ruff: noqa
"""GRPO smoke test for the ``fast_inference=True`` vLLM rollout path.

Exercises the vLLM LoRA activation path (`WorkerLoRAManager`) that regressed on
vLLM >= 0.25.0 (unsloth#7283): the stacked `WeightsMapper` collapsed q/k/v and
gate/up LoRA weights onto one key, crashing adapter activation with
`IndexError`. All seven attention and MLP projections are LoRA targets so both
the fused `qkv_proj` and `gate_up_proj` families are covered.

Kept deliberately tiny so it finishes in well under a minute: a 0.6B model,
`enforce_eager`, no torch.compile, three short training steps, and short
prompts/completions. Seeded, so the asserted metrics are reproducible.

Run directly (`python tests/fast_inference/test_fast_inference.py`) or via
pytest; it skips automatically when no CUDA device is present.
"""

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest
import torch

from tests.utils import header_footer_context

# Spins up a real inference path on the accelerator.
pytestmark = pytest.mark.gpu


MODEL_NAME = "unsloth/Qwen3-0.6B"
MAX_SEQ_LENGTH = 256
LORA_RANK = 8
NUM_GENERATIONS = 2
MAX_PROMPT_LENGTH = 64
MAX_COMPLETION_LENGTH = 16
# >1 so the updated LoRA adapter is re-synced into vLLM on every step, not just loaded once; that repeat sync is the
# path that regressed.
MAX_STEPS = 3
GPU_MEMORY_UTILIZATION = 0.3
COMPILATION_CONFIG = 0
# Pins torch's global RNG (via the Trainer's set_seed), which the colocated vLLM sampler draws from, so the rollout and
# every metric below is reproducible.
SEED = 42

# Loose sanity bounds, not fitted values: they catch divergence and degenerate rollouts while staying valid across
# GPUs, models and vLLM versions.
MAX_CHARS_PER_TOKEN = 20
MAX_GRAD_NORM = 1e3
MAX_KL = 1.0

# All attention + MLP projections, so both fused vLLM LoRA families (qkv_proj and gate_up_proj) are exercised
# the >= 0.25.0 collision hit both.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

SYSTEM_PROMPT = "Respond concisely."
QUESTIONS = ["What is the capital of France?", "What is 2 + 2?"]
PROMPTS = [
    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
    for q in QUESTIONS
]


def length_reward_func(completions, **kwargs) -> list[float]:
    """Reward longer completions. The fractional tie-break keeps rewards distinct
    even if the model samples equal-length completions, so GRPO advantages are
    never all-zero and the step stays meaningful on any vLLM/GPU combination."""
    n = len(completions)
    return [float(len(c[0]["content"])) + i / (n + 1) for i, c in enumerate(completions)]


def _metric(metrics, *names):
    """First present key; TRL spells some metrics differently across versions."""
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "fast_inference needs a CUDA GPU + vLLM")
def test_fast_inference():
    # Import here, not at module load: importing unsloth probes for an accelerator and errors on CPU-only machines, so
    # deferring keeps pytest collection and the skip path import-free. Unsloth must precede TRL.
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    with header_footer_context("Load model (fast_inference=True)"):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = MODEL_NAME,
            max_seq_length = MAX_SEQ_LENGTH,
            load_in_4bit = False,
            fast_inference = True,
            max_lora_rank = LORA_RANK,
            gpu_memory_utilization = GPU_MEMORY_UTILIZATION,
            enforce_eager = True,  # skip CUDA graph capture for fast startup
            compilation_config = COMPILATION_CONFIG,
        )
    assert hasattr(model, "vllm_engine"), "fast_inference=True did not attach a vLLM engine"

    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_RANK,
        target_modules = TARGET_MODULES,
        lora_alpha = LORA_RANK,
        use_gradient_checkpointing = False,
        random_state = SEED,
    )

    dataset = Dataset.from_dict({"prompt": PROMPTS})

    with header_footer_context("GRPO config and trainer"):
        training_args = GRPOConfig(
            learning_rate = 5e-6,
            per_device_train_batch_size = NUM_GENERATIONS,
            gradient_accumulation_steps = 1,
            num_generations = NUM_GENERATIONS,
            max_prompt_length = MAX_PROMPT_LENGTH,
            max_completion_length = MAX_COMPLETION_LENGTH,
            max_steps = MAX_STEPS,
            logging_steps = 1,
            report_to = "none",
            seed = SEED,
        )
        trainer = GRPOTrainer(
            model = model,
            processing_class = tokenizer,
            reward_funcs = [length_reward_func],
            args = training_args,
            train_dataset = dataset,
        )
    # The trainer must actually route rollouts through vLLM, otherwise it would
    # fall back to HF generation and never exercise WorkerLoRAManager.
    assert trainer.args.use_vllm, "GRPO is not configured to use vLLM"
    assert getattr(trainer, "llm", None) is not None, "GRPO did not bind a vLLM engine"

    with header_footer_context("GRPO train (vLLM LoRA rollout)"):
        trainer_stats = trainer.train()

    assert trainer_stats is not None, "trainer.train() returned None"
    assert trainer_stats.global_step == MAX_STEPS, "GRPO ran the wrong number of steps"
    assert math.isfinite(trainer_stats.training_loss), "training loss is not finite"

    # Without these, a rollout that silently produced nothing, or an update that diverged to NaN, would still pass the
    # wiring assertions above.
    steps = [log for log in trainer.state.log_history if "loss" in log]
    assert len(steps) == MAX_STEPS, f"expected {MAX_STEPS} logged steps, got {len(steps)}"

    # Every reward is a completion's character count, so this bounds reward and its spread without hard-coding
    # model-specific values.
    max_reward = MAX_COMPLETION_LENGTH * MAX_CHARS_PER_TOKEN

    for i, step in enumerate(steps, start = 1):
        loss = step["loss"]
        grad_norm = step.get("grad_norm")
        reward = step.get("reward")
        zero_std = step.get("frac_reward_zero_std")
        kl = step.get("kl")
        # Key names differ across the supported TRL range, so accept either.
        length = _metric(step, "completion_length", "completions/mean_length")
        reward_std = _metric(step, "reward_std", "rewards/std")

        assert math.isfinite(loss), f"step {i}: loss not finite ({loss})"
        assert grad_norm is not None, f"step {i}: no grad_norm logged"
        assert math.isfinite(grad_norm), f"step {i}: grad_norm not finite ({grad_norm})"
        # Sign check only: a step can legitimately be near zero (0.004 observed), so any tighter lower bound would be
        # flaky.
        assert 0.0 < grad_norm < MAX_GRAD_NORM, f"step {i}: grad_norm {grad_norm}"
        assert length is not None, f"step {i}: no completion length logged"
        assert 0.0 < length <= MAX_COMPLETION_LENGTH, f"step {i}: empty rollout ({length})"
        assert reward is not None, f"step {i}: no reward logged"
        assert 0.0 < reward <= max_reward, f"step {i}: reward {reward} out of range"
        assert reward_std is not None, f"step {i}: no reward_std logged"
        assert 0.0 < reward_std <= max_reward, f"step {i}: no reward spread ({reward_std})"
        assert zero_std in (None, 0.0), f"step {i}: {zero_std} of groups had no spread"
        assert kl is None or math.isfinite(kl), f"step {i}: kl not finite ({kl})"
        assert kl is None or abs(kl) < MAX_KL, f"step {i}: kl diverged ({kl})"

    print("fast_inference GRPO rollout completed:", trainer_stats)


if __name__ == "__main__":
    if torch.cuda.is_available():
        test_fast_inference()
    else:
        print("Skipping fast_inference test: needs a CUDA GPU + vLLM")
