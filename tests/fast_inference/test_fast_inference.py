# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa
"""GRPO smoke test for the ``fast_inference=True`` vLLM rollout path.

Exercises the vLLM LoRA activation path (`WorkerLoRAManager`) that regressed on
vLLM >= 0.25.0 (unsloth#7283): the stacked `WeightsMapper` collapsed q/k/v and
gate/up LoRA weights onto one key, crashing adapter activation with
`IndexError`. All seven attention and MLP projections are LoRA targets so both
the fused `qkv_proj` and `gate_up_proj` families are covered.

Kept deliberately tiny so it finishes in well under a minute: a 0.5B model,
`enforce_eager` (no CUDA graph capture), a single training step, and short
prompts/completions.

Run directly (`python tests/fast_inference/test_fast_inference.py`) or via
pytest; it skips automatically when no CUDA device is present.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest
import torch

from tests.utils import header_footer_context


MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LENGTH = 256
LORA_RANK = 8
NUM_GENERATIONS = 2
MAX_PROMPT_LENGTH = 64
MAX_COMPLETION_LENGTH = 16
MAX_STEPS = 1
GPU_MEMORY_UTILIZATION = 0.3

# All attention + MLP projections, so both fused vLLM LoRA families (qkv_proj and
# gate_up_proj) are exercised -- the >= 0.25.0 collision hit both.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

SYSTEM_PROMPT = "Respond concisely."
QUESTIONS = ["What is the capital of France?", "What is 2 + 2?"]
PROMPTS = [
    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
    for q in QUESTIONS
]


def length_reward_func(completions, **kwargs) -> list[float]:
    """Reward longer completions, so rewards vary and GRPO advantages are non-zero."""
    return [float(len(c[0]["content"])) for c in completions]


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "fast_inference needs a CUDA GPU + vLLM")
def test_fast_inference():
    # Import here, not at module load: importing unsloth probes for an
    # accelerator and errors on CPU-only machines, so deferring keeps pytest
    # collection and the skip path import-free. Unsloth must precede TRL.
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
        )
    assert hasattr(model, "vllm_engine"), "fast_inference=True did not attach a vLLM engine"

    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_RANK,
        target_modules = TARGET_MODULES,
        lora_alpha = LORA_RANK,
        use_gradient_checkpointing = False,
        random_state = 42,
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
    print("fast_inference GRPO rollout completed:", trainer_stats)
    return trainer_stats


if __name__ == "__main__":
    if torch.cuda.is_available():
        test_fast_inference()
    else:
        print("Skipping fast_inference test: needs a CUDA GPU + vLLM")
