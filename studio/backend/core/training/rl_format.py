# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Format warm-up for GRPO, as in the Qwen3 (4B), Llama FP8 and LFM2.5 GRPO notebooks: a reasoning chat
template, then a short SFT pass on formatted examples so a base model already writes the tags before
GRPO starts rewarding them."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

SYSTEM_PROMPT = (
    "You are given a problem.\n"
    "Think about the problem and provide your working out.\n"
    f"Place it between {REASONING_START} and {REASONING_END}.\n"
    f"Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"
)

WARMUP_DATASET = "unsloth/OpenMathReasoning-mini"
WARMUP_SPLIT = "cot"
WARMUP_LEARNING_RATE = 2e-4
# Its shortest examples are ~800 tokens and only those under half the context are kept.
WARMUP_MIN_SEQ_LENGTH = 2048
# The notebooks pre fine-tune for about this many steps before GRPO.
WARMUP_DEFAULT_STEPS = 100


def _jinja_string(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n") + "'"


CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] + eos_token }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{{ " + _jinja_string(SYSTEM_PROMPT) + " + eos_token }}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ " + _jinja_string(REASONING_START) + " }}{% endif %}"
)


def apply_reasoning_template(tokenizer) -> None:
    """Sets the notebook template on the tokenizer (and the inner tokenizer of a processor)."""
    for tok in {
        id(t): t for t in (tokenizer, getattr(tokenizer, "tokenizer", None)) if t is not None
    }.values():
        tok.chat_template = CHAT_TEMPLATE


def _is_number(value: Any) -> bool:
    try:
        float(str(value).replace(",", ""))
    except ValueError:
        return False
    return True


def warmup_messages(problem: str, solution: str, answer: str) -> list[dict]:
    thinking = solution.replace("<think>", "").replace("</think>", "").strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {
            "role": "assistant",
            "content": f"{REASONING_START}{thinking}{REASONING_END}{SOLUTION_START}{answer}{SOLUTION_END}",
        },
    ]


def load_warmup_source():
    from datasets import load_dataset
    return load_dataset(WARMUP_DATASET, split = WARMUP_SPLIT)


def warmup_offline_warning() -> str:
    return (
        f"Format warm-up skipped: its examples ({WARMUP_DATASET}) are not in the local cache "
        "and Hugging Face is offline. GRPO runs without it."
    )


def build_warmup_dataset(
    tokenizer,
    max_seq_length: int,
    rows: int,
    dataset = None,
    max_completion_length: Optional[int] = None,
):
    """Formatted rows that fit in half the context, like the notebooks, and whose answer fits the
    GRPO completion cap, so the warm-up never teaches answers GRPO will cut off. ``rows`` caps how
    many get tokenized."""
    from datasets import Dataset

    if dataset is None:
        dataset = load_warmup_source()
    limit = max_seq_length // 2
    texts = []
    for row in dataset:
        if len(texts) >= rows:
            break
        answer = str(row.get("expected_answer") or "").strip()
        if not answer or not _is_number(answer):
            continue
        messages = warmup_messages(
            row.get("problem") or "", row.get("generated_solution") or "", answer
        )
        if max_completion_length and _token_count(tokenizer, messages[-1]["content"]) > (
            max_completion_length
        ):
            continue
        text = tokenizer.apply_chat_template(messages, tokenize = False)
        if _token_count(tokenizer, text) <= limit:
            texts.append(text)
    if not texts:
        cap = f" with answers under {max_completion_length} tokens" if max_completion_length else ""
        raise ValueError(
            f"No format warm-up examples fit in {limit} tokens{cap}; raise the max sequence "
            "length or the max completion length."
        )
    return Dataset.from_dict({"text": texts})


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens = False)["input_ids"])


def warmup_steps_to_run(rl_settings: Optional[dict], objective: str, resuming: bool) -> int:
    """SFT warm-up steps before GRPO; none on a resume, whose checkpoint already has them."""
    if objective != "grpo" or resuming:
        return 0
    return int((rl_settings or {}).get("format_warmup_steps") or 0)


def warmup_callback(
    steps: int,
    should_stop: Callable[[], bool],
    on_step: Optional[Callable[[int, int], None]] = None,
):
    """Stops the warm-up on cancel and reports each finished step as (step, steps)."""
    from transformers import TrainerCallback

    class _WarmupProgress(TrainerCallback):
        def on_step_end(self, _args, state, control, **_kw):
            if on_step is not None:
                on_step(int(state.global_step), steps)
            if should_stop():
                control.should_training_stop = True

    return _WarmupProgress()


def _warmup_optimizer(device_backend: Optional[str] = None) -> str:
    """The notebooks' 8-bit AdamW, normalised for the device as the run's own optimizer is (XPU has
    no bitsandbytes optimizers)."""
    from core.training.training import normalize_training_optimizer_for_device

    if device_backend is None:
        from utils.hardware import get_device
        device_backend = get_device().value
    return normalize_training_optimizer_for_device("adamw_8bit", device_backend = device_backend)


def run_format_warmup(
    model,
    tokenizer,
    config_args: dict,
    steps: int,
    should_stop: Callable[[], bool] = lambda: False,
    dataset = None,
    on_step: Optional[Callable[[int, int], None]] = None,
    max_completion_length: Optional[int] = None,
) -> Optional[dict]:
    """Runs ``steps`` SFT steps on the warm-up set before GRPO. Returns the trainer's final metrics."""
    from trl import SFTConfig, SFTTrainer

    from core.training.rl import _config_kwargs

    batch = int(config_args.get("per_device_train_batch_size") or 1)
    accum = int(config_args.get("gradient_accumulation_steps") or 1)
    max_seq_length = int(config_args.get("max_seq_length") or 2048)
    train_dataset = build_warmup_dataset(
        tokenizer,
        max_seq_length,
        rows = max(steps * batch * accum * 2, 64),
        dataset = dataset,
        max_completion_length = max_completion_length,
    )
    logger.info(f"Format warm-up: {steps} SFT steps on {len(train_dataset)} examples\n")

    keep = (
        "seed",
        "bf16",
        "fp16",
        "disable_tqdm",
        "dataset_num_proc",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "report_to",
    )
    base = {k: v for k, v in config_args.items() if k in keep}
    base["max_length"] = max_seq_length
    args = SFTConfig(
        **_config_kwargs(SFTConfig, base),
        output_dir = str(config_args.get("output_dir") or "outputs") + "/format-warmup",
        dataset_text_field = "text",
        max_steps = steps,
        learning_rate = WARMUP_LEARNING_RATE,
        warmup_steps = min(5, max(steps // 10, 0)),
        optim = _warmup_optimizer(),
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        logging_steps = 5,
        save_strategy = "no",
        packing = False,
    )

    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = train_dataset,
        args = args,
        callbacks = [warmup_callback(steps, should_stop, on_step)],
    )
    result = trainer.train()
    return getattr(result, "metrics", None)
