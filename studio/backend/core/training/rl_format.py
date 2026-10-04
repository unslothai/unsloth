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
    for tok in {id(t): t for t in (tokenizer, getattr(tokenizer, "tokenizer", None)) if t is not None}.values():
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


def build_warmup_dataset(tokenizer, max_seq_length: int, rows: int, dataset=None):
    """Formatted rows that fit in half the context, like the notebooks. ``rows`` caps how many get tokenized."""
    from datasets import Dataset, load_dataset

    if dataset is None:
        dataset = load_dataset(WARMUP_DATASET, split = WARMUP_SPLIT)
    limit = max_seq_length // 2
    texts = []
    for row in dataset:
        if len(texts) >= rows:
            break
        answer = str(row.get("expected_answer") or "").strip()
        if not answer or not _is_number(answer):
            continue
        text = tokenizer.apply_chat_template(
            warmup_messages(row.get("problem") or "", row.get("generated_solution") or "", answer),
            tokenize = False,
        )
        if len(tokenizer(text, add_special_tokens = False)["input_ids"]) <= limit:
            texts.append(text)
    if not texts:
        raise ValueError(
            f"No format warm-up examples fit in {limit} tokens; raise the max sequence length."
        )
    return Dataset.from_dict({"text": texts})


def run_format_warmup(
    model,
    tokenizer,
    config_args: dict,
    steps: int,
    should_stop: Callable[[], bool] = lambda: False,
    dataset = None,
) -> Optional[dict]:
    """Runs ``steps`` SFT steps on the warm-up set before GRPO. Returns the trainer's final metrics."""
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    from core.training.rl import _config_kwargs

    batch = int(config_args.get("per_device_train_batch_size") or 1)
    accum = int(config_args.get("gradient_accumulation_steps") or 1)
    max_seq_length = int(config_args.get("max_seq_length") or 2048)
    train_dataset = build_warmup_dataset(
        tokenizer, max_seq_length, rows = max(steps * batch * accum * 2, 64), dataset = dataset
    )
    logger.info(f"Format warm-up: {steps} SFT steps on {len(train_dataset)} examples\n")

    keep = ("seed", "bf16", "fp16", "disable_tqdm", "dataset_num_proc", "per_device_train_batch_size", "gradient_accumulation_steps", "report_to")
    base = {k: v for k, v in config_args.items() if k in keep}
    base["max_length"] = max_seq_length
    args = SFTConfig(
        **_config_kwargs(SFTConfig, base),
        output_dir = str(config_args.get("output_dir") or "outputs") + "/format-warmup",
        dataset_text_field = "text",
        max_steps = steps,
        learning_rate = WARMUP_LEARNING_RATE,
        warmup_steps = min(5, max(steps // 10, 0)),
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        logging_steps = 5,
        save_strategy = "no",
        packing = False,
    )

    class _Stop(TrainerCallback):
        def on_step_end(self, _args, _state, control, **_kw):
            if should_stop():
                control.should_training_stop = True

    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = train_dataset,
        args = args,
        callbacks = [_Stop()],
    )
    result = trainer.train()
    return getattr(result, "metrics", None)
