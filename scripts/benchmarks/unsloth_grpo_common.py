"""Shared helpers for the Qwen3-4B GRPO comparison scripts.

Exports:
    REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END, SYSTEM_PROMPT
    CHAT_TEMPLATE
    build_dataset(tokenizer, max_seq_length=2048)
    build_reward_funcs(tokenizer)
    build_grpo_kwargs(tokenizer, maximum_length, max_seq_length)

Keeps dataset loading, chat template, formatting rewards, and GRPO hparams
identical between the vLLM baseline script and the transformers-CB candidate.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset


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


CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] + eos_token }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{{ '%%%SYSTEM_PROMPT%%%' + eos_token }}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '%%%REASONING_START%%%' }}"
    "{% endif %}"
)


def apply_chat_template_to_tokenizer(tokenizer):
    tmpl = CHAT_TEMPLATE.replace("%%%SYSTEM_PROMPT%%%", SYSTEM_PROMPT)
    tmpl = tmpl.replace("%%%REASONING_START%%%", REASONING_START)
    tokenizer.chat_template = tmpl
    return tokenizer


def build_dataset(tokenizer, *, max_seq_length: int = 2048):
    """Build the DAPO-Math-17k GRPO dataset with the prompt formatting from the
    notebook. Returns `(dataset, maximum_prompt_length)`.
    """
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split = "train")

    def _map_row(x):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["prompt"]},
            ],
            "answer": x["solution"],
        }

    ds = ds.map(_map_row)

    # Tokenize for length measurement (batched for speed).
    def _tokenize(batch):
        return {
            "tokens": tokenizer.apply_chat_template(
                batch["prompt"],
                add_generation_prompt = True,
                tokenize = True,
            )
        }

    tokenized = ds.map(_tokenize, batched = True)
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
    lengths = np.array(tokenized["L"])
    maximum_length = int(np.quantile(lengths, 0.9))
    ds = ds.select(np.where(lengths <= maximum_length)[0])
    return ds, maximum_length


def build_reward_funcs(tokenizer):
    """Return the 4 reward functions used in the notebook, wired to `tokenizer`."""
    solution_end_regex = (
        r"</SOLUTION>[\s]{0,}" + "(?:" + re.escape(tokenizer.eos_token) + ")?"
    )
    match_format = re.compile(
        rf"{REASONING_END}.*?"
        rf"{SOLUTION_START}(.+?){solution_end_regex}"
        rf"[\s]{{0,}}$",
        flags = re.MULTILINE | re.DOTALL,
    )
    match_numbers = re.compile(
        SOLUTION_START + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
        flags = re.MULTILINE | re.DOTALL,
    )

    def match_format_exactly(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0.0
            response = completion[0]["content"]
            if match_format.search(response) is not None:
                score += 3.0
            scores.append(score)
        return scores

    def match_format_approximately(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0.0
            response = completion[0]["content"]
            score += 0.5 if response.count(REASONING_END) == 1 else -1.0
            score += 0.5 if response.count(SOLUTION_START) == 1 else -1.0
            score += 0.5 if response.count(SOLUTION_END) == 1 else -1.0
            scores.append(score)
        return scores

    def check_answer(prompts, completions, answer, **kwargs):
        responses = [c[0]["content"] for c in completions]
        extracted = [
            guess.group(1) if (guess := match_format.search(r)) is not None else None
            for r in responses
        ]
        scores = []
        for guess, true_answer in zip(extracted, answer):
            score = 0.0
            if guess is None:
                scores.append(-2.0)
                continue
            if guess == true_answer:
                score += 5.0
            elif guess.strip() == true_answer.strip():
                score += 3.5
            else:
                try:
                    ratio = float(guess) / float(true_answer)
                    if 0.9 <= ratio <= 1.1:
                        score += 2.0
                    elif 0.8 <= ratio <= 1.2:
                        score += 1.5
                    else:
                        score -= 2.5
                except Exception:
                    score -= 4.5
            scores.append(score)
        return scores

    _printed_state = {"n": 0, "every": 5}

    def check_numbers(prompts, completions, answer, **kwargs):
        question = prompts[0][-1]["content"]
        responses = [c[0]["content"] for c in completions]
        extracted = [
            guess.group(1) if (guess := match_numbers.search(r)) is not None else None
            for r in responses
        ]
        if _printed_state["n"] % _printed_state["every"] == 0:
            print(
                "*" * 20 + f"Question:\n{question}",
                f"\nAnswer:\n{answer[0]}",
                f"\nResponse:\n{responses[0]}",
                f"\nExtracted:\n{extracted[0]}",
            )
        _printed_state["n"] += 1

        scores = []
        for guess, true_answer in zip(extracted, answer):
            if guess is None:
                scores.append(-2.5)
                continue
            try:
                t = float(true_answer.strip())
                g = float(guess.strip().replace(",", ""))
                scores.append(3.5 if g == t else -1.5)
            except Exception:
                scores.append(0.0)
        return scores

    return [
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ]


def build_grpo_kwargs(
    tokenizer,
    maximum_length: int,
    *,
    max_seq_length: int = 2048,
    max_steps: int = 100,
    num_generations: int = 4,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    output_dir: str = "outputs",
):
    """Return the shared dict of GRPOConfig kwargs used by both backends.

    Caller adds backend-specific keys (use_vllm / use_transformers_paged / etc).
    """
    max_prompt_length = maximum_length + 1
    max_completion_length = max_seq_length - max_prompt_length

    return dict(
        temperature = 1.0,
        top_p = 1.0,
        top_k = -1,
        min_p = 0.1,
        learning_rate = 5e-6,
        weight_decay = 0.001,
        warmup_ratio = 0.1,
        lr_scheduler_type = "linear",
        optim = "adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        num_generations = num_generations,
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
        max_steps = max_steps,
        save_steps = max_steps,
        report_to = "none",
        output_dir = output_dir,
        seed = 3407,
    )
