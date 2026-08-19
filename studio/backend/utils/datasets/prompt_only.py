# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Prompt-only dataset preparation for GRPO.

GRPO generates the completion itself, so the trainer wants a ``prompt`` column
and no completion. Everything else in Studio assumes an input/output pair, so
this module recognises the prompt-only shape and reduces the common
instruction/answer datasets to it, keeping the reference column around for the
reward functions.
"""

# `Dataset` is annotation-only, matching raw_text: a module-scope `datasets`
# import drags torch in via datasets.formatting.torch_formatter.
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

from .raw_text import resolve_column_names

if TYPE_CHECKING:
    from datasets import Dataset

PROMPT_COLUMN_ALIASES = (
    "prompt",
    "question",
    "instruction",
    "query",
    "problem",
    "input",
)
ANSWER_COLUMN_ALIASES = (
    "answer",
    "solution",
    "output",
    "response",
    "label",
)


@dataclass(frozen = True)
class PromptOnlyNotice:
    message: str
    level: Literal["info", "warning"]
    update_status: bool = False


@dataclass(frozen = True)
class PromptOnlyPreparationResult:
    dataset: Dataset
    prompt_column: str
    answer_column: str | None
    notices: list[PromptOnlyNotice]


def detect_prompt_only_format(dataset, custom_format_mapping: dict | None = None) -> dict:
    """Detect the prompt (and optional reference answer) column.

    Returns ``{"prompt_column": str | None, "answer_column": str | None,
    "columns": list[str]}``. A user mapping wins over the alias tables: the
    format-mapping UI sends {column: role} and 'user'/'prompt' names the prompt.
    """
    columns = resolve_column_names(dataset)
    column_set = set(columns)

    prompt_column = None
    answer_column = None
    if custom_format_mapping:
        for column, role in custom_format_mapping.items():
            if column.startswith("__") or column not in column_set:
                continue
            if prompt_column is None and role in ("user", "prompt", "instruction"):
                prompt_column = column
            elif answer_column is None and role in ("assistant", "answer", "output"):
                answer_column = column

    if prompt_column is None:
        for alias in PROMPT_COLUMN_ALIASES:
            if alias in column_set:
                prompt_column = alias
                break
    if answer_column is None:
        for alias in ANSWER_COLUMN_ALIASES:
            if alias in column_set and alias != prompt_column:
                answer_column = alias
                break

    return {
        "prompt_column": prompt_column,
        "answer_column": answer_column,
        "columns": columns,
    }


def prepare_prompt_only_dataset(
    dataset: Dataset,
    *,
    split_name: str | None = None,
    custom_format_mapping: dict | None = None,
    system_prompt: str | None = None,
    conversational: bool = True,
) -> PromptOnlyPreparationResult:
    """Reduce *dataset* to a GRPO-shaped ``prompt`` (+ ``answer``) dataset."""
    notices: list[PromptOnlyNotice] = []
    scope = f"the {split_name} split" if split_name else "this dataset"

    detected = detect_prompt_only_format(dataset, custom_format_mapping)
    prompt_column = detected["prompt_column"]
    answer_column = detected["answer_column"]
    if prompt_column is None:
        raise ValueError(
            "GRPO training requires a prompt column but none was found in "
            f"{scope} (columns: {detected['columns']}). Map one of your columns "
            "to the user/prompt role, or rename it to 'prompt'."
        )

    if prompt_column != "prompt":
        notices.append(
            PromptOnlyNotice(
                message = f"GRPO: using column '{prompt_column}' as the prompt for {scope}",
                level = "info",
            )
        )
    if answer_column is None:
        notices.append(
            PromptOnlyNotice(
                message = (
                    "GRPO: no reference answer column found in "
                    f"{scope}; reward functions that compare against a reference "
                    "answer will score every rollout 0."
                ),
                level = "warning",
                update_status = True,
            )
        )
    elif answer_column != "answer":
        notices.append(
            PromptOnlyNotice(
                message = (
                    f"GRPO: using column '{answer_column}' as the reference answer for {scope}"
                ),
                level = "info",
            )
        )

    def _to_prompt_only(example):
        prompt_text = example[prompt_column]
        prompt_text = "" if prompt_text is None else str(prompt_text)
        if conversational:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt_text})
            prompt_value = messages
        else:
            prompt_value = f"{system_prompt}\n\n{prompt_text}" if system_prompt else prompt_text
        row = {"prompt": prompt_value}
        if answer_column is not None:
            answer_value = example[answer_column]
            row["answer"] = "" if answer_value is None else str(answer_value)
        return row

    remove_columns = [c for c in detected["columns"] if c not in ("prompt", "answer")]
    prepared = dataset.map(_to_prompt_only, remove_columns = remove_columns)

    return PromptOnlyPreparationResult(
        dataset = prepared,
        prompt_column = prompt_column,
        answer_column = answer_column,
        notices = notices,
    )
