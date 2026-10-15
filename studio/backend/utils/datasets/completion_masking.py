# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Completion-only masking policy shared by the CUDA and MLX training paths.

Decides how train_on_responses_only is applied for a model: explicit dataset
markers when requested, otherwise chat template auto-detection with manual
TEMPLATE_TO_RESPONSES_MAPPER markers as the fallback. gpt-oss included: its
quantized checkpoints ship a different chat template, so only detection from
the actual template is reliable.
"""

from .iterable import is_streaming_dataset
from .model_mappings import (
    MODEL_TO_TEMPLATE_MAPPER,
    TEMPLATE_TO_RESPONSES_MAPPER,
    is_gpt_oss_model_name,
)


def lookup_manual_markers(model_name):
    """Return (template_name, instruction_part, response_part) from the
    manual template table, with None parts when the model or template is
    not mapped."""
    template = MODEL_TO_TEMPLATE_MAPPER.get((model_name or "").lower())
    markers = TEMPLATE_TO_RESPONSES_MAPPER.get(template) if template else None
    if markers:
        return template, markers["instruction"], markers["response"]
    return template, None, None


def _mask_tool_responses(trainer, start_id, end_id, turn_id):
    def mask(batch):
        all_labels = []
        for input_ids, labels in zip(batch["input_ids"], batch["labels"]):
            if start_id in input_ids:
                input_ids, labels = list(input_ids), list(labels)
                starts = [i for i, token in enumerate(input_ids) if token == start_id]
                for start, limit in zip(starts, starts[1:] + [len(input_ids)]):
                    span = input_ids[start + 1 : limit]
                    if end_id in span:
                        stop = span.index(end_id) + 1
                    else:
                        stop = span.index(turn_id) if turn_id in span else len(span)
                    labels[start + 1 : start + 1 + stop] = [-100] * stop
            all_labels.append(labels)
        return {"labels": all_labels}

    for name in ("train_dataset", "eval_dataset"):
        dataset = getattr(trainer, name, None)
        columns = (
            next(iter(dataset), {})
            if is_streaming_dataset(dataset)
            else getattr(dataset, "column_names", None)
        )
        if "labels" in (columns or ()):
            setattr(trainer, name, dataset.map(mask, batched = True))
    return trainer


def apply_completion_masking(
    trainer,
    model_name,
    train_fn,
    num_proc = None,
    notify = None,
    detect_fn = None,
    dataset_template = None,
):
    """Only marker detection failures trigger the table fallback; errors while applying masking
    propagate."""
    if notify is None:
        notify = lambda level, message: None
    kwargs = {}
    if num_proc is not None:
        kwargs["num_proc"] = num_proc

    processor = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
    if type(processor).__name__ == "TokenizerWrapper":
        wrapped = getattr(processor, "_tokenizer", None)
        if wrapped is not None:
            processor = wrapped
    inner = getattr(processor, "tokenizer", processor)

    vocab = inner.get_added_vocab() if hasattr(inner, "get_added_vocab") else {}
    tool_response = (vocab.get("<|tool_response>"), vocab.get("<tool_response|>"))
    if None not in tool_response and type(trainer).__name__ != "MLXTrainer":
        mask_responses = train_fn

        def train_fn(trainer, **kwargs):
            return _mask_tool_responses(
                mask_responses(trainer, **kwargs), *tool_response, vocab.get("<|turn>")
            )

    if dataset_template is not None:
        markers = TEMPLATE_TO_RESPONSES_MAPPER.get(dataset_template)
        if not markers:
            raise ValueError(f"Unknown completion masking template: {dataset_template}")
        has_preset_markers = hasattr(inner, "_unsloth_input_part") and hasattr(
            inner, "_unsloth_output_part"
        )
        if has_preset_markers:
            previous_instruction = inner._unsloth_input_part
            previous_response = inner._unsloth_output_part
            inner._unsloth_input_part = markers["instruction"]
            inner._unsloth_output_part = markers["response"]
            try:
                trainer = train_fn(trainer, **kwargs)
            finally:
                inner._unsloth_input_part = previous_instruction
                inner._unsloth_output_part = previous_response
        else:
            trainer = train_fn(
                trainer,
                instruction_part = markers["instruction"],
                response_part = markers["response"],
                **kwargs,
            )
        notify(
            "info",
            f"Train on responses only configured with dataset template markers ({dataset_template})",
        )
        return trainer, True

    template, instruction_part, response_part = lookup_manual_markers(model_name)

    # gpt-oss goes auto-first: some checkpoints ship a channel-less template.
    if is_gpt_oss_model_name(model_name) and not (instruction_part and response_part):
        markers = TEMPLATE_TO_RESPONSES_MAPPER.get("gpt-oss")
        if markers:
            template = "gpt-oss"
            instruction_part = markers["instruction"]
            response_part = markers["response"]
    if hasattr(inner, "_unsloth_input_part") and hasattr(inner, "_unsloth_output_part"):
        trainer = train_fn(trainer, **kwargs)
        notify(
            "info",
            "Train on responses only configured via tokenizer preset markers",
        )
        return trainer, True
    auto_instruction = auto_response = None
    try:
        if detect_fn is None:
            # MLX train_fn needs unsloth_zoo.dataset_utils anyway, so torch import is fine.
            from unsloth_zoo.dataset_utils import get_chat_template_parts as detect_fn
        auto_instruction, auto_response = detect_fn(processor)
    except Exception as e:
        notify(
            "warning",
            f"Auto-detection of instruction/response markers failed ({e}); "
            f"falling back to the template table",
        )
    if auto_instruction and auto_response:
        trainer = train_fn(
            trainer,
            instruction_part = auto_instruction,
            response_part = auto_response,
            **kwargs,
        )
        notify(
            "info",
            "Train on responses only configured via chat template auto-detection",
        )
        return trainer, True

    if instruction_part and response_part:
        trainer = train_fn(
            trainer,
            instruction_part = instruction_part,
            response_part = response_part,
            **kwargs,
        )
        notify(
            "info",
            f"Train on responses only configured with template table markers ({template})",
        )
        return trainer, True

    notify(
        "warning",
        f"'Train on completions' could not be applied for {model_name}: no "
        f"auto-detected or mapped instruction/response markers. Training "
        f"will run on full sequences (prompts included).",
    )
    return trainer, False
