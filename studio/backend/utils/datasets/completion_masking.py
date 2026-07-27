# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Completion-only masking policy shared by the CUDA and MLX training paths.

Decides how train_on_responses_only is applied for a model: chat template
auto-detection first, manual TEMPLATE_TO_RESPONSES_MAPPER markers as the
fallback. gpt-oss included: its quantized checkpoints ship a different
chat template, so only detection from the actual template is reliable.
"""

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


def apply_completion_masking(
    trainer,
    model_name,
    train_fn,
    num_proc = None,
    notify = None,
    detect_fn = None,
):
    """Apply completion-only masking with auto-detection first and the manual
    template table as fallback.

    Args:
        trainer: The platform trainer (SFTTrainer or MLXTrainer).
        model_name: Model repo id used for table lookup and the gpt-oss
            renamed-checkpoint fallback.
        train_fn: The platform train_on_responses_only callable.
        num_proc: Forwarded to train_fn when not None (CUDA path only).
        notify: Optional callback notify(level, message) with level "info" or
            "warning" for user-visible progress and warnings.
        detect_fn: Marker detector (tokenizer/processor) -> (instruction_part,
            response_part). Defaults to unsloth_zoo's get_chat_template_parts,
            which raises loudly when the template cannot be parsed. Test seam.

    Returns:
        (trainer, applied): the possibly wrapped trainer and whether masking
        was applied. When applied is False the trainer is unchanged and
        training runs on full sequences.

    Only marker DETECTION failures trigger the table fallback. Exceptions
    raised while applying the masking (dataset map, tokenization) propagate
    to the caller in both the auto and manual paths, so a real failure stops
    the run instead of silently changing the training objective.
    """
    if notify is None:
        notify = lambda level, message: None
    kwargs = {}
    if num_proc is not None:
        kwargs["num_proc"] = num_proc

    template, instruction_part, response_part = lookup_manual_markers(model_name)

    # gpt-oss goes auto-first: quantized/BF16 checkpoints ship a channel-less
    # template, so the manual markers match nothing (zero tokens trained). Auto
    # derives markers from whichever template ships, and per the harmony format
    # only the final terminator carries stop supervision. Renamed checkpoints
    # miss the exact-name table, so give the fallback the gpt-oss markers.
    if is_gpt_oss_model_name(model_name) and not (instruction_part and response_part):
        markers = TEMPLATE_TO_RESPONSES_MAPPER.get("gpt-oss")
        if markers:
            template = "gpt-oss"
            instruction_part = markers["instruction"]
            response_part = markers["response"]
    processor = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
    # mlx-lm TokenizerWrapper hides underscore attrs, so preset _unsloth_*
    # markers are invisible through it. Unwrap to the real tokenizer (as
    # zoo's MLX resolver does) before the preset check and detection.
    if type(processor).__name__ == "TokenizerWrapper":
        wrapped = getattr(processor, "_tokenizer", None)
        if wrapped is not None:
            processor = wrapped
    inner = getattr(processor, "tokenizer", processor)
    if hasattr(inner, "_unsloth_input_part") and hasattr(inner, "_unsloth_output_part"):
        # Markers preset on the tokenizer; zoo reuses them on a bare call.
        trainer = train_fn(trainer, **kwargs)
        notify(
            "info",
            "Train on responses only configured via tokenizer preset markers",
        )
        return trainer, True
    auto_instruction = auto_response = None
    try:
        if detect_fn is None:
            # Torch-backed import is fine: the MLX train_fn itself requires
            # unsloth_zoo.dataset_utils, so a torch-free host cannot mask either way.
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
