# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Completion-only masking policy shared by the CUDA and MLX training paths.

Decides how train_on_responses_only is applied for a model: chat template
auto-detection first, manual TEMPLATE_TO_RESPONSES_MAPPER markers as the
fallback, with gpt-oss pinned to its manual markers.
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


def apply_completion_masking(trainer, model_name, train_fn, num_proc = None, notify = None):
    """Apply completion-only masking with auto-detection first and the manual
    template table as fallback.

    Args:
        trainer: The platform trainer (SFTTrainer or MLXTrainer).
        model_name: Model repo id used for table lookup and gpt-oss detection.
        train_fn: The platform train_on_responses_only callable. Called with
            no marker arguments to auto-detect from the chat template
            (unsloth_zoo.get_chat_template_parts), which raises loudly when
            the template cannot be parsed.
        num_proc: Forwarded to train_fn when not None (CUDA path only).
        notify: Optional callback notify(level, message) with level "info" or
            "warning" for user-visible progress and warnings.

    Returns:
        (trainer, applied): the possibly wrapped trainer and whether masking
        was applied. When applied is False the trainer is unchanged and
        training runs on full sequences.

    Exceptions from the manual fallback propagate to the caller; exceptions
    from the auto attempt are caught and trigger the fallback.
    """
    if notify is None:
        notify = lambda level, message: None
    kwargs = {}
    if num_proc is not None:
        kwargs["num_proc"] = num_proc

    template, instruction_part, response_part = lookup_manual_markers(model_name)

    # gpt-oss keeps its manual markers: with them, non-final assistant <|end|>
    # tokens stay trained, whereas auto-detection would mask them. Preserve
    # the current trained behavior.
    if not is_gpt_oss_model_name(model_name):
        try:
            trainer = train_fn(trainer, **kwargs)
            notify(
                "info",
                "Train on responses only configured via chat template auto-detection",
            )
            return trainer, True
        except Exception as e:
            notify(
                "warning",
                f"Auto-detection of instruction/response markers failed ({e}); "
                f"falling back to the template table",
            )

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
