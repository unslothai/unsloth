# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Completion-only masking policy shared by the CUDA and MLX training paths.

Decides how train_on_responses_only is applied for a model: explicit dataset
markers when requested, otherwise chat template auto-detection with manual
TEMPLATE_TO_RESPONSES_MAPPER markers as the fallback. gpt-oss included: its
quantized checkpoints ship a different chat template, so only detection from
the actual template is reliable.
"""

import random
import re

from .iterable import is_streaming_dataset
from .model_mappings import (
    MODEL_TO_TEMPLATE_MAPPER,
    TEMPLATE_TO_RESPONSES_MAPPER,
    is_gpt_oss_model_name,
)

# unsloth_zoo's train_on_responses_only raises this when masking left nothing to
# train on. Its text blames the markers, but max_seq_length truncation produces
# the exact same condition (the marker is cut off before masking runs), so the
# error is re-raised with a truncation-aware diagnosis when the evidence fits.
_NO_TRAINING_SIGNAL_RE = re.compile(
    "train_on_responses_only masked every label to -100.*was not found in any sample",
    re.DOTALL,
)
_TRUNCATION_SAMPLE_SIZE = 100  # same cap as unsloth_zoo's truncation diagnosis


def lookup_manual_markers(model_name):
    """Return (template_name, instruction_part, response_part) from the
    manual template table, with None parts when the model or template is
    not mapped."""
    template = MODEL_TO_TEMPLATE_MAPPER.get((model_name or "").lower())
    markers = TEMPLATE_TO_RESPONSES_MAPPER.get(template) if template else None
    if markers:
        return template, markers["instruction"], markers["response"]
    return template, None, None


def _masking_failed_with_truncation(exc, trainer):
    """Return True when `exc` is the zoo's no-training-signal error and the
    trainer's already-tokenized rows sit at the sequence cap, i.e. truncation
    cut the response marker off before masking could find it."""
    if not _NO_TRAINING_SIGNAL_RE.search(str(exc)):
        return False
    args = getattr(trainer, "args", None)
    max_length = getattr(args, "max_length", None) or getattr(args, "max_seq_length", None)
    if not max_length:
        return False
    dataset = getattr(trainer, "train_dataset", None)
    if dataset is None or is_streaming_dataset(dataset):
        return False
    input_ids = dataset._data["input_ids"]
    if len(input_ids) == 0:
        return False
    sample = random.sample(range(len(input_ids)), min(_TRUNCATION_SAMPLE_SIZE, len(input_ids)))
    at_cap = sum(1 for i in sample if input_ids[i] is not None and len(input_ids[i]) >= max_length)
    return at_cap / len(sample) >= 0.9


def _truncation_error(exc, trainer):
    """Re-raise the zoo's no-training-signal error with the truncation cause
    named, so users are pointed at max_seq_length instead of the (correct)
    markers."""
    args = getattr(trainer, "args", None)
    max_length = getattr(args, "max_length", None) or getattr(args, "max_seq_length", None)
    raise ValueError(
        f"{exc}\n\n"
        f"Unsloth Studio: every sample was truncated at max_seq_length={max_length} "
        "before the response marker could appear, so nothing was trainable. "
        "This dataset has prompts longer than max_seq_length: raise "
        "max_seq_length above your longest sample (GPU memory permitting), or "
        "turn off 'Train on completions' to train on the full truncated "
        "sequences instead."
    ) from exc


def _apply_with_truncation_diagnosis(trainer, train_fn, **kwargs):
    """Call train_fn and, when the zoo's no-training-signal error fires, sample
    rows to tell truncation (marker cut off by max_seq_length) apart from a
    genuine marker/template mismatch, and raise the matching diagnosis."""
    try:
        return train_fn(trainer, **kwargs)
    except ValueError as exc:
        if _masking_failed_with_truncation(exc, trainer):
            _truncation_error(exc, trainer)
        raise


def apply_completion_masking(
    trainer,
    model_name,
    train_fn,
    num_proc = None,
    notify = None,
    detect_fn = None,
    dataset_template = None,
):
    """Apply completion-only masking with an explicit dataset template or
    auto-detection followed by the manual model-template fallback.

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
        dataset_template: Explicit template-table key for already-rendered
            dataset text. Bypasses tokenizer marker detection when provided.

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

    processor = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
    if type(processor).__name__ == "TokenizerWrapper":
        wrapped = getattr(processor, "_tokenizer", None)
        if wrapped is not None:
            processor = wrapped
    inner = getattr(processor, "tokenizer", processor)

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
                trainer = _apply_with_truncation_diagnosis(trainer, train_fn, **kwargs)
            finally:
                inner._unsloth_input_part = previous_instruction
                inner._unsloth_output_part = previous_response
        else:
            trainer = _apply_with_truncation_diagnosis(
                trainer,
                train_fn,
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

    # gpt-oss goes auto-first: quantized/BF16 checkpoints ship a channel-less template, so the manual markers match
    # nothing and zero tokens are trained.
    if is_gpt_oss_model_name(model_name) and not (instruction_part and response_part):
        markers = TEMPLATE_TO_RESPONSES_MAPPER.get("gpt-oss")
        if markers:
            template = "gpt-oss"
            instruction_part = markers["instruction"]
            response_part = markers["response"]
    if hasattr(inner, "_unsloth_input_part") and hasattr(inner, "_unsloth_output_part"):
        # Markers preset on the tokenizer; zoo reuses them on a bare call.
        trainer = _apply_with_truncation_diagnosis(trainer, train_fn, **kwargs)
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
        trainer = _apply_with_truncation_diagnosis(
            trainer,
            train_fn,
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
        trainer = _apply_with_truncation_diagnosis(
            trainer,
            train_fn,
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
