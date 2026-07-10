# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Completion-only masking policy: auto-detect first, manual table fallback.

Covers utils.datasets.completion_masking.apply_completion_masking, shared by
the CUDA trainer (core/training/trainer.py) and the MLX worker
(core/training/worker.py):
  - unmapped models use chat template auto-detection (previously masking was
    silently disabled),
  - gpt-oss goes auto-first too (its quantized checkpoints ship a template
    the manual markers cannot match),
  - an auto-detection failure falls back to the template table markers,
  - a table miss after an auto failure warns and leaves the trainer unchanged.
"""

from __future__ import annotations

import pytest

from utils.datasets.completion_masking import apply_completion_masking, lookup_manual_markers
from utils.datasets.model_mappings import TEMPLATE_TO_RESPONSES_MAPPER


class _Trainer:
    """Sentinel trainer; train_fn wraps it in a new object when applied."""


class _Recorder:
    """Fake train_on_responses_only that records calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, trainer, **kwargs):
        self.calls.append(kwargs)
        wrapped = _Trainer()
        wrapped.wrapped_from = trainer
        return wrapped


def _detect_ok(processor):
    return "<INS>", "<RES>"


def _detect_fail(processor):
    raise ValueError(
        "Unsloth: Could not reliably auto-detect response_part - "
        "pass instruction_part and response_part."
    )


_AUTO = {"instruction_part": "<INS>", "response_part": "<RES>"}


class _Notes:
    def __init__(self):
        self.messages = []

    def __call__(self, level, message):
        self.messages.append((level, message))

    def warnings(self):
        return [m for level, m in self.messages if level == "warning"]


def test_unmapped_model_uses_auto_detection():
    # Unmapped model: the auto path applies masking (was silently disabled).
    trainer = _Trainer()
    train_fn = _Recorder()
    notes = _Notes()

    result, applied = apply_completion_masking(
        trainer, "LiquidAI/LFM2-8B-A1B", train_fn, notify = notes, detect_fn = _detect_ok
    )

    assert applied is True
    assert result.wrapped_from is trainer
    assert train_fn.calls == [dict(_AUTO)]  # applied with the detected markers
    assert notes.warnings() == []


def test_mapped_model_prefers_auto_detection():
    trainer = _Trainer()
    train_fn = _Recorder()

    _, applied = apply_completion_masking(
        trainer, "unsloth/Qwen3-0.6B", train_fn, detect_fn = _detect_ok
    )

    assert applied is True
    assert train_fn.calls == [dict(_AUTO)]


def test_gpt_oss_uses_auto_detection_first():
    # The quantized gpt-oss checkpoints ship a template without the
    # <|channel|>final header, where the manual markers match nothing; auto
    # derives markers from the template the checkpoint actually ships.
    trainer = _Trainer()
    train_fn = _Recorder()

    _, applied = apply_completion_masking(
        trainer, "unsloth/gpt-oss-20b", train_fn, detect_fn = _detect_ok
    )

    assert applied is True
    assert train_fn.calls == [dict(_AUTO)]


def test_gpt_oss_detection_failure_falls_back_to_manual_markers():
    trainer = _Trainer()
    train_fn = _Recorder()

    _, applied = apply_completion_masking(
        trainer, "unsloth/gpt-oss-20b", train_fn, detect_fn = _detect_fail
    )

    assert applied is True
    expected = TEMPLATE_TO_RESPONSES_MAPPER["gpt-oss"]
    assert train_fn.calls == [
        {
            "instruction_part": expected["instruction"],
            "response_part": expected["response"],
        }
    ]


def test_auto_failure_falls_back_to_template_table():
    trainer = _Trainer()
    train_fn = _Recorder()
    notes = _Notes()

    result, applied = apply_completion_masking(
        trainer, "unsloth/Qwen3-0.6B", train_fn, notify = notes, detect_fn = _detect_fail
    )

    assert applied is True
    assert result.wrapped_from is trainer
    expected = TEMPLATE_TO_RESPONSES_MAPPER["qwen3"]
    assert train_fn.calls == [
        {
            "instruction_part": expected["instruction"],
            "response_part": expected["response"],
        },
    ]
    assert any("falling back to the template table" in m for m in notes.warnings())


def test_application_failure_propagates_not_fallback():
    # Detection succeeds; a failure while APPLYING the masking must propagate,
    # never silently fall back to full-sequence training.
    def train_fn(trainer, **kwargs):
        raise RuntimeError("dataset map worker crashed")

    with pytest.raises(RuntimeError, match = "dataset map worker crashed"):
        apply_completion_masking(_Trainer(), "LiquidAI/LFM2-8B-A1B", train_fn, detect_fn = _detect_ok)


def test_preset_tokenizer_markers_used_directly():
    # Preset unsloth marker attrs skip detection; zoo reuses them on a bare call.
    class _Tok:
        _unsloth_input_part = "<I>"
        _unsloth_output_part = "<O>"

    trainer = _Trainer()
    trainer.processing_class = _Tok()
    train_fn = _Recorder()

    _, applied = apply_completion_masking(
        trainer, "LiquidAI/LFM2-8B-A1B", train_fn, detect_fn = _detect_fail
    )
    assert applied is True
    assert train_fn.calls == [{}]  # bare call, stored parts


def test_table_miss_warns_and_disables_without_crashing():
    trainer = _Trainer()
    train_fn = _Recorder()
    notes = _Notes()

    result, applied = apply_completion_masking(
        trainer, "some-org/not-in-any-mapper", train_fn, notify = notes, detect_fn = _detect_fail
    )

    assert applied is False
    assert result is trainer  # unchanged: full sequence training
    assert train_fn.calls == []  # detection failed; nothing applied
    assert any("could not be applied" in m for m in notes.warnings())
    assert any("full sequences" in m for m in notes.warnings())


def test_num_proc_forwarded_only_when_given():
    # CUDA path passes num_proc; the MLX path omits it.
    train_fn = _Recorder()
    apply_completion_masking(
        _Trainer(), "unsloth/Qwen3-0.6B", train_fn, num_proc = 4, detect_fn = _detect_ok
    )
    assert train_fn.calls == [dict(_AUTO, num_proc = 4)]

    train_fn = _Recorder()
    apply_completion_masking(
        _Trainer(), "unsloth/Qwen3-0.6B", train_fn, num_proc = 4, detect_fn = _detect_fail
    )
    assert train_fn.calls[0]["num_proc"] == 4

    train_fn = _Recorder()
    apply_completion_masking(_Trainer(), "unsloth/Qwen3-0.6B", train_fn, detect_fn = _detect_ok)
    assert train_fn.calls == [dict(_AUTO)]


def test_manual_fallback_failure_propagates_to_caller():
    # Errors while applying the manual fallback must propagate to the caller.
    def train_fn(trainer, **kwargs):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match = "boom"):
        apply_completion_masking(_Trainer(), "unsloth/gpt-oss-20b", train_fn)


def test_notify_is_optional():
    train_fn = _Recorder()
    _, applied = apply_completion_masking(
        _Trainer(), "some-org/not-in-any-mapper", train_fn, detect_fn = _detect_fail
    )
    assert applied is False


def test_lookup_manual_markers():
    template, instruction, response = lookup_manual_markers("unsloth/Qwen3-0.6B")
    assert template == "qwen3"
    assert instruction == TEMPLATE_TO_RESPONSES_MAPPER["qwen3"]["instruction"]
    assert response == TEMPLATE_TO_RESPONSES_MAPPER["qwen3"]["response"]

    template, instruction, response = lookup_manual_markers("some-org/unknown")
    assert (template, instruction, response) == (None, None, None)

    template, instruction, response = lookup_manual_markers(None)
    assert (template, instruction, response) == (None, None, None)


def test_renamed_gpt_oss_gets_template_markers():
    # Name-detected as gpt-oss but not in the exact-name table: must use the
    # gpt-oss markers, not fall through to full-sequence training.
    trainer = _Trainer()
    train_fn = _Recorder()

    _, applied = apply_completion_masking(
        trainer, "some-org/gpt-oss-20b-sft", train_fn, detect_fn = _detect_fail
    )
    assert applied is True
    expected = TEMPLATE_TO_RESPONSES_MAPPER["gpt-oss"]
    assert train_fn.calls == [
        {
            "instruction_part": expected["instruction"],
            "response_part": expected["response"],
        }
    ]


class _FakeTokenizerWrapper:
    """mlx-lm TokenizerWrapper semantics: plain reads delegate to the wrapped
    tokenizer, underscore attrs do not (so preset markers are hidden)."""

    def __init__(self, tokenizer):
        object.__setattr__(self, "_tokenizer", tokenizer)

    def __getattr__(self, attr):
        if attr.startswith("_"):
            return object.__getattribute__(self, attr)
        return getattr(object.__getattribute__(self, "_tokenizer"), attr)


_FakeTokenizerWrapper.__name__ = "TokenizerWrapper"


def test_mlx_tokenizer_wrapper_unwrapped_for_preset_markers():
    # Markers live on the inner HF tokenizer that the wrapper hides; the helper
    # must unwrap so the preset bare-call path still fires on MLX.
    class _Tok:
        _unsloth_input_part = "<I>"
        _unsloth_output_part = "<O>"

    trainer = _Trainer()
    trainer.tokenizer = _FakeTokenizerWrapper(_Tok())
    train_fn = _Recorder()

    _, applied = apply_completion_masking(
        trainer, "LiquidAI/LFM2-8B-A1B", train_fn, detect_fn = _detect_fail
    )
    assert applied is True
    assert train_fn.calls == [{}]  # bare call, stored parts


def test_mlx_tokenizer_wrapper_unwrapped_for_detection():
    # Detection must see the real tokenizer, not the wrapper, so it does not
    # depend on the loader's __call__ patch.
    class _Tok:
        pass

    inner = _Tok()
    trainer = _Trainer()
    trainer.tokenizer = _FakeTokenizerWrapper(inner)
    train_fn = _Recorder()
    seen = []

    def detect(processor):
        seen.append(processor)
        return "<INS>", "<RES>"

    _, applied = apply_completion_masking(
        trainer, "LiquidAI/LFM2-8B-A1B", train_fn, detect_fn = detect
    )
    assert applied is True
    assert seen == [inner]
