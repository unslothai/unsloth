# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Completion-only masking policy shared across CUDA and MLX training.

Covers utils.datasets.completion_masking.apply_completion_masking, shared by
the CUDA trainer (core/training/trainer.py) and the MLX worker
(core/training/worker.py):
  - unmapped models use chat template auto-detection (previously masking was
    silently disabled),
  - gpt-oss goes auto-first too (its quantized checkpoints ship a template
    the manual markers cannot match),
  - an auto-detection failure falls back to the template table markers,
  - explicit dataset templates take precedence over tokenizer markers,
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


def test_dataset_template_uses_alpaca_markers_without_detection():
    trainer = _Trainer()
    train_fn = _Recorder()

    def detect(_processor):
        raise AssertionError("dataset template must bypass tokenizer detection")

    result, applied = apply_completion_masking(
        trainer,
        "unsloth/Llama-3.2-1B-Instruct",
        train_fn,
        detect_fn = detect,
        dataset_template = "alpaca",
    )

    expected = TEMPLATE_TO_RESPONSES_MAPPER["alpaca"]
    assert applied is True
    assert result.wrapped_from is trainer
    assert train_fn.calls == [
        {
            "instruction_part": expected["instruction"],
            "response_part": expected["response"],
        }
    ]


def test_dataset_template_temporarily_replaces_tokenizer_markers():
    class _Tok:
        _unsloth_input_part = "<MODEL_INPUT>"
        _unsloth_output_part = "<MODEL_OUTPUT>"

    trainer = _Trainer()
    trainer.processing_class = _Tok()
    expected = TEMPLATE_TO_RESPONSES_MAPPER["alpaca"]
    calls = []

    def train_fn(current_trainer, **kwargs):
        if kwargs and hasattr(current_trainer.processing_class, "_unsloth_input_part"):
            raise ValueError("custom markers conflict with tokenizer markers")
        calls.append(kwargs)
        assert current_trainer.processing_class._unsloth_input_part == expected["instruction"]
        assert current_trainer.processing_class._unsloth_output_part == expected["response"]
        return current_trainer

    result, applied = apply_completion_masking(
        trainer,
        "unsloth/Llama-3.2-1B-Instruct",
        train_fn,
        dataset_template = "alpaca",
    )

    assert applied is True
    assert result is trainer
    assert calls == [{}]
    assert trainer.processing_class._unsloth_input_part == "<MODEL_INPUT>"
    assert trainer.processing_class._unsloth_output_part == "<MODEL_OUTPUT>"


def test_dataset_template_restores_tokenizer_markers_after_failure():
    class _Tok:
        _unsloth_input_part = "<MODEL_INPUT>"
        _unsloth_output_part = "<MODEL_OUTPUT>"

    trainer = _Trainer()
    trainer.processing_class = _Tok()

    def train_fn(_trainer, **_kwargs):
        raise RuntimeError("masking failed")

    with pytest.raises(RuntimeError, match = "masking failed"):
        apply_completion_masking(
            trainer,
            "unsloth/Llama-3.2-1B-Instruct",
            train_fn,
            dataset_template = "alpaca",
        )

    assert trainer.processing_class._unsloth_input_part == "<MODEL_INPUT>"
    assert trainer.processing_class._unsloth_output_part == "<MODEL_OUTPUT>"


def test_dataset_template_forwards_num_proc():
    train_fn = _Recorder()

    apply_completion_masking(
        _Trainer(),
        "unsloth/Llama-3.2-1B-Instruct",
        train_fn,
        num_proc = 4,
        dataset_template = "alpaca",
    )

    assert train_fn.calls[0]["num_proc"] == 4


def test_unknown_dataset_template_fails_loudly():
    train_fn = _Recorder()

    with pytest.raises(ValueError, match = "Unknown completion masking template"):
        apply_completion_masking(
            _Trainer(),
            "unsloth/Llama-3.2-1B-Instruct",
            train_fn,
            dataset_template = "missing",
        )

    assert train_fn.calls == []


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


# --- Truncation-aware no-training-signal diagnosis (issue #11321) -------------


def _zoo_no_signal_error(response_part = "<|turn>model\\n"):
    return ValueError(
        "Unsloth: train_on_responses_only masked every label to -100 in "
        "train_dataset, so there is nothing to train on. The response marker "
        f"{response_part!r} was not found in any sample - check that "
        "instruction_part and response_part match your chat template."
    )


def _truncated_trainer(
    n_rows = 10,
    max_seq_length = 2048,
    drop_args = False,
):
    """A trainer whose train_dataset mimics the post-masking state: rows already
    tokenized and sitting at the sequence cap (prompts longer than max_seq_length)."""
    cap = max_seq_length

    class _Args:
        packing = False
        max_length = None
        max_seq_length = cap

    class _Ds:
        def __init__(self, rows):
            self._data = {"input_ids": rows}

    trainer = _Trainer()
    if not drop_args:
        trainer.args = _Args()
    trainer.train_dataset = _Ds([[42] * cap for _ in range(n_rows)])
    return trainer


def test_no_signal_error_with_truncated_rows_names_max_seq_length():
    # Issue #11321: a long-prompt dataset truncated at max_seq_length masks every
    # label; the zoo's error blames the markers. The rows sit at the cap, so the
    # raised error must point at max_seq_length instead.
    trainer = _truncated_trainer()
    notes = _Notes()

    def train_fn(_trainer, **_kwargs):
        raise _zoo_no_signal_error()

    with pytest.raises(ValueError, match = "raise.*max_seq_length") as exc_info:
        apply_completion_masking(
            trainer, "unsloth/gemma-4-E4B-it", train_fn, notify = notes, detect_fn = _detect_ok
        )

    message = str(exc_info.value)
    assert "max_seq_length=2048" in message
    assert str(_zoo_no_signal_error()) in message  # original text preserved
    assert exc_info.value.__cause__ is not None  # chained for debugging


def test_no_signal_error_with_short_rows_is_preserved_verbatim():
    # Genuinely wrong markers: rows are short, so the original zoo error must
    # pass through unchanged (no bogus truncation advice).
    trainer = _truncated_trainer(max_seq_length = 2048)
    trainer.train_dataset._data = {"input_ids": [[42] * 30 for _ in range(10)]}

    def train_fn(_trainer, **_kwargs):
        raise _zoo_no_signal_error()

    with pytest.raises(ValueError, match = "was not found in any sample") as exc_info:
        apply_completion_masking(trainer, "unsloth/gemma-4-E4B-it", train_fn, detect_fn = _detect_ok)

    assert "max_seq_length" not in str(exc_info.value)
    assert str(exc_info.value) == str(_zoo_no_signal_error())


def test_other_value_errors_propagate_unchanged():
    trainer = _truncated_trainer()

    def train_fn(_trainer, **_kwargs):
        raise ValueError("Unsloth: instruction_part and response_part must be given!")

    with pytest.raises(ValueError, match = "must be given") as exc_info:
        apply_completion_masking(trainer, "unsloth/gemma-4-E4B-it", train_fn, detect_fn = _detect_ok)

    assert str(exc_info.value) == "Unsloth: instruction_part and response_part must be given!"


def test_no_signal_diagnosis_skipped_without_args():
    # No .args (defensive): the zoo error propagates as-is rather than crashing
    # in the diagnosis itself.
    trainer = _truncated_trainer(drop_args = True)

    def train_fn(_trainer, **_kwargs):
        raise _zoo_no_signal_error()

    with pytest.raises(ValueError, match = "was not found in any sample"):
        apply_completion_masking(trainer, "unsloth/gemma-4-E4B-it", train_fn, detect_fn = _detect_ok)


def test_no_signal_diagnosis_skipped_for_streaming_dataset():
    # Iterable train_dataset: rows cannot be sampled, so the zoo error must
    # propagate untouched instead of raising AttributeError in the diagnosis.
    from torch.utils.data import IterableDataset as TorchIterableDataset

    import utils.datasets.iterable as iterable_mod

    class _Stream(TorchIterableDataset):
        def __iter__(self):
            return iter(())

    trainer = _Trainer()
    trainer.args = type("A", (), {"max_length": None, "max_seq_length": 2048})()
    trainer.train_dataset = _Stream()

    def train_fn(_trainer, **_kwargs):
        raise _zoo_no_signal_error()

    assert iterable_mod.is_streaming_dataset(_Stream())  # guard actually fires

    with pytest.raises(ValueError, match = "was not found in any sample"):
        apply_completion_masking(trainer, "unsloth/gemma-4-E4B-it", train_fn, detect_fn = _detect_ok)
