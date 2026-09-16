# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for preserving codec-audio evaluation datasets."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import importlib  # noqa: E402
import json  # noqa: E402
import types  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402


_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    """Stub dependencies missing from the backend test environment."""
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - stub unusable imports
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

from core.training import trainer as tmod  # noqa: E402

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


CODEC_TYPES = ("csm", "snac", "bicodec", "dac")


@pytest.fixture
def audio_trainer(monkeypatch):
    # Avoid MLX substitution on Apple silicon.
    monkeypatch.setattr(tmod, "should_use_mlx_training_backend", lambda *a, **k: False)
    t = tmod.UnslothTrainer()
    t.model_name = "unsloth/csm-1b"
    return t


def _rows(path: Path, text: str) -> str:
    path.write_text(json.dumps({"audio": text, "text": text}) + "\n", encoding = "utf-8")
    return str(path)


@pytest.mark.parametrize("audio_type", CODEC_TYPES)
def test_codec_branches_return_the_uploaded_eval_split(
    audio_trainer, tmp_path, monkeypatch, audio_type
):
    audio_trainer._audio_type = audio_type
    monkeypatch.setattr(tmod, "ensure_audio_decoding", lambda: True)
    seen = []

    def fake_preprocess(dataset, custom_format_mapping = None):
        seen.append(len(dataset))
        return dataset

    monkeypatch.setattr(
        audio_trainer, f"_preprocess_{audio_type}_dataset", fake_preprocess, raising = True
    )

    result = audio_trainer.load_and_format_dataset(
        None,
        local_datasets = [_rows(tmp_path / "train.jsonl", "train")],
        local_eval_datasets = [_rows(tmp_path / "eval.jsonl", "eval")],
        eval_steps = 0.1,
    )

    assert result is not None
    _train, eval_dataset = result
    assert eval_dataset is not None, "the uploaded eval split was dropped"
    assert len(seen) == 2, "the eval split must go through the same preprocessing as the train one"


@pytest.mark.parametrize("audio_type", CODEC_TYPES)
def test_no_eval_upload_still_returns_no_eval_split(
    audio_trainer, tmp_path, monkeypatch, audio_type
):
    audio_trainer._audio_type = audio_type
    monkeypatch.setattr(tmod, "ensure_audio_decoding", lambda: True)
    monkeypatch.setattr(
        audio_trainer, f"_preprocess_{audio_type}_dataset", lambda ds, m = None: ds, raising = True
    )

    _train, eval_dataset = audio_trainer.load_and_format_dataset(
        None,
        local_datasets = [_rows(tmp_path / "train.jsonl", "train")],
        eval_steps = 0.1,
    )

    assert eval_dataset is None


def test_an_unpreparable_eval_split_warns_instead_of_failing_the_run(audio_trainer):
    def explode(dataset, custom_format_mapping = None):
        raise ValueError("no audio column found in dataset")

    assert audio_trainer._preprocess_audio_eval_split(object(), explode, None) is None
    assert any("no evaluation" in w for w in audio_trainer.training_progress.warnings)
    assert any("no audio column found" in w for w in audio_trainer.training_progress.warnings)


def test_eval_args_enable_evaluation_when_eval_steps_is_set(audio_trainer):
    dataset = ["a", "b"]
    args, eval_dataset = audio_trainer._audio_eval_config(
        {"eval_dataset": dataset, "eval_steps": 0.1, "batch_size": 2}
    )
    assert eval_dataset is dataset
    assert args["eval_strategy"] == "steps"
    assert args["eval_steps"] == 0.1
    assert args["per_device_eval_batch_size"] == 2


def test_eval_steps_zero_disables_evaluation(audio_trainer):
    args, eval_dataset = audio_trainer._audio_eval_config(
        {"eval_dataset": ["a"], "eval_steps": 0.0}
    )
    assert (args, eval_dataset) == ({}, None)


def test_no_eval_dataset_disables_evaluation(audio_trainer):
    args, eval_dataset = audio_trainer._audio_eval_config({"eval_dataset": None, "eval_steps": 0.1})
    assert (args, eval_dataset) == ({}, None)


@pytest.mark.parametrize("eval_steps", [True, float("inf"), float("nan"), "abc"])
def test_hostile_eval_steps_disable_evaluation(audio_trainer, eval_steps):
    """`eval_steps <= 0` lets these through: True is every step, inf raises, NaN never fires."""
    args, eval_dataset = audio_trainer._audio_eval_config(
        {"eval_dataset": ["a", "b"], "eval_steps": eval_steps, "batch_size": 2}
    )
    assert (args, eval_dataset) == ({}, None)


def test_audio_eval_config_agrees_with_the_shared_validator(audio_trainer):
    from core.training.eval_dataset import evaluation_enabled
    for value in (0.1, 0.25, 1, 2, 0, 0.0, -1, None, True, False, float("inf"), float("nan")):
        _args, eval_dataset = audio_trainer._audio_eval_config(
            {"eval_dataset": ["a"], "eval_steps": value, "batch_size": 2}
        )
        assert (eval_dataset is not None) == evaluation_enabled(value), value


def test_empty_eval_split_disables_evaluation_with_a_warning(audio_trainer):
    args, eval_dataset = audio_trainer._audio_eval_config(
        {"eval_dataset": [], "eval_steps": 0.1, "batch_size": 2}
    )
    assert (args, eval_dataset) == ({}, None)
    assert any("empty" in w for w in audio_trainer.training_progress.warnings)


def test_missing_batch_size_falls_back_instead_of_passing_none(audio_trainer):
    args, _ = audio_trainer._audio_eval_config(
        {"eval_dataset": ["a"], "eval_steps": 0.1, "batch_size": None}
    )
    assert args["per_device_eval_batch_size"] == 2


def test_a_stop_during_eval_preprocessing_is_not_reported_as_a_bad_eval_file(audio_trainer):
    """A stop is reported as "no valid examples"; that is the cancel, not the user's upload."""
    audio_trainer.should_stop = True

    def stopped(dataset, custom_format_mapping = None):
        raise ValueError("No valid examples after CSM preprocessing (skipped 4)")

    assert audio_trainer._preprocess_audio_eval_split(object(), stopped, None) is None
    assert not audio_trainer.training_progress.warnings


def test_a_real_failure_still_warns_when_not_stopping(audio_trainer):
    def explode(dataset, custom_format_mapping = None):
        raise ValueError("no audio column found in dataset")

    assert audio_trainer._preprocess_audio_eval_split(object(), explode, None) is None
    assert any("no evaluation" in w for w in audio_trainer.training_progress.warnings)


def test_numeric_string_eval_steps_is_normalised(audio_trainer):
    """A numeric string is a valid cadence, but TrainingArguments compares it against an int."""
    args, eval_dataset = audio_trainer._audio_eval_config(
        {"eval_dataset": ["a"], "eval_steps": "0.1", "batch_size": 2}
    )
    assert eval_dataset is not None
    assert isinstance(args["eval_steps"], float) and args["eval_steps"] == 0.1


def test_a_length_less_eval_split_still_enables_evaluation(audio_trainer):
    """A streaming split has no row count and must not be mistaken for an empty one."""

    class _NoLen:
        pass

    args, eval_dataset = audio_trainer._audio_eval_config(
        {"eval_dataset": _NoLen(), "eval_steps": 0.1, "batch_size": 2}
    )
    assert eval_dataset is not None
    assert args["eval_strategy"] == "steps"


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_explicit_batch_sizes_are_preserved(audio_trainer, batch_size):
    args, _ = audio_trainer._audio_eval_config(
        {"eval_dataset": ["a"], "eval_steps": 0.1, "batch_size": batch_size}
    )
    assert args["per_device_eval_batch_size"] == batch_size


@pytest.mark.parametrize("eval_steps,expected", [(0.1, 0.1), (0.25, 0.25), (1, 1), (2, 2)])
def test_transformers_accepts_the_produced_config(audio_trainer, tmp_path, eval_steps, expected):
    """Normalising eval_steps to float must not change the cadence transformers ends up with."""
    transformers = pytest.importorskip("transformers")

    training_args = {
        "eval_dataset": ["a", "b"],
        "eval_steps": eval_steps,
        "batch_size": 2,
        "max_steps": 8,
        "optim": "adamw_torch",
    }
    eval_args, eval_dataset = audio_trainer._audio_eval_config(training_args)
    assert eval_dataset is not None
    config = audio_trainer._build_audio_training_args(
        training_args, str(tmp_path), extra_args = {"remove_unused_columns": False, **eval_args}
    )
    config.update(bf16 = False, fp16 = False, use_cpu = True, report_to = [])
    args = transformers.TrainingArguments(**config)
    assert args.eval_strategy == "steps"
    assert args.eval_steps == expected
    assert args.per_device_eval_batch_size == 2


def test_a_stop_skips_the_eval_preprocessor_entirely(audio_trainer):
    """A stopped train pass returns partial rows, so eval would reload a codec model to abort."""
    audio_trainer.should_stop = True
    calls = []

    def preprocess(dataset, custom_format_mapping = None):
        calls.append(dataset)
        return dataset

    assert audio_trainer._preprocess_audio_eval_split(object(), preprocess, None) is None
    assert calls == [], "the codec preprocessor ran after the run was stopped"
    assert not audio_trainer.training_progress.warnings


@pytest.mark.parametrize("eval_steps", [float("inf"), float("nan"), 0, -1])
@pytest.mark.parametrize("audio_type", CODEC_TYPES)
def test_an_invalid_cadence_never_preprocesses_the_eval_split(
    audio_trainer, tmp_path, monkeypatch, audio_type, eval_steps
):
    """Gating on `eval_steps > 0` let inf and NaN codec-encode a split _audio_eval_config drops."""
    audio_trainer._audio_type = audio_type
    monkeypatch.setattr(tmod, "ensure_audio_decoding", lambda: True)
    seen = []
    monkeypatch.setattr(
        audio_trainer,
        f"_preprocess_{audio_type}_dataset",
        lambda ds, m = None: (seen.append(len(ds)), ds)[1],
        raising = True,
    )

    _train, evaluation = audio_trainer.load_and_format_dataset(
        None,
        local_datasets = [_rows(tmp_path / "train.jsonl", "train")],
        local_eval_datasets = [_rows(tmp_path / "eval.jsonl", "eval")],
        eval_steps = eval_steps,
    )

    assert evaluation is None
    assert len(seen) == 1, f"eval_steps={eval_steps} still preprocessed the eval split"


@pytest.mark.parametrize(
    "eval_steps,expect_enabled",
    [(0.25, True), (2, True), (float("inf"), False), (float("nan"), False), (0, False)],
)
def test_the_generic_sft_path_uses_the_same_cadence_gate(
    audio_trainer, tmp_path, monkeypatch, eval_steps, expect_enabled
):
    """BiCodec and DAC fall through here, so this gate must match the one in _audio_eval_config."""
    captured = {}

    class _FakeSFTConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeSFTTrainer:
        def __init__(self, **kwargs):
            captured["trainer_kwargs"] = kwargs

        def add_callback(self, cb):
            pass

        def train(self, **kwargs):
            pass

    monkeypatch.setattr(tmod, "SFTConfig", _FakeSFTConfig, raising = False)
    monkeypatch.setattr(tmod, "SFTTrainer", _FakeSFTTrainer, raising = False)
    monkeypatch.setattr(tmod, "resolve_output_dir", lambda p: tmp_path, raising = True)
    monkeypatch.setattr(tmod, "ensure_dir", lambda p: p, raising = True)
    monkeypatch.setattr(tmod, "_drop_hf_stdout_callbacks", lambda trainer: None, raising = True)
    monkeypatch.setattr(
        tmod.UnslothTrainer, "_finalize_training", lambda self, *a, **k: None, raising = True
    )
    monkeypatch.setattr(
        tmod.UnslothTrainer, "_preflight_first_batch", lambda self: None, raising = True
    )

    audio_trainer._audio_type = "bicodec"
    audio_trainer.model = object()
    audio_trainer.tokenizer = object()
    audio_trainer.model_name = "unsloth/spark-tts"

    rows = [{"text": "a"}, {"text": "b"}]
    try:
        audio_trainer._train_worker(
            {"dataset": rows, "final_format": "audio_bicodec"},
            eval_dataset = rows,
            eval_steps = eval_steps,
            batch_size = 2,
            gradient_accumulation_steps = 1,
            max_steps = 8,
            warmup_steps = 0,
            output_dir = str(tmp_path),
        )
    except Exception:
        # The generic path needs a real model; the eval decision happens before that, so only
        # the captured config matters.
        pass

    if expect_enabled:
        assert captured.get("eval_strategy") == "steps", f"eval_steps={eval_steps} was not enabled"
    else:
        assert captured, "the config was never built, so this asserts nothing"
        assert (
            "eval_strategy" not in captured
        ), f"eval_steps={eval_steps} reached TrainingArguments as a cadence"
