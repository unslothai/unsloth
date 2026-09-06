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
