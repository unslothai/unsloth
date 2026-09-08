# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for preserving the eval split on the Whisper and audio-VLM paths."""

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


@pytest.fixture
def audio_trainer(monkeypatch):
    # Avoid MLX substitution on Apple silicon.
    monkeypatch.setattr(tmod, "should_use_mlx_training_backend", lambda *a, **k: False)
    monkeypatch.setattr(tmod, "ensure_audio_decoding", lambda: True)
    t = tmod.UnslothTrainer()
    t.model_name = "unsloth/whisper-small"
    return t


def _rows(path: Path, texts) -> str:
    with path.open("w", encoding = "utf-8") as fh:
        for text in texts:
            fh.write(json.dumps({"audio": text, "text": text}) + "\n")
    return str(path)


def _texts(dataset):
    if isinstance(dataset, dict):
        dataset = dataset["dataset"]
    return list(dataset["text"])


# ---------------------------------------------------------------------------- Whisper


def test_whisper_uses_the_uploaded_eval_split(audio_trainer, tmp_path, monkeypatch):
    """An uploaded eval file used to be dropped: eval_split is None for a local upload, so
    the 6% carve-out never ran and the run trained with no evaluation at all."""
    audio_trainer._audio_type = "whisper"
    seen = {}

    def fake_preprocess(
        dataset,
        eval_split = None,
        custom_format_mapping = None,
        eval_dataset = None,
    ):
        seen["eval_split"] = eval_split
        seen["eval_rows"] = None if eval_dataset is None else list(eval_dataset["text"])
        return (list(dataset["text"]), None if eval_dataset is None else list(eval_dataset["text"]))

    monkeypatch.setattr(audio_trainer, "_preprocess_whisper_dataset", fake_preprocess, raising = True)

    train, evaluation = audio_trainer.load_and_format_dataset(
        None,
        local_datasets = [_rows(tmp_path / "train.jsonl", ["tr-1", "tr-2", "tr-3"])],
        local_eval_datasets = [_rows(tmp_path / "eval.jsonl", ["ev-1", "ev-2"])],
        eval_steps = 0.1,
    )

    assert seen["eval_rows"] == ["ev-1", "ev-2"], "the uploaded eval split never reached Whisper"
    assert train == ["tr-1", "tr-2", "tr-3"]
    assert evaluation == ["ev-1", "ev-2"]


def test_whisper_without_an_eval_upload_is_unchanged(audio_trainer, tmp_path, monkeypatch):
    audio_trainer._audio_type = "whisper"
    seen = {}

    def fake_preprocess(
        dataset,
        eval_split = None,
        custom_format_mapping = None,
        eval_dataset = None,
    ):
        seen["eval_dataset"] = eval_dataset
        return (list(dataset["text"]), None)

    monkeypatch.setattr(audio_trainer, "_preprocess_whisper_dataset", fake_preprocess, raising = True)

    _train, evaluation = audio_trainer.load_and_format_dataset(
        None,
        local_datasets = [_rows(tmp_path / "train.jsonl", ["tr-1"])],
        eval_steps = 0.1,
    )
    assert seen["eval_dataset"] is None
    assert evaluation is None


class _FakeWhisperTokenizer:
    """The two members _preprocess_whisper_dataset touches."""

    class _Features:
        def __init__(self, arrays):
            self.input_features = arrays

    def feature_extractor(
        self,
        array,
        sampling_rate = None,
    ):
        return self._Features([list(array)])

    def tokenizer(self, text):
        return types.SimpleNamespace(input_ids = [len(text)])


class _FakeAudioDataset:
    """A datasets.Dataset stand-in for the members _preprocess_whisper_dataset touches.

    Real Audio() casting needs torchcodec, which the CPU test runners do not ship.
    """

    def __init__(
        self,
        texts,
        *,
        with_audio = True,
    ):
        self._texts = list(texts)
        self._with_audio = with_audio

    @property
    def column_names(self):
        return (["audio"] if self._with_audio else []) + ["text"]

    def cast_column(self, column, feature):
        if column not in self.column_names:
            raise ValueError(
                f"Column {column} not in the dataset. Current columns: {self.column_names}"
            )
        return self

    def train_test_split(
        self,
        test_size = None,
        seed = None,
    ):
        n_eval = max(1, round(len(self._texts) * test_size))
        return {
            "train": _FakeAudioDataset(self._texts[:-n_eval], with_audio = self._with_audio),
            "test": _FakeAudioDataset(self._texts[-n_eval:], with_audio = self._with_audio),
        }

    def __len__(self):
        return len(self._texts)

    def __getitem__(self, idx):
        row = {"text": self._texts[idx]}
        if self._with_audio:
            row["audio"] = {"array": [0.0] * 160, "sampling_rate": 16000}
        return row


def _audio_rows(texts):
    return _FakeAudioDataset(texts)


def test_whisper_preprocess_prefers_the_separate_split_over_the_carve_out(audio_trainer):
    """With a separate split the train set must stay whole: the 6% carve-out is only the
    fallback for when there is no separate eval source."""
    audio_trainer.tokenizer = _FakeWhisperTokenizer()

    train_data, eval_data = audio_trainer._preprocess_whisper_dataset(
        _audio_rows([f"tr-{i}" for i in range(20)]),
        eval_split = "validation",
        eval_dataset = _audio_rows(["ev-1", "ev-2"]),
    )

    assert len(train_data) == 20, "the train set was carved despite a separate eval split"
    assert len(eval_data) == 2


def test_whisper_preprocess_still_carves_when_there_is_no_separate_split(audio_trainer):
    audio_trainer.tokenizer = _FakeWhisperTokenizer()

    train_data, eval_data = audio_trainer._preprocess_whisper_dataset(
        _audio_rows([f"tr-{i}" for i in range(100)]),
        eval_split = "validation",
        eval_dataset = None,
    )

    assert len(train_data) == 94
    assert len(eval_data) == 6


def test_whisper_preprocess_without_eval_returns_none(audio_trainer):
    audio_trainer.tokenizer = _FakeWhisperTokenizer()

    train_data, eval_data = audio_trainer._preprocess_whisper_dataset(
        _audio_rows(["tr-1", "tr-2"]), eval_split = None, eval_dataset = None
    )

    assert len(train_data) == 2
    assert eval_data is None


def test_whisper_unusable_eval_split_warns_instead_of_failing_the_run(audio_trainer):
    audio_trainer.tokenizer = _FakeWhisperTokenizer()

    train_data, eval_data = audio_trainer._preprocess_whisper_dataset(
        _audio_rows(["tr-1", "tr-2"]),
        eval_split = None,
        eval_dataset = _FakeAudioDataset(["ev-1"], with_audio = False),
    )

    assert len(train_data) == 2, "a bad eval split must not take the training data with it"
    assert eval_data is None
    assert any("no evaluation" in w for w in audio_trainer.training_progress.warnings)


# ---------------------------------------------------------------------------- audio VLM


def test_audio_vlm_uses_the_uploaded_eval_split(audio_trainer, tmp_path, monkeypatch):
    audio_trainer._audio_type = None
    audio_trainer.is_audio_vlm = True
    seen = []

    def fake_format(dataset, custom_format_mapping = None):
        seen.append(list(dataset["text"]))
        return dataset

    monkeypatch.setattr(audio_trainer, "_format_audio_vlm_dataset", fake_format, raising = True)

    train, evaluation = audio_trainer.load_and_format_dataset(
        None,
        local_datasets = [_rows(tmp_path / "train.jsonl", ["tr-1", "tr-2"])],
        local_eval_datasets = [_rows(tmp_path / "eval.jsonl", ["ev-1"])],
        eval_steps = 0.1,
    )

    assert evaluation is not None, "the uploaded eval split was dropped"
    assert _texts(train) == ["tr-1", "tr-2"]
    assert _texts(evaluation) == ["ev-1"], "eval split aliases the train split"
    assert seen == [["tr-1", "tr-2"], ["ev-1"]]


def test_audio_vlm_without_an_eval_upload_returns_none(audio_trainer, tmp_path, monkeypatch):
    audio_trainer._audio_type = None
    audio_trainer.is_audio_vlm = True
    monkeypatch.setattr(
        audio_trainer, "_format_audio_vlm_dataset", lambda ds, m = None: ds, raising = True
    )

    _train, evaluation = audio_trainer.load_and_format_dataset(
        None,
        local_datasets = [_rows(tmp_path / "train.jsonl", ["tr-1"])],
        eval_steps = 0.1,
    )
    assert evaluation is None


def test_audio_vlm_unpreparable_eval_split_warns_instead_of_failing_the_run(
    audio_trainer, tmp_path, monkeypatch
):
    audio_trainer._audio_type = None
    audio_trainer.is_audio_vlm = True
    calls = {"n": 0}

    def flaky(dataset, custom_format_mapping = None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("no audio column found in dataset")
        return dataset

    monkeypatch.setattr(audio_trainer, "_format_audio_vlm_dataset", flaky, raising = True)

    train, evaluation = audio_trainer.load_and_format_dataset(
        None,
        local_datasets = [_rows(tmp_path / "train.jsonl", ["tr-1", "tr-2"])],
        local_eval_datasets = [_rows(tmp_path / "eval.jsonl", ["ev-1"])],
        eval_steps = 0.1,
    )

    assert _texts(train) == ["tr-1", "tr-2"]
    assert evaluation is None
    assert any("no evaluation" in w for w in audio_trainer.training_progress.warnings)
