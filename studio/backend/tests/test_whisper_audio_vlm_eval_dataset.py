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


def test_whisper_uses_the_uploaded_eval_split(audio_trainer, tmp_path, monkeypatch):
    """eval_split is None for a local upload, so nothing used to reach Whisper."""
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
    """Dataset stand-in: real Audio() casting needs torchcodec, absent on CPU runners."""

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
        base = (["audio"] if self._with_audio else []) + ["text"]
        return base + list(getattr(self, "_phantom_columns", []))

    def cast_column(self, column, feature):
        """Like cast_column: for a decode_example feature (Audio) the name is not
        validated, so a missing column is silently ADDED as all-null."""
        if column not in self.column_names:
            phantom = _FakeAudioDataset(self._texts, with_audio = False)
            phantom._phantom_columns = list(getattr(self, "_phantom_columns", [])) + [column]
            return phantom
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
        for column in getattr(self, "_phantom_columns", []):
            row[column] = None
        return row


def _audio_rows(texts):
    return _FakeAudioDataset(texts)


def test_whisper_preprocess_prefers_the_separate_split_over_the_carve_out(audio_trainer):
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


def test_whisper_eval_split_without_an_audio_column_warns(audio_trainer):
    audio_trainer.tokenizer = _FakeWhisperTokenizer()

    train_data, eval_data = audio_trainer._preprocess_whisper_dataset(
        _audio_rows(["tr-1", "tr-2"]),
        eval_split = None,
        eval_dataset = _FakeAudioDataset(["ev-1"], with_audio = False),
    )

    assert len(train_data) == 2, "a bad eval split must not take the training data with it"
    assert eval_data is None
    assert any("no evaluation" in w for w in audio_trainer.training_progress.warnings)


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


def test_whisper_length_less_eval_split_warns_instead_of_crashing(audio_trainer):
    class _NoLen(_FakeAudioDataset):
        def __len__(self):
            raise TypeError("object of type 'IterableDataset' has no len()")

    audio_trainer.tokenizer = _FakeWhisperTokenizer()

    train_data, eval_data = audio_trainer._preprocess_whisper_dataset(
        _audio_rows(["tr-1", "tr-2"]),
        eval_split = None,
        eval_dataset = _NoLen(["ev-1"]),
    )

    assert len(train_data) == 2
    assert not eval_data
    assert any("no evaluation" in w for w in audio_trainer.training_progress.warnings)


def test_whisper_zero_row_eval_split_is_falsy(audio_trainer):
    audio_trainer.tokenizer = _FakeWhisperTokenizer()

    train_data, eval_data = audio_trainer._preprocess_whisper_dataset(
        _audio_rows(["tr-1", "tr-2"]),
        eval_split = None,
        eval_dataset = _audio_rows([]),
    )

    assert len(train_data) == 2
    assert not eval_data


def test_whisper_cancel_during_eval_preprocessing_leaves_no_eval(audio_trainer):
    tokenizer = _FakeWhisperTokenizer()
    real_extractor = tokenizer.feature_extractor
    calls = {"n": 0}

    def counting_extractor(array, sampling_rate = None):
        calls["n"] += 1
        if calls["n"] == 2:  # both train rows are done
            audio_trainer.should_stop = True
        return real_extractor(array, sampling_rate = sampling_rate)

    tokenizer.feature_extractor = counting_extractor
    audio_trainer.tokenizer = tokenizer

    train_data, eval_data = audio_trainer._preprocess_whisper_dataset(
        _audio_rows(["tr-1", "tr-2"]),
        eval_split = None,
        eval_dataset = _audio_rows(["ev-1"]),
    )

    assert len(train_data) == 2
    assert not eval_data


def test_audio_vlm_eval_split_with_a_different_audio_column_is_refused(
    audio_trainer, tmp_path, monkeypatch
):
    """audio_vlm_collate_fn reads one recorded audio column for BOTH splits, so an eval split
    must never redefine it."""
    audio_trainer._audio_type = None
    audio_trainer.is_audio_vlm = True
    columns = iter(["audio", "speech"])

    def fake_format(dataset, custom_format_mapping = None):
        audio_trainer._audio_vlm_audio_col = next(columns)
        return dataset

    monkeypatch.setattr(audio_trainer, "_format_audio_vlm_dataset", fake_format, raising = True)

    _train, evaluation = audio_trainer.load_and_format_dataset(
        None,
        local_datasets = [_rows(tmp_path / "train.jsonl", ["tr-1"])],
        local_eval_datasets = [_rows(tmp_path / "eval.jsonl", ["ev-1"])],
        eval_steps = 0.1,
    )

    assert evaluation is None
    assert (
        audio_trainer._audio_vlm_audio_col == "audio"
    ), "the eval split redefined the column the train collator reads"
    assert any("same audio column name" in w for w in audio_trainer.training_progress.warnings)


def test_audio_vlm_matching_eval_split_leaves_the_column_alone(
    audio_trainer, tmp_path, monkeypatch
):
    audio_trainer._audio_type = None
    audio_trainer.is_audio_vlm = True

    def fake_format(dataset, custom_format_mapping = None):
        audio_trainer._audio_vlm_audio_col = "audio"
        return dataset

    monkeypatch.setattr(audio_trainer, "_format_audio_vlm_dataset", fake_format, raising = True)

    _train, evaluation = audio_trainer.load_and_format_dataset(
        None,
        local_datasets = [_rows(tmp_path / "train.jsonl", ["tr-1"])],
        local_eval_datasets = [_rows(tmp_path / "eval.jsonl", ["ev-1"])],
        eval_steps = 0.1,
    )

    assert evaluation is not None
    assert audio_trainer._audio_vlm_audio_col == "audio"


def _drive_whisper_branch(audio_trainer, tmp_path, monkeypatch, *, eval_rows):
    """Drive the real Whisper _train_worker branch; only train() is stubbed, so the asserted
    args and eval dataset are genuine transformers objects."""
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(transformers.Seq2SeqTrainer, "train", lambda self, **kw: None)
    monkeypatch.setattr(tmod, "_drop_hf_stdout_callbacks", lambda trainer: None)
    monkeypatch.setattr(
        tmod.UnslothTrainer, "_finalize_training", lambda self, *a, **k: None, raising = True
    )
    monkeypatch.setattr(tmod, "resolve_output_dir", lambda p: tmp_path, raising = True)
    monkeypatch.setattr(tmod, "ensure_dir", lambda p: p, raising = True)
    monkeypatch.setattr(tmod, "is_bfloat16_supported", lambda: False, raising = False)

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = transformers.PretrainedConfig()
            self.lin = torch.nn.Linear(1, 1)

        def forward(self, **kwargs):
            return {"loss": self.lin(torch.zeros(1, 1)).sum()}

    class _FakeProcessor:
        feature_extractor = object()
        tokenizer = object()

    audio_trainer._audio_type = "whisper"
    audio_trainer.model = _TinyModel()
    audio_trainer.tokenizer = _FakeProcessor()

    monkeypatch.setattr(
        tmod,
        "DataCollatorSpeechSeq2SeqWithPadding",
        lambda processor: lambda f: f,
        raising = False,
    )

    audio_trainer._train_worker(
        [{"input_features": [0.0], "labels": [1]}] * 4,
        eval_dataset = eval_rows,
        eval_steps = 0.25,
        batch_size = 2,
        gradient_accumulation_steps = 1,
        max_steps = 8,
        warmup_steps = 0,
        num_epochs = 1,
        output_dir = str(tmp_path),
        fp16 = False,
        bf16 = False,
    )
    return audio_trainer.trainer


def test_whisper_trainer_branch_wires_eval(audio_trainer, tmp_path, monkeypatch):
    """HF defaults per_device_eval_batch_size to 8, which can OOM an audio eval pass."""
    eval_rows = [{"input_features": [0.0], "labels": [2]}] * 2
    trainer = _drive_whisper_branch(audio_trainer, tmp_path, monkeypatch, eval_rows = eval_rows)

    assert trainer is not None, "the Whisper branch did not build a trainer"
    assert trainer.eval_dataset is eval_rows
    assert trainer.args.eval_strategy == "steps"
    assert (
        trainer.args.per_device_eval_batch_size == 2
    ), "Whisper eval falls back to HF's default batch size of 8"
    assert trainer.args.per_device_train_batch_size == 2
    assert trainer.args.remove_unused_columns is False


def test_whisper_trainer_branch_omits_eval_when_there_is_none(audio_trainer, tmp_path, monkeypatch):
    trainer = _drive_whisper_branch(audio_trainer, tmp_path, monkeypatch, eval_rows = None)

    assert trainer is not None
    assert trainer.eval_dataset is None
    assert trainer.args.eval_strategy == "no"


def test_whisper_trainer_branch_omits_eval_for_an_empty_split(audio_trainer, tmp_path, monkeypatch):
    trainer = _drive_whisper_branch(audio_trainer, tmp_path, monkeypatch, eval_rows = [])

    assert trainer is not None
    assert not trainer.eval_dataset
    assert trainer.args.eval_strategy == "no"


def test_whisper_eval_split_whose_rows_are_all_skipped_warns(audio_trainer):
    class _EmptyRows(_FakeAudioDataset):
        def __getitem__(self, idx):
            return {"audio": None, "text": ""}

    audio_trainer.tokenizer = _FakeWhisperTokenizer()

    train_data, eval_data = audio_trainer._preprocess_whisper_dataset(
        _audio_rows(["tr-1", "tr-2"]),
        eval_split = None,
        eval_dataset = _EmptyRows(["ev-1", "ev-2"]),
    )

    assert len(train_data) == 2
    assert not eval_data
    assert any("No usable rows" in w for w in audio_trainer.training_progress.warnings)


def test_whisper_cancel_does_not_warn_about_the_eval_file(audio_trainer):
    """A stop empties the eval pass too; that is the cancel, not the user's data."""
    tokenizer = _FakeWhisperTokenizer()
    real_extractor = tokenizer.feature_extractor
    calls = {"n": 0}

    def counting_extractor(array, sampling_rate = None):
        calls["n"] += 1
        if calls["n"] == 2:
            audio_trainer.should_stop = True
        return real_extractor(array, sampling_rate = sampling_rate)

    tokenizer.feature_extractor = counting_extractor
    audio_trainer.tokenizer = tokenizer

    _train_data, eval_data = audio_trainer._preprocess_whisper_dataset(
        _audio_rows(["tr-1", "tr-2"]),
        eval_split = None,
        eval_dataset = _audio_rows(["ev-1"]),
    )

    assert not eval_data
    assert not any("No usable rows" in w for w in audio_trainer.training_progress.warnings)


def test_cast_column_really_does_not_validate_the_column_name():
    """Pins the datasets behaviour the explicit column check exists for, so _FakeAudioDataset
    cannot drift from it."""
    datasets = pytest.importorskip("datasets")

    ds = datasets.Dataset.from_dict({"path": ["/a.wav"], "text": ["x"]})
    cast = ds.cast_column("audio", datasets.Audio(sampling_rate = 16000))

    assert "audio" in cast.column_names, "cast_column started validating; simplify the guard"
    assert cast[0]["audio"] is None

    with pytest.raises(ValueError):
        # A feature without decode_example goes through Dataset.cast, which does validate.
        ds.cast_column("audio", datasets.Value("string"))
