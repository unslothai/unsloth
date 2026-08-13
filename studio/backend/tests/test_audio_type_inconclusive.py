# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An unreadable tokenizer_config.json must not read as "not an audio model".

`detect_audio_type` returns None for both "definitely not audio" and "could not
tell", and the trainer used it to choose a preprocessing path. A TTS run whose
repo could not be read therefore took the text path and failed much later with
"Could not auto-detect format mapping", which names a column-mapping problem
rather than the read that failed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.training.trainer import UnslothTrainer

_BACKEND = Path(__file__).resolve().parents[1]
_load_and_format_dataset = UnslothTrainer.load_and_format_dataset


class _Dataset:
    column_names = ["audio", "text"]

    def __len__(self):
        return 4


def _trainer(*, audio_type, known, dataset_audio):
    """A stand-in carrying only what load_and_format_dataset reads."""
    errors: list[str] = []
    trainer = SimpleNamespace(
        should_stop = False,
        _audio_type = audio_type,
        _audio_type_known = known,
        _is_dataset_audio = dataset_audio,
        is_audio_vlm = False,
        is_vlm = False,
        model_name = "org/model",
        tokenizer = None,
        errors = errors,
        _update_progress = lambda **kw: errors.append(kw.get("error")) if kw.get("error") else None,
        _resolve_eval_split_from_dataset = lambda dataset: None,
    )
    trainer.load_and_format_dataset = _load_and_format_dataset.__get__(trainer)
    return trainer


def _run(
    trainer,
    monkeypatch,
    dataset = None,
):
    from hub.utils import dataset_cache

    monkeypatch.setattr(
        dataset_cache, "load_cached_hf_dataset", lambda *a, **k: dataset or _Dataset()
    )
    monkeypatch.setattr(
        "core.training.trainer.format_and_template_dataset",
        lambda dataset, **kw: {"dataset": dataset, "detected_format": "test", "warnings": []},
    )
    monkeypatch.setattr(
        "core.training.trainer.load_dataset",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no remote access")),
    )
    monkeypatch.setattr(
        sys.modules["datasets"],
        "get_dataset_split_names",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no remote access")),
    )
    return trainer.load_and_format_dataset(
        "org/dataset",
        dataset_local_files_only = True,
        dataset_local_path = "/cache/snapshot",
    )


def test_an_inconclusive_probe_on_an_audio_dataset_is_refused(monkeypatch):
    trainer = _trainer(audio_type = None, known = False, dataset_audio = True)
    assert _run(trainer, monkeypatch) is None
    assert trainer.errors, "the run failed without reporting why"
    reported = trainer.errors[0]
    # The point of the change: the message names the read that failed, not a column map.
    assert "tokenizer_config.json" in reported
    assert "Could not auto-detect format mapping" not in reported


def test_a_definitive_non_audio_model_still_takes_the_text_path(monkeypatch):
    # The common case. A probe that succeeded and said "not audio" is unaffected.
    trainer = _trainer(audio_type = None, known = True, dataset_audio = True)
    assert _run(trainer, monkeypatch) is not None
    assert not trainer.errors


def test_an_inconclusive_probe_without_audio_data_is_left_alone(monkeypatch):
    # A text run whose repo happens to be gated must not start failing.
    trainer = _trainer(audio_type = None, known = False, dataset_audio = False)
    assert _run(trainer, monkeypatch) is not None
    assert not trainer.errors


def test_the_trainer_probes_with_the_checked_variant():
    # detect_audio_type collapses "not audio" and "unreadable" into None, and its own
    # docstring sends callers gating a user action to the checked variant instead.
    text = (_BACKEND / "core/training/trainer.py").read_text(encoding = "utf-8")
    assert "detect_audio_type_checked(" in text
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("self._audio_type") and "detect_audio_type" in stripped:
            assert "detect_audio_type_checked(" in stripped, stripped


def test_every_probe_site_records_whether_it_was_definitive():
    # A site that reassigns _audio_type without its flag leaves the flag describing
    # the previous probe, which is worse than not having it.
    text = (_BACKEND / "core/training/trainer.py").read_text(encoding = "utf-8")
    sites = [
        l for l in text.splitlines() if "detect_audio_type_checked(" in l and "import" not in l
    ]
    assert sites, "no probe sites found"
    for line in sites:
        assert "self._audio_type_known" in line, line.strip()


class _TypedDataset(_Dataset):
    """A dataset that reports a real `features` schema, as a loaded HF dataset does."""

    def __init__(self, features):
        self.features = features


def _audio_features():
    from datasets import Audio, Value
    return {"audio": Audio(sampling_rate = 16000), "text": Value("string")}


def _text_features():
    from datasets import Value
    return {"audio": Value("string"), "text": Value("string")}


def test_a_text_dataset_with_a_column_named_audio_is_not_refused(monkeypatch):
    # is_dataset_audio comes from the client, which sets it from the format check's
    # is_audio -- and that is true on a column-NAME keyword match alone, without ever
    # looking at the value. A text dataset with a column called `audio` holding
    # filenames therefore arrives here flagged as audio. Refusing it would take away a
    # text path that works, so the dataset's own schema gets the last word.
    trainer = _trainer(audio_type = None, known = False, dataset_audio = True)
    result = _run(trainer, monkeypatch, dataset = _TypedDataset(_text_features()))
    assert result is not None, "a text dataset was refused for having an audio-named column"
    assert not trainer.errors


def test_a_real_audio_column_is_still_refused(monkeypatch):
    # The case the guard exists for: the schema confirms audio, so the run must stop
    # rather than fall through to a text path that cannot map it.
    trainer = _trainer(audio_type = None, known = False, dataset_audio = True)
    assert _run(trainer, monkeypatch, dataset = _TypedDataset(_audio_features())) is None
    assert trainer.errors and "tokenizer_config.json" in trainer.errors[0]


def test_a_dataset_that_cannot_report_a_schema_is_still_refused(monkeypatch):
    # A DatasetDict or an iterable dataset has no usable `features`. Unknown is not
    # "no audio here", and an unreadable model probe remains the likelier explanation,
    # so the guard stays closed only on a positive answer.
    trainer = _trainer(audio_type = None, known = False, dataset_audio = True)
    assert _run(trainer, monkeypatch, dataset = _Dataset()) is None
    assert trainer.errors and "tokenizer_config.json" in trainer.errors[0]
