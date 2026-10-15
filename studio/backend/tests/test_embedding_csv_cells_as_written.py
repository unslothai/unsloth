# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import importlib  # noqa: E402
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

from core.training import trainer as _trainer  # noqa: E402,F401
from core.training import worker  # noqa: E402

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__spec__ = None
    mod.__dict__.update(attrs)
    return mod


@pytest.fixture
def run_embedding(monkeypatch, tmp_path):
    trained = {}

    class Trainer:
        def __init__(self, **kwargs):
            trained.update(kwargs)

        def remove_callback(self, callback):
            pass

        def train(self, **kwargs):
            pass

    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        _module(
            "unsloth",
            FastSentenceTransformer = MagicMock(),
            is_bfloat16_supported = lambda: False,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        _module(
            "sentence_transformers",
            SentenceTransformerTrainer = Trainer,
            SentenceTransformerTrainingArguments = MagicMock(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers.losses",
        _module("sentence_transformers.losses", MultipleNegativesRankingLoss = MagicMock()),
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers.training_args",
        _module("sentence_transformers.training_args", BatchSamplers = MagicMock()),
    )
    monkeypatch.setattr(worker, "_model_load_security_error", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_start_worker_stop_poller", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_emit_resource_provenance", lambda *a, **k: None)

    def run(csv_text):
        path = tmp_path / "pairs.csv"
        path.write_text(csv_text)
        events = []
        worker._run_embedding_training(
            types.SimpleNamespace(put = events.append),
            None,
            {
                "model_name": "unsloth/all-MiniLM-L6-v2",
                "training_type": "Full Finetuning",
                "local_datasets": [str(path)],
                "output_dir": "embedding-csv-cells",
                "max_steps": 1,
            },
        )
        errors = [event["error"] for event in events if event["type"] == "error"]
        assert errors == []
        warnings = [event["message"] for event in events if event["type"] == "warning"]
        return trained["train_dataset"], warnings

    return run


def test_uploaded_csv_trains_embedding_pairs_as_written(run_embedding):
    dataset, warnings = run_embedding(
        "query,answer\n"
        "Zip code of Holtsville NY?,00501\n"
        "Price for item 7?,NA\n"
        "What does the API return on a miss?,None\n"
        "Say nothing.,\n"
        "Zip code of Schenectady NY?,12345\n"
    )

    assert dataset.to_list() == [
        {"query": "Zip code of Holtsville NY?", "answer": "00501"},
        {"query": "Price for item 7?", "answer": "NA"},
        {"query": "What does the API return on a miss?", "answer": "None"},
        {"query": "Zip code of Schenectady NY?", "answer": "12345"},
    ]
    assert warnings == ["Dropped 1 of 5 rows because a text cell was empty."]


def test_embedding_csv_score_column_still_reaches_the_trainer_as_numbers(run_embedding):
    dataset, warnings = run_embedding(
        "sentence1,sentence2,score\nA cat sits.,A cat is sitting.,0.9\nA dog runs.,Rain falls.,0\n"
    )

    assert dataset.to_list() == [
        {"sentence1": "A cat sits.", "sentence2": "A cat is sitting.", "score": 0.9},
        {"sentence1": "A dog runs.", "sentence2": "Rain falls.", "score": 0.0},
    ]
    assert warnings == []
