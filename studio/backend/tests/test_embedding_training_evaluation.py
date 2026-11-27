# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
from datasets import Dataset  # noqa: E402

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


def _pairs(prefix, count):
    return [{"anchor": f"{prefix} q{i}", "positive": f"{prefix} a{i}"} for i in range(count)]


HUB_SPLITS = {"train": _pairs("train", 40), "dev": _pairs("dev", 20)}


@pytest.fixture
def run_embedding(monkeypatch, tmp_path):
    seen = {}

    class Trainer:
        def __init__(self, **kwargs):
            seen["trainer"] = kwargs

        def remove_callback(self, callback):
            pass

        def train(self, **kwargs):
            pass

    def training_arguments(**kwargs):
        seen["args"] = kwargs
        return types.SimpleNamespace(**kwargs)

    from utils.datasets import cache_safe

    load_files = cache_safe.load_dataset_cache_safe

    def load_dataset(path, *args, split, **kwargs):
        if path in ("csv", "json"):
            return load_files(path, *args, split = split, **kwargs)
        return Dataset.from_list(HUB_SPLITS[split])

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
            SentenceTransformerTrainingArguments = training_arguments,
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
    monkeypatch.setattr(cache_safe, "load_dataset_cache_safe", load_dataset)
    monkeypatch.setattr(worker, "_model_load_security_error", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_start_worker_stop_poller", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_emit_resource_provenance", lambda *a, **k: True)

    def write_csv(name, rows):
        path = tmp_path / name
        path.write_text(
            "anchor,positive\n" + "".join(f"{r['anchor']},{r['positive']}\n" for r in rows)
        )
        return str(path)

    def run(**config):
        events = []
        worker._run_embedding_training(
            types.SimpleNamespace(put = events.append),
            None,
            {
                "model_name": "unsloth/all-MiniLM-L6-v2",
                "training_type": "Full Finetuning",
                "output_dir": "embedding-eval",
                "max_steps": 10,
                "batch_size": 8,
                **config,
            },
        )
        assert [e for e in events if e["type"] == "error"] == []
        return seen["trainer"], seen["args"], events

    run.write_csv = write_csv
    return run


def _rows(dataset):
    return dataset.to_list() if dataset is not None else None


def test_embedding_run_evaluates_on_the_chosen_eval_split(run_embedding):
    trainer, args, _ = run_embedding(
        hf_dataset = "sentence-transformers/all-nli",
        train_split = "train",
        eval_split = "dev",
        eval_steps = 0.1,
        dataset_slice_end = 9,
    )

    assert _rows(trainer["eval_dataset"]) == HUB_SPLITS["dev"]
    assert len(trainer["train_dataset"]) == 10
    assert args["eval_strategy"] == "steps"
    assert args["eval_steps"] == 0.1
    assert args["per_device_eval_batch_size"] == 8


def test_embedding_run_without_eval_steps_does_not_evaluate(run_embedding):
    trainer, args, _ = run_embedding(
        hf_dataset = "sentence-transformers/all-nli",
        eval_split = "dev",
        eval_steps = 0,
    )

    assert trainer.get("eval_dataset") is None
    assert "eval_strategy" not in args
    assert len(trainer["train_dataset"]) == 40


def test_embedding_run_evaluates_on_uploaded_eval_file(run_embedding):
    trainer, args, _ = run_embedding(
        local_datasets = [run_embedding.write_csv("train.csv", _pairs("train", 40))],
        local_eval_datasets = [run_embedding.write_csv("eval.csv", _pairs("eval", 5))],
        eval_steps = 0.25,
    )

    assert _rows(trainer["eval_dataset"]) == _pairs("eval", 5)
    assert len(trainer["train_dataset"]) == 40
    assert args["eval_steps"] == 0.25


def test_embedding_run_evaluates_on_eval_folder_validation_split(run_embedding, tmp_path):
    folder = tmp_path / "eval_folder"
    folder.mkdir()
    run_embedding.write_csv("eval_folder/train.csv", _pairs("leak", 7))
    run_embedding.write_csv("eval_folder/validation.csv", _pairs("eval", 5))
    trainer, _, _ = run_embedding(
        local_datasets = [run_embedding.write_csv("train.csv", _pairs("train", 40))],
        local_eval_datasets = [str(folder)],
        eval_steps = 0.25,
    )

    assert _rows(trainer["eval_dataset"]) == _pairs("eval", 5)


def test_embedding_run_holds_out_eval_rows_when_no_eval_source(run_embedding):
    trainer, args, _ = run_embedding(
        local_datasets = [run_embedding.write_csv("train.csv", _pairs("train", 40))],
        eval_steps = 0.1,
    )

    train_rows = _rows(trainer["train_dataset"])
    eval_rows = _rows(trainer["eval_dataset"])
    assert eval_rows and train_rows
    assert sorted(map(str, train_rows + eval_rows)) == sorted(map(str, _pairs("train", 40)))
    assert args["eval_strategy"] == "steps"


def test_embedding_run_too_small_to_hold_out_trains_without_eval(run_embedding):
    trainer, args, events = run_embedding(
        local_datasets = [run_embedding.write_csv("train.csv", _pairs("train", 5))],
        eval_steps = 0.1,
    )

    assert trainer.get("eval_dataset") is None
    assert "eval_strategy" not in args
    assert any(e["type"] == "warning" and "without evaluation" in e["message"] for e in events)


def test_embedding_run_csv_train_with_jsonl_eval_upload(run_embedding, tmp_path):
    eval_path = tmp_path / "eval.jsonl"
    eval_rows = [{"sentence1": f"eval q{i}", "sentence2": f"eval a{i}"} for i in range(5)]
    eval_path.write_text("".join(json.dumps(row) + "\n" for row in eval_rows))
    trainer, _, _ = run_embedding(
        local_datasets = [run_embedding.write_csv("train.csv", _pairs("train", 40))],
        local_eval_datasets = [str(eval_path)],
        eval_steps = 0.25,
    )

    assert _rows(trainer["eval_dataset"]) == eval_rows
