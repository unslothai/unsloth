from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from core.training import portable_data as portable


class _Info:
    sha = "a" * 40


def test_pins_main_and_multiple_hub_datasets(monkeypatch):
    seen = []

    def resolve(repository, *, token = None):
        seen.append(repository)
        return repository.replace("/", "-") + "-commit"

    monkeypatch.setattr(portable, "resolve_hub_revision", resolve)
    config = {
        "hf_dataset": "org/main",
        "training_datasets": [
            {"hf_dataset": "org/one", "split": "train"},
            {"hf_dataset": "org/two", "split": "validation"},
        ],
    }
    pins = portable.pin_hub_datasets(config)
    assert seen == ["org/main", "org/one", "org/two"]
    assert config["dataset_revision"] == "org-main-commit"
    assert [entry["revision"] for entry in config["training_datasets"]] == [
        "org-one-commit",
        "org-two-commit",
    ]
    assert len(pins) == 3


def test_local_train_eval_sources_are_copied_and_changed_source_is_detectable(tmp_path):
    train = tmp_path / "train.jsonl"
    evaluate = tmp_path / "eval.jsonl"
    train.write_text('{"text":"train"}\n')
    evaluate.write_text('{"text":"eval"}\n')
    output = tmp_path / "run"
    config = {"local_datasets": [str(train)], "local_eval_datasets": [str(evaluate)]}
    records = portable.package_local_sources(config, output)
    assert {record["role"] for record in records} == {"local_datasets", "local_eval_datasets"}
    assert all(not Path(record["relative_path"]).is_absolute() for record in records)
    original_hash = records[0]["sha256"]
    train.write_text('{"text":"changed"}\n')
    assert portable.file_hash(train) != original_hash
    assert portable.file_hash(output / records[0]["relative_path"]) == original_hash


class _Dataset:
    def __init__(self, rows):
        self.rows = rows

    def save_to_disk(self, destination):
        root = Path(destination)
        root.mkdir(parents = True)
        (root / "data.json").write_text(json.dumps(self.rows))
        (root / "features.json").write_text(json.dumps({"text": "string"}))


def test_snapshot_preserves_train_eval_and_detects_hash_corruption(tmp_path):
    snapshot = portable.snapshot_processed_datasets(
        tmp_path, _Dataset(["first", "second"]), _Dataset(["eval"]), {"format": "chat"}
    )
    assert set(snapshot["splits"]) == {"train", "eval"}
    portable.verify_snapshot(tmp_path, snapshot)
    (tmp_path / snapshot["splits"]["train"] / "data.json").write_text("corrupt")
    with pytest.raises(portable.PortableDatasetError, match = "hash mismatch"):
        portable.verify_snapshot(tmp_path, snapshot)


def test_streaming_requires_explicit_materialization(tmp_path):
    with pytest.raises(portable.PortableDatasetError, match = "streaming dataset"):
        portable.snapshot_processed_datasets(tmp_path, iter([1, 2]), None, {})


def test_resume_loads_snapshot_relative_to_checkpoint_without_original_sources(
    tmp_path, monkeypatch
):
    run = tmp_path / "copied-run"
    checkpoint = run / "checkpoint-12"
    checkpoint.mkdir(parents = True)
    snapshot = portable.snapshot_processed_datasets(
        run, _Dataset(["train"]), _Dataset(["eval"]), {"format": "chat"}
    )
    loaded_paths = []

    def load_from_disk(path):
        loaded_paths.append(Path(path))
        return Path(path).name

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_from_disk = load_from_disk))
    train, evaluate = portable.load_snapshot_for_resume(checkpoint)

    assert (train, evaluate) == ("train", "eval")
    assert loaded_paths == [run / snapshot["splits"]["train"], run / snapshot["splits"]["eval"]]


def test_resume_rejects_snapshot_path_escape(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoint-1"
    bundle = run / portable.PORTABLE_DATA_DIR / "snapshot-v1" / "bundle.json"
    checkpoint.mkdir(parents = True)
    bundle.parent.mkdir(parents = True)
    bundle.write_text(
        json.dumps(
            {
                "bundle_version": portable.PORTABLE_DATA_VERSION,
                "splits": {"train": "../outside"},
            }
        )
    )
    with pytest.raises(portable.PortableDatasetError, match = "escapes the run"):
        portable.load_snapshot_for_resume(checkpoint)


def test_unavailable_hub_revision_warns_replacement_is_not_exact(monkeypatch):
    import huggingface_hub

    class Api:
        def __init__(self, token = None):
            pass

        def dataset_info(self, **kwargs):
            raise RuntimeError("gone")

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    with pytest.raises(portable.PortableDatasetError, match = "continued training, not an exact resume"):
        portable.verify_hub_revisions({"hf_dataset": "org/data", "dataset_revision": "dead"})
