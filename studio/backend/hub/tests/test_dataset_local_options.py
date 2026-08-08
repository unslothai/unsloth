# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import gzip
import json

import pytest

from hub.schemas.datasets import LocalDatasetOptionsRequest
from hub.services.datasets import local_options
from hub.utils import dataset_cache


def _write_processed_info(path, config: str, splits: list[str]) -> None:
    path.mkdir(parents = True)
    (path / "dataset_info.json").write_text(
        json.dumps(
            {
                "config_name": config,
                "splits": {name: {"name": name} for name in splits},
            }
        ),
        encoding = "utf-8",
    )
    (path / f"data-{splits[0]}.arrow").write_bytes(b"arrow")


def test_processed_cache_options_are_local_deduplicated_and_non_train_capable(
    monkeypatch, tmp_path
):
    cache_root = tmp_path / "datasets"
    processed = cache_root / "org___data"
    _write_processed_info(
        processed / "config with spaces" / "0.0.0" / "hash-a",
        "config with spaces",
        ["validation", "test", "train.clean"],
    )
    _write_processed_info(
        processed / "v1..v2" / "0.0.0" / "hash-d",
        "v1..v2",
        ["test"],
    )
    _write_processed_info(
        processed / "default" / "0.0.0" / "hash-b",
        "default",
        ["train"],
    )
    _write_processed_info(
        processed / "default" / "0.0.0" / "hash-c",
        "default",
        ["train"],
    )
    monkeypatch.setattr(dataset_cache, "hf_datasets_cache_roots", lambda: [cache_root])

    response = local_options.local_dataset_options(
        LocalDatasetOptionsRequest(dataset_name = "org/data", local_path = str(processed))
    )

    assert response.cache_available is True
    # "config with spaces" is dropped: TrainingStartRequest._check_subset rejects the space,
    # so offering it would turn the click into a 422. "v1..v2" stays, because that same
    # validator accepts it -- only split names refuse "..".
    assert [item.model_dump() for item in response.splits] == [
        {"dataset": "org/data", "config": "default", "split": "train"},
        {"dataset": "org/data", "config": "v1..v2", "split": "test"},
    ]


def test_processed_cache_options_ignore_symlinked_and_malformed_metadata(monkeypatch, tmp_path):
    cache_root = tmp_path / "datasets"
    processed = cache_root / "org___data"
    _write_processed_info(
        processed / "default" / "0.0.0" / "good",
        "default",
        ["validation"],
    )
    outside = tmp_path / "outside.json"
    outside.write_text(
        json.dumps({"config_name": "stolen", "splits": {"secret": {}}}),
        encoding = "utf-8",
    )
    linked = processed / "linked" / "0.0.0" / "hash"
    linked.mkdir(parents = True)
    (linked / "dataset_info.json").symlink_to(outside)
    (linked / "data-secret.arrow").write_bytes(b"arrow")
    malformed = processed / "broken" / "0.0.0" / "hash"
    malformed.mkdir(parents = True)
    (malformed / "dataset_info.json").write_text("{", encoding = "utf-8")
    (malformed / "data-train.arrow").write_bytes(b"arrow")
    monkeypatch.setattr(dataset_cache, "hf_datasets_cache_roots", lambda: [cache_root])

    response = local_options.local_dataset_options(
        LocalDatasetOptionsRequest(dataset_name = "org/data", local_path = str(processed))
    )

    assert [item.model_dump() for item in response.splits] == [
        {"dataset": "org/data", "config": "default", "split": "validation"}
    ]


def test_snapshot_options_merge_card_and_json_metadata_without_network(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "README.md").write_text(
        """---
configs:
- config_name: card
  data_files:
  - split: train
    path: card/train-*.parquet
- config_name: implicit-train
  data_files: card/data-*.parquet
- config_name: implicit-list
  data_files:
  - card/part-1.parquet
  - card/part-2.parquet
- config_name: ../unsafe
  data_files: card/secret.parquet
dataset_info:
- config_name: measured
  splits:
  - name: validation
---
Dataset card.
""",
        encoding = "utf-8",
    )
    (snapshot / "dataset_infos.json").write_text(
        json.dumps({"legacy": {"splits": {"test": {"name": "test"}}}}),
        encoding = "utf-8",
    )
    (snapshot / "dataset_info.json").write_text(
        json.dumps({"splits": {"holdout": 12}}),
        encoding = "utf-8",
    )

    assert [
        item.model_dump()
        for item in local_options._sorted_options(local_options._snapshot_options(snapshot))
    ] == [
        {"dataset": "", "config": "default", "split": "holdout"},
        {"dataset": "", "config": "card", "split": "train"},
        {"dataset": "", "config": "implicit-list", "split": "train"},
        {"dataset": "", "config": "implicit-train", "split": "train"},
        {"dataset": "", "config": "legacy", "split": "test"},
        {"dataset": "", "config": "measured", "split": "validation"},
    ]


def test_snapshot_options_do_not_follow_metadata_outside_cache(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    outside = tmp_path / "outside.md"
    outside.write_text(
        """---
configs:
- config_name: stolen
  data_files:
  - split: secret
    path: secret.parquet
---
""",
        encoding = "utf-8",
    )
    (snapshot / "README.md").symlink_to(outside)

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_ignore_malformed_card_yaml(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "README.md").write_text(
        "---\nconfigs: [unterminated\n---\n",
        encoding = "utf-8",
    )

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_infer_undeclared_splits_from_loadable_files(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "README.md").write_text(
        "---\nlanguage:\n- en\n---\nDataset card without split metadata.\n",
        encoding = "utf-8",
    )
    # The multi30k shape from the report: one format, so datasets can build every split.
    for name in ("train.jsonl", "test.jsonl", "val.jsonl"):
        (snapshot / name).write_text('{"text":"row"}\n', encoding = "utf-8")

    assert [
        item.model_dump()
        for item in local_options._sorted_options(local_options._snapshot_options(snapshot))
    ] == [
        {"dataset": "", "config": "default", "split": "train"},
        {"dataset": "", "config": "default", "split": "test"},
        {"dataset": "", "config": "default", "split": "validation"},
    ]


def test_snapshot_options_do_not_infer_splits_that_mix_formats(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "train.jsonl").write_text('{"text":"train"}\n', encoding = "utf-8")
    (snapshot / "test.csv").write_text("text\ntest\n", encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_infer_default_train_for_unlabelled_data(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "records.jsonl.gz").write_bytes(gzip.compress(b'{"text":"row"}\n'))

    assert local_options._snapshot_options(snapshot) == {("default", "train")}


@pytest.mark.parametrize(
    "filename",
    ["dataset_infos.json", "dataset_info.json", "config.json", "dataset_dict.json"],
)
def test_snapshot_options_do_not_infer_from_reserved_metadata(tmp_path, filename):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / filename).write_text("{}", encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_ignore_reserved_metadata_when_choosing_a_split(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / "test").mkdir(parents = True)
    (snapshot / "test" / "dataset_infos.json").write_text("{}", encoding = "utf-8")
    (snapshot / "records.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("default", "train")}


@pytest.mark.parametrize("split", ["train-clean", "my split"])
def test_snapshot_options_do_not_infer_sharded_names_datasets_rejects(tmp_path, split):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    data = snapshot / "data"
    data.mkdir(parents = True)
    (data / "train-00000-of-00002.parquet").write_bytes(b"parquet")
    (data / f"{split}-00001-of-00002.parquet").write_bytes(b"parquet")

    # datasets raises on the bad name before any split loads, so offer nothing at all.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_infer_sharded_custom_split(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    data = snapshot / "data"
    data.mkdir(parents = True)
    (data / "holdout-00000-of-00001.parquet").write_bytes(b"parquet")

    assert local_options._snapshot_options(snapshot) == {("default", "holdout")}


def test_snapshot_options_do_not_infer_from_non_data_or_unsafe_symlink(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "README.md").write_text("No metadata or data.\n", encoding = "utf-8")
    outside = tmp_path / "train.jsonl"
    outside.write_text('{"text":"outside"}\n', encoding = "utf-8")
    (snapshot / "train.jsonl").symlink_to(outside)

    assert local_options._snapshot_options(snapshot) == set()


def test_local_options_rejects_an_arbitrary_supplied_path(tmp_path):
    arbitrary = tmp_path / "org___data"
    _write_processed_info(arbitrary / "default" / "0.0.0" / "hash", "default", ["train"])

    response = local_options.local_dataset_options(
        LocalDatasetOptionsRequest(dataset_name = "org/data", local_path = str(arbitrary))
    )

    assert response.cache_available is False
    assert response.splits == []
