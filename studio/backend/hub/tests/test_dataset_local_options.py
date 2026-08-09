# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

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
    assert [item.model_dump() for item in response.splits] == [
        {"dataset": "org/data", "config": "default", "split": "train"},
        {"dataset": "org/data", "config": "config with spaces", "split": "test"},
        {
            "dataset": "org/data",
            "config": "config with spaces",
            "split": "train.clean",
        },
        {
            "dataset": "org/data",
            "config": "config with spaces",
            "split": "validation",
        },
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


def test_local_options_rejects_an_arbitrary_supplied_path(tmp_path):
    arbitrary = tmp_path / "org___data"
    _write_processed_info(arbitrary / "default" / "0.0.0" / "hash", "default", ["train"])

    response = local_options.local_dataset_options(
        LocalDatasetOptionsRequest(dataset_name = "org/data", local_path = str(arbitrary))
    )

    assert response.cache_available is False
    assert response.splits == []


def _snapshot(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    return snapshot


def _rows(snapshot, *names: str) -> None:
    for name in names:
        path = snapshot / name
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text('{"text":"row"}\n', encoding = "utf-8")


def _card(snapshot, body: str) -> None:
    snapshot.mkdir(parents = True, exist_ok = True)
    (snapshot / "README.md").write_text(f"---\n{body}---\nCard.\n", encoding = "utf-8")


def test_snapshot_options_infer_the_splits_of_a_metadata_free_cache(tmp_path):
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, "train.jsonl", "test.jsonl", "val.jsonl")

    # The shape reported in #8140: no card, three keyword-named files.
    assert local_options._snapshot_options(snapshot) == {
        ("default", "train"),
        ("default", "test"),
        ("default", "validation"),
    }


def test_snapshot_options_infer_one_train_split_from_unlabelled_files(tmp_path):
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, "records.jsonl")

    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_infer_splits_from_directory_names(tmp_path):
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, "train/a.parquet", "test/b.parquet")

    assert local_options._snapshot_options(snapshot) == {("default", "train"), ("default", "test")}


def test_snapshot_options_infer_splits_from_sharded_names(tmp_path):
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, "data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet")

    assert local_options._snapshot_options(snapshot) == {("default", "train"), ("default", "test")}


def test_snapshot_options_reject_a_shard_name_the_loader_refuses(tmp_path):
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, "data/train-clean-00000-of-00001.parquet")

    # datasets raises on the name rather than falling through to the keyword stages.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_offer_nothing_when_the_splits_disagree_on_a_format(tmp_path):
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, "train.jsonl")
    (snapshot / "test.csv").write_text("text\nrow\n", encoding = "utf-8")

    # One builder serves the whole dataset, so datasets cannot load this at all.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_treat_tsv_as_its_own_builder(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "train.csv").write_text("text\nrow\n", encoding = "utf-8")
    (snapshot / "test.tsv").write_text("text\trow\n", encoding = "utf-8")

    # tsv carries a tab separator, so datasets sees two different builders.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_ignore_the_metadata_filenames_the_loader_drops(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "dataset_infos.json").write_text("{}", encoding = "utf-8")

    # FILES_TO_IGNORE drops these by basename, so the cache holds no data at all.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_ignore_a_metadata_file_when_it_steers_a_split(tmp_path):
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, "records.jsonl")
    (snapshot / "test").mkdir()
    (snapshot / "test" / "dataset_info.json").write_text("{}", encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_match_extensions_case_sensitively(tmp_path):
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, "TRAIN.JSONL")

    # A POSIX glob never matches .JSONL, so datasets finds no data files here.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_let_a_media_file_poison_a_split(tmp_path):
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, "train.jsonl")
    (snapshot / "test.JPG").write_bytes(b"image")

    # The folder builders are registered in both cases, so this is a second builder.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_reject_a_split_with_no_builder_of_its_own(tmp_path):
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, "train.jsonl")
    (snapshot / "test").mkdir()
    (snapshot / "test" / "notes.bin").write_bytes(b"data")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_read_through_a_compression_suffix(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "records.jsonl.gz").write_bytes(b"gzipped")

    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_skip_a_codec_the_install_cannot_read(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "records.jsonl.zstd").write_bytes(b"zstd")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_keep_a_readable_suffix_after_an_unknown_one(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "records.parquet.backup").write_bytes(b"parquet")

    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_rank_folder_metadata_last(tmp_path):
    snapshot = _snapshot(tmp_path)
    for name in ("train/data.csv", "test/data.csv"):
        path = snapshot / name
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text("text\nrow\n", encoding = "utf-8")
    (snapshot / "train" / "metadata.parquet").write_bytes(b"parquet")

    assert local_options._snapshot_options(snapshot) == {("default", "train"), ("default", "test")}


def test_snapshot_options_skip_hidden_and_dunder_directories(tmp_path):
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, "records.jsonl", ".git/val.jsonl", "__pycache__/val.jsonl")

    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_refuse_a_file_symlinked_out_of_the_cache(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "test.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, "train.jsonl")
    (snapshot / "test.jsonl").symlink_to(outside / "test.jsonl")

    # datasets reads every file in the config, so one escape condemns all of it.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_read_a_blob_symlink_the_cache_owns(tmp_path):
    repo = tmp_path / "datasets--org--data"
    blobs = repo / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "abc123").write_text('{"text":"row"}\n', encoding = "utf-8")
    snapshot = repo / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "records.jsonl").symlink_to(blobs / "abc123")

    # The normal cache layout: every file is a link into blobs.
    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_leave_a_declared_card_to_its_own_configs(tmp_path):
    snapshot = _snapshot(tmp_path)
    _card(snapshot, "configs:\n- config_name: foo\n  data_dir: foo\n")
    _rows(snapshot, "foo/records.jsonl", "test.jsonl")

    # The card names the loader's configs, and inference cannot reproduce them, so the
    # picker does not invent a default one beside them.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_leave_a_standalone_yaml_card_alone(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / ".huggingface.yaml").write_text("configs:\n- config_name: foo\n", encoding = "utf-8")
    _rows(snapshot, "records.jsonl")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_offer_nothing_for_a_card_the_loader_cannot_parse(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "README.md").write_text("---\nconfigs: [\n---\n", encoding = "utf-8")
    _rows(snapshot, "train.jsonl")

    # DatasetCard.load raises on it, so nothing in the snapshot is loadable.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_still_infer_beside_a_plain_readme(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "README.md").write_text("# Just prose\n", encoding = "utf-8")
    _rows(snapshot, "records.jsonl")

    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_stay_empty_for_a_readme_only_cache(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "README.md").write_text("# Alpaca\n", encoding = "utf-8")

    # The first #8140 negative: nothing to train on, so nothing is offered.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_stay_empty_for_a_licence_only_subdirectory(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "legal").mkdir()
    (snapshot / "legal" / "LICENSE").write_text("MIT\n", encoding = "utf-8")

    # The second #8140 negative.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_skip_a_split_name_the_picker_cannot_start(tmp_path):
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, f"data/{'x' * 200}-00000-of-00001.jsonl")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_give_up_on_a_snapshot_too_large_to_compare(tmp_path, monkeypatch):
    monkeypatch.setattr(local_options, "_MAX_SNAPSHOT_DATA_FILES", 1)
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, "train.jsonl", "test.jsonl")

    # Past the cap the scan is a traversal-order prefix, which proves nothing.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_vote_on_the_files_the_loader_samples(tmp_path):
    snapshot = _snapshot(tmp_path)
    for index in range(200):
        _rows(snapshot, f"train/{index:04d}.jsonl")
    for index in range(300):
        (snapshot / "train" / f"z{index:04d}.csv").write_text("text\nrow\n", encoding = "utf-8")
    (snapshot / "test").mkdir()
    (snapshot / "test" / "a.csv").write_text("text\nrow\n", encoding = "utf-8")

    # datasets samples a split's first 200 files, so train is json and disagrees with test.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_ignore_a_training_file_a_folder_builder_drops(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "train").mkdir()
    for name in ("a.JPG", "b.JPG"):
        (snapshot / "train" / name).write_bytes(b"image")
    _rows(snapshot, "train/c.jsonl")

    # The folder builder wins the vote and then reads only the images.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_count_the_full_folder_extension_set(tmp_path):
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, "train/a.jsonl", "test/b.jsonl")
    for name in ("c.blp", "d.blp"):
        (snapshot / "test" / name).write_bytes(b"image")

    # .blp is an imagefolder extension, so test is a different builder from train.
    assert local_options._snapshot_options(snapshot) == set()


@pytest.mark.parametrize(
    "front", ["---\n---\n", "---\nnull\n---\n", "---\n# only a comment\n---\n"]
)
def test_snapshot_options_read_empty_front_matter_as_an_empty_card(tmp_path, front):
    snapshot = _snapshot(tmp_path)
    (snapshot / "README.md").write_text(front + "Prose.\n", encoding = "utf-8")
    _rows(snapshot, "records.jsonl")

    # RepoCard treats these as an empty card and carries on with file inference.
    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_reject_a_split_whose_only_file_is_empty(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "train.jsonl").write_text("", encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_keep_a_json_split_holding_one_empty_file(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "train.jsonl").write_text("", encoding = "utf-8")
    _rows(snapshot, "train2.jsonl")

    # The json builder skips an empty file as long as another still holds rows.
    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_reject_a_csv_split_holding_one_empty_file(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "train.csv").write_text("", encoding = "utf-8")
    (snapshot / "train2.csv").write_text("text\nrow\n", encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_stand_down_beside_a_card_too_large_to_read(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "README.md").write_text(
        "---\nconfigs:\n- config_name: foo\n---\n" + "x" * 3_000_000, encoding = "utf-8"
    )
    _rows(snapshot, "records.jsonl")

    # The loader reads the card and names foo, so inventing a default beside it is wrong.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_stand_down_beside_a_card_outside_the_cache(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "README.md").write_text("---\nconfigs:\n- config_name: foo\n---\n", encoding = "utf-8")
    snapshot = _snapshot(tmp_path)
    (snapshot / "README.md").symlink_to(outside / "README.md")
    _rows(snapshot, "records.jsonl")

    assert local_options._snapshot_options(snapshot) == set()
