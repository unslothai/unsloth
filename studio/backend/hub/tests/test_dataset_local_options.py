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

    # A card with no bytes declares nothing, so it blocks nothing.
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
    # datasets calls .get on each entry, so a list of scalars raises.
    # DatasetCard.load reads the card as utf-8 and raises before any data file.
    # The inner .jsonl still votes, so datasets picks json and dies on the .zst.
    # DatasetCardData updates a dict from it and raises on a scalar.
    # imagefolder wins the vote and drops the jsonl, so there is nothing to train on.
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

    # The header alone yields no row, and datasets still builds train around it.
    # resolve_pattern keeps a link only when its target is a file, so the loader never sees this one and the
    # surviving split still loads.
    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_match_extensions_case_sensitively(tmp_path):
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, "TRAIN.JSONL")

    # A POSIX glob never matches .JSONL, so datasets finds no data files here.
    # datasets treats the empty declaration as authoritative and exposes no config.
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

    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_leave_a_declared_card_to_its_own_configs(tmp_path):
    snapshot = _snapshot(tmp_path)
    _card(snapshot, "configs:\n- config_name: foo\n  data_dir: foo\n")
    _rows(snapshot, "foo/records.jsonl", "test.jsonl")

    # The card names the loader's configs, and inference cannot reproduce them, so the picker does not
    # invent a default one beside them.
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

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_stay_empty_for_a_licence_only_subdirectory(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "legal").mkdir()
    (snapshot / "legal" / "LICENSE").write_text("MIT\n", encoding = "utf-8")

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


def test_snapshot_options_reject_every_split_when_a_sibling_is_empty(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "train.csv").write_text("text\nrow\n", encoding = "utf-8")
    (snapshot / "test.csv").write_text("", encoding = "utf-8")

    # datasets prepares both splits, so the empty test file fails train as well.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_keep_a_json_split_beside_an_empty_sibling(tmp_path):
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, "train.jsonl")
    (snapshot / "test.jsonl").write_text("", encoding = "utf-8")

    # The json builder skips a file with no rows rather than failing the build.
    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_stand_down_when_standalone_yaml_declares_a_config(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / ".huggingface.yaml").write_text(
        "configs:\n- config_name: foo\n  data_files: train.jsonl\n", encoding = "utf-8"
    )
    (snapshot / "README.md").write_text("---\nlicense: mit\n---\n", encoding = "utf-8")
    _rows(snapshot, "train.jsonl")

    # A README declaring nothing must not undo the standalone YAML's declaration.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_infer_beside_a_standalone_yaml_declaring_nothing(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / ".huggingface.yaml").write_text("viewer: false\n", encoding = "utf-8")
    _rows(snapshot, "train.jsonl")

    # The loader finds no config there either, so it resolves the files by pattern.
    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_stand_down_beside_an_unreadable_standalone_yaml(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / ".huggingface.yaml").write_text("configs: [\n", encoding = "utf-8")
    _rows(snapshot, "train.jsonl")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_infer_beside_dataset_info_naming_no_split(tmp_path):
    snapshot = _snapshot(tmp_path)
    _card(snapshot, "dataset_info:\n  features:\n  - name: text\n    dtype: string\n")
    _rows(snapshot, "train.jsonl")

    # A feature schema alone names no config, so the loader still infers by pattern.
    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_reject_a_split_holding_an_undecompressible_file(tmp_path):
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, "train.jsonl")
    (snapshot / "train2.jsonl.zst").write_bytes(b"\x28\xb5\x2f\xfd\x00\x00")

    # datasets keeps the .zst for the json builder and dies on the missing codec.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_infer_beside_standalone_yaml_dataset_info(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / ".huggingface.yaml").write_text(
        "dataset_info:\n  features:\n  - name: text\n    dtype: string\n", encoding = "utf-8"
    )
    _rows(snapshot, "train.jsonl")

    # 4.3.0 builds no config from dataset_info declared there, so it infers the files.
    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_stand_down_beside_unparsable_legacy_metadata(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "dataset_infos.json").write_text("{not json", encoding = "utf-8")
    _rows(snapshot, "train.jsonl")

    # datasets json.loads it while resolving configs and raises before any split exists.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_infer_beside_an_empty_readme(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "README.md").write_text("", encoding = "utf-8")
    _rows(snapshot, "train.jsonl")

    # A card with no bytes declares nothing, so it blocks nothing.
    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_count_bw_images_as_a_folder_builder(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "a.bw").write_bytes(b"\x89PNG\r\n\x1a\n")
    (snapshot / "b.bw").write_bytes(b"\x89PNG\r\n\x1a\n")
    _rows(snapshot, "notes.jsonl")

    # imagefolder wins the vote and drops the jsonl, so there is nothing to train on.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_stand_down_beside_a_non_mapping_standalone_yaml(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / ".huggingface.yaml").write_text("hello\n", encoding = "utf-8")
    _rows(snapshot, "train.jsonl")

    # DatasetCardData updates a dict from it and raises on a scalar.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_infer_beside_an_empty_standalone_yaml(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / ".huggingface.yaml").write_text("", encoding = "utf-8")
    _rows(snapshot, "train.jsonl")

    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_reject_a_split_whose_module_needs_a_missing_codec(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "train.csv").write_text("text\nrow\n", encoding = "utf-8")
    (snapshot / "train2.jsonl.zst").write_bytes(b"\x28\xb5\x2f\xfd\x00\x00")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_stand_down_beside_an_empty_legacy_metadata_file(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "dataset_infos.json").write_text("", encoding = "utf-8")
    _rows(snapshot, "train.jsonl")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_stand_down_beside_a_card_that_cannot_be_decoded(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "README.md").write_bytes(b"\xff\xfe---\nconfigs: x\n---\n")
    _rows(snapshot, "train.jsonl")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_stand_down_beside_a_malformed_dataset_info_list(tmp_path):
    snapshot = _snapshot(tmp_path)
    _card(snapshot, "dataset_info: [foo]\n")
    _rows(snapshot, "train.jsonl")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_ignore_a_dangling_link_beside_a_good_file(tmp_path):
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, "train.jsonl")
    (snapshot / "train-missing.jsonl").symlink_to("../../blobs/gone")

    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_drop_a_header_only_csv_split(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "train.csv").write_text("text\nrow\n", encoding = "utf-8")
    (snapshot / "test.csv").write_text("text\n", encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_offer_nothing_when_every_csv_is_header_only(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "train.csv").write_text("text\n", encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_stand_down_beside_an_empty_split_declaration(tmp_path):
    snapshot = _snapshot(tmp_path)
    _card(snapshot, "dataset_info:\n  splits: {}\n")
    _rows(snapshot, "train.jsonl")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_skip_file_checks_for_an_untrainable_builder(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path)
    for index in range(8):
        (snapshot / f"img{index}.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    def _fail(*args, **kwargs):
        raise AssertionError("resolved every file for a builder that trains nothing")

    monkeypatch.setattr(local_options, "_offerable", _fail)

    # imagefolder wins, so no file can be offered and none needs opening to say so.
    assert local_options._inferred_snapshot_options(snapshot) == set()


def test_snapshot_options_keep_a_compressed_csv_split(tmp_path):
    import gzip

    snapshot = _snapshot(tmp_path)
    (snapshot / "train.csv.gz").write_bytes(gzip.compress(b"text\nrow\n"))

    # `[]` parses fine and yields no row, so datasets builds train around it.
    # Compressed bytes say nothing about the rows inside, so the split still stands.
    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_drop_a_split_holding_an_empty_json_container(tmp_path):
    snapshot = _snapshot(tmp_path)
    _rows(snapshot, "train.jsonl")
    (snapshot / "test.json").write_text("[]", encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_reject_a_legacy_lzma_stream(tmp_path):
    import lzma

    snapshot = _snapshot(tmp_path)
    (snapshot / "train.jsonl.lzma").write_bytes(
        lzma.compress(b'{"text":"row"}\n', format = lzma.FORMAT_ALONE)
    )

    # datasets registers .xz for its filter, not the alone-format .lzma.
    assert local_options._snapshot_options(snapshot) == set()
