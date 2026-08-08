# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import gzip
import json
import os

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
dataset_info:
- config_name: measured
  splits:
  - name: validation
---
Dataset card.
""",
        encoding = "utf-8",
    )
    (snapshot / "card").mkdir()
    for name in ("train-0.parquet", "data-0.parquet", "part-1.parquet", "part-2.parquet"):
        (snapshot / "card" / name).write_bytes(b"parquet")
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


def test_snapshot_options_reject_a_config_name_the_builder_refuses(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "README.md").write_text(
        "---\nconfigs:\n- config_name: ../unsafe\n  data_files: a.jsonl\n"
        "- config_name: good\n  data_files: b.jsonl\n---\n",
        encoding = "utf-8",
    )
    (snapshot / "a.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    (snapshot / "b.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # datasets raises InvalidConfigName on the blacklisted characters, so the good sibling
    # never gets built either and skipping the bad one would advertise a dead option.
    assert local_options._snapshot_options(snapshot) == set()


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


def test_snapshot_options_suppress_inference_after_a_card_parse_failure(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "README.md").write_text("---\nconfigs: [unterminated\n---\n", encoding = "utf-8")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # datasets raises the YAML error out of DatasetCard.load, so nothing here is loadable.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_read_the_standalone_yaml_over_the_card(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / ".huggingface.yaml").write_text(
        "configs:\n- config_name: foo\n  data_files:\n  - split: test\n    path: records.jsonl\n",
        encoding = "utf-8",
    )
    (snapshot / "records.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("foo", "test")}


def test_snapshot_options_keep_a_declared_config_name_when_inferring(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "README.md").write_text(
        "---\nconfigs:\n- config_name: foo\n---\ncard\n", encoding = "utf-8"
    )
    (snapshot / "records.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # datasets still builds config foo over the inferred patterns, so default would 422.
    assert local_options._snapshot_options(snapshot) == {("foo", "train")}


def test_snapshot_options_read_a_list_valued_declared_path(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / "a").mkdir(parents = True)
    (snapshot / "b").mkdir(parents = True)
    (snapshot / "README.md").write_text(
        "---\nconfigs:\n- config_name: a\n  data_files:\n  - split: train\n    path:\n"
        "    - a/train.jsonl\n- config_name: b\n  data_dir: b\n---\n",
        encoding = "utf-8",
    )
    (snapshot / "a" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    (snapshot / "b" / "train.csv").write_text("text\nrow\n", encoding = "utf-8")

    # The list form still fixes the dataset's module, so the csv config cannot be offered.
    assert local_options._snapshot_options(snapshot) == {("a", "train")}


def test_snapshot_options_infer_a_config_whose_data_files_is_null(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / "a").mkdir(parents = True)
    (snapshot / "b").mkdir(parents = True)
    (snapshot / "README.md").write_text(
        "---\nconfigs:\n- config_name: a\n  data_files: a/train.jsonl\n"
        "- config_name: b\n  data_files: null\n  data_dir: b\n---\n",
        encoding = "utf-8",
    )
    (snapshot / "a" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    (snapshot / "b" / "test.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # datasets treats a null value as no data_files and infers that config's patterns.
    assert local_options._snapshot_options(snapshot) == {("a", "train"), ("b", "test")}


def test_snapshot_options_reject_a_data_dir_with_an_embedded_null(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "README.md").write_text(
        '---\nconfigs:\n- config_name: cfg\n  data_dir: "foo\\0bar"\n---\n', encoding = "utf-8"
    )
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # Path.resolve raises on the null, and the endpoint has to answer, not fail.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_reject_an_oversized_standalone_yaml(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / ".huggingface.yaml").write_text(
        "configs:\n- config_name: foo\n#" + "x" * (2 * 1024 * 1024), encoding = "utf-8"
    )
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_reject_a_split_whose_archive_escapes_the_cache(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / "train").mkdir(parents = True)
    (snapshot / "train" / "safe.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    outside = tmp_path / "outside.zip"
    outside.write_bytes(b"zip")
    (snapshot / "train" / "data.zip").symlink_to(outside)

    # datasets keeps zips for every module, so the split's builder would open this one.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_do_not_infer_beside_an_unreadable_declared_pattern(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / "a").mkdir(parents = True)
    (snapshot / "b").mkdir(parents = True)
    (snapshot / "README.md").write_text(
        "---\nconfigs:\n- config_name: a\n  data_files: a/*\n- config_name: b\n  data_dir: b\n---\n",
        encoding = "utf-8",
    )
    (snapshot / "a" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    (snapshot / "b" / "train.csv").write_text("text\nrow\n", encoding = "utf-8")

    # `a/*` names no extension, so the module it settles is unknown and b cannot be claimed.
    assert local_options._snapshot_options(snapshot) == {("a", "train")}


def test_snapshot_options_reject_a_scalar_standalone_yaml(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / ".huggingface.yaml").write_text("metadata\n", encoding = "utf-8")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # The loader hands this to dict.update, which raises.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_reject_a_card_too_large_to_read(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "README.md").write_text(
        "---\nconfigs:\n- config_name: foo\n---\n" + "x" * (2 * 1024 * 1024),
        encoding = "utf-8",
    )
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # datasets parses the card whatever its size, so calling it absent invents a default.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_reject_a_card_with_two_default_configs(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "README.md").write_text(
        "---\nconfigs:\n- config_name: a\n  default: true\n"
        "- config_name: b\n  default: true\n---\n",
        encoding = "utf-8",
    )
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # get_default_config_name raises on several defaults before anything is built.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_ignore_a_malformed_singular_dataset_info(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "dataset_info.json").write_text("{oops", encoding = "utf-8")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # Only the plural legacy file is opened by the local factory; this one it just ignores.
    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_reject_a_shard_name_needing_a_trim(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / "data").mkdir(parents = True)
    (snapshot / "data" / " train-00000-of-00001.parquet").write_bytes(b"parquet")

    # datasets validates the captured name as it stands and raises on the space.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_treat_empty_front_matter_as_no_metadata(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "README.md").write_text("---\n---\n", encoding = "utf-8")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # DatasetCard.load turns a null block into empty metadata rather than raising.
    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_keyword_split_files_place_each_synonym_at_its_own_position(tmp_path):
    files = [
        (local_options.PurePosixPath("a_train_training.csv"), "csv"),
        (local_options.PurePosixPath("b_training.jsonl"), "json"),
    ]

    # train resolves before training, so the csv's train copy leads and its training copy
    # sits beside the other training file rather than next to itself.
    grouped = local_options._keyword_split_files(files, local_options._FILENAME_SPLITS)
    assert [str(path) for path, _module in grouped["train"]] == [
        "a_train_training.csv",
        "a_train_training.csv",
        "b_training.jsonl",
    ]


def _card(snapshot, body: str) -> None:
    snapshot.mkdir(parents = True, exist_ok = True)
    (snapshot / "README.md").write_text(f"---\n{body}---\n", encoding = "utf-8")


def test_snapshot_options_share_one_module_across_configs(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / "a").mkdir(parents = True)
    (snapshot / "b").mkdir(parents = True)
    _card(
        snapshot,
        "configs:\n- config_name: a\n  data_files: a/train.jsonl\n"
        "- config_name: b\n  data_dir: b\n",
    )
    (snapshot / "a" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    (snapshot / "b" / "train.csv").write_text("text\nrow\n", encoding = "utf-8")

    # datasets settles on one module before it builds any config, so b is read as json.
    assert local_options._snapshot_options(snapshot) == {("a", "train")}


def test_snapshot_options_do_not_infer_for_an_explicitly_empty_data_files(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, "configs:\n- config_name: foo\n  data_files: []\n")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # Declared and empty resolves to nothing, which is not the same as never declared.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_treat_an_empty_configs_list_as_undeclared(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, "configs: []\n")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_drop_a_config_whose_data_dir_is_absolute(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / "data").mkdir(parents = True)
    _card(snapshot, "configs:\n- config_name: cfg\n  data_dir: /data\n")
    (snapshot / "data" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # datasets joins that onto the base path, where the absolute part wins and leaves the
    # snapshot entirely, so scanning snapshot/data would advertise something else's split.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_offer_a_compound_suffix_file(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "records.txt.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # .jsonl beats .txt in the tie-break, and membership has to agree with that.
    assert local_options._snapshot_options(snapshot) == {("default", "train")}


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        (".gz", {("default", "train")}),
        (".gzip", {("default", "train")}),
        (".bz2", {("default", "train")}),
        (".xz", {("default", "train")}),
        (".lzma", {("default", "train")}),
        # datasets names these but the codecs are not shipped, so they raise on read.
        (".zst", set()),
        (".zstd", set()),
        (".lz4", set()),
    ],
)
def test_snapshot_options_accept_only_decompressible_suffixes(tmp_path, suffix, expected):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / f"records.jsonl{suffix}").write_bytes(b"compressed")

    assert local_options._snapshot_options(snapshot) == expected


def test_snapshot_options_reject_non_mapping_card_metadata(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "README.md").write_text("---\nmetadata\n---\n", encoding = "utf-8")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # DatasetCard.load requires a mapping and raises on anything else.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_reject_an_unreadable_dataset_infos(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "dataset_infos.json").write_text("{oops", encoding = "utf-8")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # datasets opens the file whenever it exists and lets the decode error escape.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_scan_each_data_dir_once(tmp_path, monkeypatch):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    _card(snapshot, "configs:\n" + "".join(f"- config_name: c{index}\n" for index in range(50)))
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    scans = []
    original = local_options._snapshot_data_files
    monkeypatch.setattr(
        local_options,
        "_snapshot_data_files",
        lambda path: (scans.append(path), original(path))[1],
    )

    assert len(local_options._snapshot_options(snapshot)) == 50
    # A card may name thousands of configs; they all share the one root here.
    assert len(scans) == 1


def test_keyword_split_files_follow_the_loaders_keyword_order(tmp_path):
    files = [
        (local_options.PurePosixPath("dev/a.csv"), "csv"),
        (local_options.PurePosixPath("validation/a.jsonl"), "json"),
    ]

    # validation resolves before dev, so its files come first in the sampled window.
    grouped = local_options._keyword_split_files(files, local_options._DIR_NAME_SPLITS)
    assert [str(path) for path, _module in grouped["validation"]] == [
        "validation/a.jsonl",
        "dev/a.csv",
    ]


def test_snapshot_options_reject_a_card_datasets_cannot_build_configs_from(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "README.md").write_text("---\nconfigs: nope\n---\n", encoding = "utf-8")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # MetadataConfigs raises on this long before datasets looks at a file.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_infer_per_config_beside_an_explicit_one(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / "foo").mkdir(parents = True)
    (snapshot / "bar").mkdir(parents = True)
    (snapshot / "README.md").write_text(
        "---\nconfigs:\n- config_name: foo\n  data_files: foo/train.jsonl\n"
        "- config_name: bar\n  data_dir: bar\n---\n",
        encoding = "utf-8",
    )
    (snapshot / "foo" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    (snapshot / "bar" / "test.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("foo", "train"), ("bar", "test")}


def test_snapshot_options_take_the_last_of_duplicate_config_names(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / "foo" / "a").mkdir(parents = True)
    (snapshot / "foo" / "b").mkdir(parents = True)
    (snapshot / "README.md").write_text(
        "---\nconfigs:\n- config_name: foo\n  data_dir: foo/a\n"
        "- config_name: foo\n  data_dir: foo/b\n---\n",
        encoding = "utf-8",
    )
    (snapshot / "foo" / "a" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    (snapshot / "foo" / "b" / "test.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # datasets keys metadata configs by name, so the later entry replaces the earlier one.
    assert local_options._snapshot_options(snapshot) == {("foo", "test")}


def test_snapshot_options_drop_a_config_whose_data_dir_is_unsafe(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / "bar").mkdir(parents = True)
    (snapshot / "README.md").write_text(
        "---\nconfigs:\n- config_name: cfg\n  data_dir: foo/../bar\n---\n", encoding = "utf-8"
    )
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    (snapshot / "bar" / "test.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # Widening the config to the whole snapshot would offer a split it never scoped.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_reject_a_split_holding_an_external_symlink(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / "train").mkdir(parents = True)
    (snapshot / "train" / "safe.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"text":"outside"}\n', encoding = "utf-8")
    (snapshot / "train" / "leak.jsonl").symlink_to(outside)

    # datasets would read both files, so one safe file does not make the split safe.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_count_a_file_once_per_matching_keyword(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "train_training.csv").write_text("text\nrow\n", encoding = "utf-8")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    (snapshot / "test.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # datasets resolves train's keyword patterns separately, so the csv lands in train twice
    # and wins it, leaving train on csv and test on json.
    assert local_options._snapshot_options(snapshot) == set()


@pytest.mark.parametrize(
    "files",
    [
        {"train.txt.csv": "row\n", "test.txt": "row\n"},
        {
            "train/a.geoparquet": "x",
            "train/b.gpq": "x",
            "train/c.jsonl": "{}\n",
            "train/d.jsonl": "{}\n",
            "test/a.parquet": "x",
        },
        {
            "train/a.txt": "row\n",
            "train/b.xml": "<r/>",
            "test/a.txt": "row\n",
            "test/b.jsonl": "{}\n",
        },
    ],
)
def test_snapshot_options_follow_the_loaders_extension_counting(tmp_path, files):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    for name, body in files.items():
        path = snapshot / name
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(body, encoding = "utf-8")

    # Every suffix counts under its own name, so these all end on mismatched split modules.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_keep_media_builders_apart(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    for split, media in (("train", "jpg"), ("test", "mp3")):
        (snapshot / split).mkdir(parents = True)
        for index in range(2):
            (snapshot / split / f"{index}.{media}").write_bytes(b"media")
        (snapshot / split / "rows.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # imagefolder and audiofolder are different builders, so datasets refuses the snapshot.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_infer_under_a_declared_data_dir(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / "foo").mkdir(parents = True)
    (snapshot / "README.md").write_text(
        "---\nconfigs:\n- config_name: foo\n  data_dir: foo\n---\ncard\n", encoding = "utf-8"
    )
    (snapshot / "foo" / "records.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    (snapshot / "test.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # datasets runs inference under foo/ for that config, so the root test file is not its split.
    assert local_options._snapshot_options(snapshot) == {("foo", "train")}


@pytest.mark.parametrize("other", ["test.txt", "test.JPG", "test/notes.bin"])
def test_snapshot_options_do_not_infer_when_another_split_cannot_build(tmp_path, other):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    path = snapshot / other
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text("row\n", encoding = "utf-8")

    # datasets picks its split patterns over every file, then fails on the split it cannot
    # build or on the module mismatch, so the advertised train would not have loaded either.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_do_not_infer_across_csv_and_tsv_splits(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "train.csv").write_text("text\nrow\n", encoding = "utf-8")
    (snapshot / "test.tsv").write_text("text\nrow\n", encoding = "utf-8")

    # Both build with csv, but tsv carries sep="\t" and datasets compares the whole result.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_do_not_infer_when_an_archive_decides_a_split(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    for index in range(2):
        (snapshot / f"train{index}.zip").write_bytes(b"zip")

    # datasets reads the archives to pick the module. We do not, so we cannot claim a match.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_suppressed_when_the_scan_is_truncated(tmp_path, monkeypatch):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    for index in range(4):
        (snapshot / f"records{index}.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    monkeypatch.setattr(local_options, "_MAX_SNAPSHOT_DATA_FILES", 2)

    # A truncated scan cannot see the split datasets would, so it offers nothing at all.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_do_not_infer_when_a_loader_only_file_forms_a_split(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "train.parquet").write_bytes(b"parquet")
    (snapshot / "test.txt").write_text("row\n", encoding = "utf-8")

    # datasets loads .txt too, so its splits infer parquet and text and none of them build.
    assert local_options._snapshot_options(snapshot) == set()


def test_split_module_samples_only_the_first_files_datasets_would(tmp_path):
    csv = [(local_options.PurePosixPath(f"train/{i:04d}.csv"), "csv") for i in range(200)]
    parquet = [
        (local_options.PurePosixPath(f"train/{i:04d}.parquet"), "parquet")
        for i in range(1000, 1201)
    ]

    assert local_options._split_module(csv + parquet) == "csv"


def test_split_module_ranks_folder_metadata_last(tmp_path):
    entries = [
        (local_options.PurePosixPath("train/data.csv"), "csv"),
        (local_options.PurePosixPath("train/metadata.parquet"), "parquet"),
    ]

    assert local_options._split_module(entries) == "csv"


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


def test_snapshot_options_infer_from_blob_backed_symlinks(tmp_path):
    # The real cache shape on Linux and macOS: snapshot entries link into ../../blobs.
    # Windows and HF_HUB_DISABLE_SYMLINKS write plain files, which the tests above cover.
    repo = tmp_path / "datasets--org--data"
    snapshot = repo / "snapshots" / "commit"
    blobs = repo / "blobs"
    snapshot.mkdir(parents = True)
    blobs.mkdir(parents = True)
    for index, name in enumerate(("train.jsonl", "val.jsonl", "test.jsonl")):
        blob = blobs / f"blob{index}"
        blob.write_text('{"text":"row"}\n', encoding = "utf-8")
        (snapshot / name).symlink_to(os.path.relpath(blob, snapshot))

    assert local_options._snapshot_options(snapshot) == {
        ("default", "train"),
        ("default", "validation"),
        ("default", "test"),
    }


def test_snapshot_options_infer_sharded_unicode_split(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    data = snapshot / "data"
    data.mkdir(parents = True)
    (data / "café-00000-of-00001.parquet").write_bytes(b"parquet")

    assert local_options._snapshot_options(snapshot) == {("default", "café")}


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("val.jsonl", {("default", "validation")}),
        # datasets globs through fsspec, a plain regex with no normcase, so an uppercase
        # keyword is not a keyword and an uppercase extension is not supported data.
        ("VAL.jsonl", {("default", "train")}),
        ("train.JSONL", set()),
    ],
)
def test_snapshot_options_infer_case_sensitively(tmp_path, filename, expected):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / filename).write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == expected


def test_snapshot_options_infer_from_dunder_file_but_not_dunder_directory(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / "__pycache__").mkdir(parents = True)
    (snapshot / "__pycache__" / "test.jsonl").write_text("{}\n", encoding = "utf-8")
    (snapshot / "__val.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("default", "validation")}


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


def test_snapshot_options_reject_a_named_default_beside_a_flagged_one(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(
        snapshot,
        "configs:\n- config_name: default\n  data_files: a.jsonl\n- config_name: other\n  default: true\n  data_files: b.jsonl\n",
    )
    (snapshot / "a.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    (snapshot / "b.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_keep_a_named_default_without_a_flagged_one(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(
        snapshot,
        "configs:\n- config_name: default\n  data_files: a.jsonl\n- config_name: other\n  data_files: b.jsonl\n",
    )
    (snapshot / "a.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    (snapshot / "b.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("default", "train"), ("other", "train")}


def test_snapshot_options_reject_an_empty_declared_data_files(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(
        snapshot, "configs:\n- config_name: a\n  data_files: []\n- config_name: b\n  data_dir: b\n"
    )
    (snapshot / "b").mkdir()
    (snapshot / "b" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_reject_an_unreadable_standalone_yaml(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / ".huggingface.yaml").write_bytes(b"\xff\xfe configs: x\n")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_reject_an_unreadable_plural_dataset_infos(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "dataset_infos.json").write_text("", encoding = "utf-8")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_count_a_repeated_keyword_once_per_pattern(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    # train_train.csv matches both filename patterns, so csv outvotes the single jsonl and
    # the two splits disagree on the module.
    (snapshot / "train_train.csv").write_text("text\nrow\n", encoding = "utf-8")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    (snapshot / "test.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_reject_a_config_name_the_loader_keeps_padded(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, 'configs:\n- config_name: " foo "\n  data_dir: d\n')
    (snapshot / "d").mkdir()
    (snapshot / "d" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_reject_a_config_whose_other_split_escapes_the_cache(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    outside = tmp_path / "test.jsonl"
    outside.write_text('{"text":"outside"}\n', encoding = "utf-8")
    (snapshot / "test.jsonl").symlink_to(outside)

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_read_a_data_dir_that_merely_contains_dots(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, "configs:\n- config_name: cfg\n  data_dir: release..v2\n")
    (snapshot / "release..v2").mkdir()
    (snapshot / "release..v2" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("cfg", "train")}


def test_snapshot_options_collapse_a_repeated_config_name_before_counting_defaults(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(
        snapshot,
        "configs:\n- config_name: default\n  data_files: a.jsonl\n- config_name: default\n  data_files: b.jsonl\n",
    )
    (snapshot / "a.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    (snapshot / "b.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_reject_a_null_data_files_on_the_first_config(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(
        snapshot,
        "configs:\n- config_name: a\n  data_files: null\n  data_dir: a\n- config_name: b\n  data_files: b/train.jsonl\n",
    )
    for name in ("a", "b"):
        (snapshot / name).mkdir()
        (snapshot / name / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_validate_config_entries_past_the_option_cap(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    entries = "".join(
        f"- config_name: c{index}\n  data_files: train.jsonl\n"
        for index in range(local_options._MAX_OPTIONS + 1)
    )
    _card(snapshot, "configs:\n" + entries + "- config_name: bad/name\n  data_files: train.jsonl\n")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_ignore_front_matter_the_card_parser_will_not_take(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "README.md").write_text(
        "---   \nconfigs:\n- config_name: cfg\n  data_files: train.jsonl\n---\n", encoding = "utf-8"
    )
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_ignore_front_matter_whose_closing_delimiter_is_padded(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "README.md").write_text(
        "---\nconfigs:\n- config_name: cfg\n  data_files: train.jsonl\n---   \n", encoding = "utf-8"
    )
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # The pinned huggingface_hub 0.36.2 does not allow padding on either delimiter.
    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_do_not_infer_past_an_unsafe_card_symlink(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    outside = tmp_path / "README.md"
    outside.write_text(
        "---\nconfigs:\n- config_name: outside\n  data_files: train.jsonl\n---\n", encoding = "utf-8"
    )
    (snapshot / "README.md").symlink_to(outside)

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_infer_past_an_empty_standalone_yaml(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / ".huggingface.yaml").write_text("[]\n", encoding = "utf-8")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # The loader only merges a truthy block, so a falsy one is not the scalar that raises.
    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_reject_a_scalar_standalone_yaml(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / ".huggingface.yaml").write_text("- a\n", encoding = "utf-8")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_ignore_folder_metadata_for_a_non_folder_module(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / "train").mkdir(parents = True)
    (snapshot / "train" / "data.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    outside = tmp_path / "metadata.csv"
    outside.write_text("text\nrow\n", encoding = "utf-8")
    (snapshot / "train" / "metadata.csv").symlink_to(outside)

    # The json builder's metadata allow-list is empty, so it never reads that file.
    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_reject_a_declared_data_files_shape_the_loader_refuses(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(
        snapshot,
        "configs:\n- config_name: a\n  data_files:\n  - path: a/train.jsonl\n- config_name: b\n  data_dir: b\n",
    )
    for name in ("a", "b"):
        (snapshot / name).mkdir()
        (snapshot / name / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_take_the_module_from_the_collapsed_first_config(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(
        snapshot,
        "configs:\n- config_name: foo\n  data_files: foo/train.csv\n- config_name: foo\n  data_files: foo/train.jsonl\n- config_name: bar\n  data_dir: bar\n",
    )
    (snapshot / "foo").mkdir()
    (snapshot / "foo" / "train.csv").write_text("text\nrow\n", encoding = "utf-8")
    (snapshot / "foo" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    (snapshot / "bar").mkdir()
    (snapshot / "bar" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("foo", "train"), ("bar", "train")}


def test_snapshot_options_do_not_let_a_later_config_pick_the_module(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(
        snapshot,
        "configs:\n- config_name: a\n  data_dir: a\n- config_name: b\n  data_files: b/train.csv\n",
    )
    (snapshot / "a").mkdir()
    (snapshot / "a" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    (snapshot / "b").mkdir()
    (snapshot / "b" / "train.csv").write_text("text\nrow\n", encoding = "utf-8")

    # The snapshot's own patterns settle on json, so a is offered and the csv config is not.
    assert local_options._snapshot_options(snapshot) == {("a", "train")}


def test_snapshot_options_reject_a_card_that_is_not_utf_8(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "README.md").write_bytes(b"\xff\xfe---\nconfigs: x\n---\n")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_data_files_come_back_in_resolved_order(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / "b").mkdir(parents = True)
    (snapshot / "a").mkdir()
    for name in ("b/z.jsonl", "a/y.jsonl", "b/a.jsonl"):
        (snapshot / name).write_text('{"text":"row"}\n', encoding = "utf-8")

    files = local_options._snapshot_data_files(snapshot)
    paths = [path.as_posix() for path, _module in files]
    assert paths == sorted(paths)


@pytest.mark.parametrize("value", ["{}", "false", '""'])
def test_snapshot_options_infer_past_a_falsy_configs_field(tmp_path, value):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, f"configs: {value}\n")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_reject_features_the_loader_cannot_parse(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, "configs:\n- config_name: cfg\n  data_dir: d\n  features: oops\n")
    (snapshot / "d").mkdir()
    (snapshot / "d" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


@pytest.mark.parametrize("payload", ["[]", '"x"', '{"default": 1}'])
def test_snapshot_options_reject_a_legacy_dataset_infos_that_is_not_a_mapping(tmp_path, payload):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "dataset_infos.json").write_text(payload, encoding = "utf-8")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_reject_a_first_config_whose_file_is_missing(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(
        snapshot,
        "configs:\n- config_name: a\n  data_files: missing.jsonl\n- config_name: b\n  data_dir: b\n",
    )
    (snapshot / "b").mkdir()
    (snapshot / "b" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_reject_a_data_dir_the_loader_would_not_normalise(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, "configs:\n- config_name: cfg\n  data_dir: b//\n")
    (snapshot / "b").mkdir()
    (snapshot / "b" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_reject_a_standalone_yaml_that_is_a_directory(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / ".huggingface.yaml").mkdir(parents = True)
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


@pytest.mark.parametrize("name", ["foo-00000-of-00001.", "-00000-of-00001.jsonl"])
def test_snapshot_options_let_an_empty_sharded_wildcard_take_the_stage(tmp_path, name):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / "data").mkdir(parents = True)
    (snapshot / "data" / name).write_text('{"text":"row"}\n', encoding = "utf-8")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # The loader's * matches the empty component, so the sharded stage wins and then fails.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_infer_past_an_empty_standalone_yaml_file(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / ".huggingface.yaml").write_text("", encoding = "utf-8")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("default", "train")}


@pytest.mark.parametrize("data_dir", ["./a", "a/.", "a/", ".//a"])
def test_snapshot_options_read_a_data_dir_the_loader_still_finds(tmp_path, data_dir):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, f'configs:\n- config_name: cfg\n  data_dir: "{data_dir}"\n')
    (snapshot / "a").mkdir()
    (snapshot / "a" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("cfg", "train")}


@pytest.mark.parametrize(
    ("declared", "path"),
    [(".hidden.jsonl", ".hidden.jsonl"), ("__special__/train.jsonl", "__special__/train.jsonl")],
)
def test_snapshot_options_resolve_a_declared_hidden_path(tmp_path, declared, path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, f"configs:\n- config_name: cfg\n  data_files: {declared}\n")
    target = snapshot / path
    target.parent.mkdir(parents = True, exist_ok = True)
    target.write_text('{"text":"row"}\n', encoding = "utf-8")

    # Default inference skips these, but a pattern naming them explicitly resolves.
    assert local_options._snapshot_options(snapshot) == {("cfg", "train")}


def test_snapshot_options_reject_a_split_declared_twice(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(
        snapshot,
        "configs:\n- config_name: a\n  data_files:\n  - split: train\n    path: a/one.jsonl\n  - split: train\n    path: a/two.jsonl\n- config_name: b\n  data_dir: b\n",
    )
    (snapshot / "a").mkdir(parents = True)
    for name in ("one.jsonl", "two.jsonl"):
        (snapshot / "a" / name).write_text('{"text":"row"}\n', encoding = "utf-8")
    (snapshot / "b").mkdir()
    (snapshot / "b" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_reject_a_first_config_whose_wildcard_matches_nothing(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(
        snapshot,
        "configs:\n- config_name: a\n  data_files: missing*.jsonl\n- config_name: b\n  data_dir: b\n",
    )
    (snapshot / "b").mkdir(parents = True)
    (snapshot / "b" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_reject_a_dataset_info_the_loader_cannot_walk(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, "dataset_info: [oops]\n")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_weigh_a_declared_module_by_the_files_it_resolves(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(
        snapshot,
        "configs:\n- config_name: a\n  data_files:\n  - a/*.jsonl\n  - a/*.csv\n- config_name: b\n  data_dir: b\n",
    )
    (snapshot / "a").mkdir(parents = True)
    (snapshot / "a" / "one.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    for name in ("one.csv", "two.csv"):
        (snapshot / "a" / name).write_text("text\nrow\n", encoding = "utf-8")
    (snapshot / "b").mkdir()
    (snapshot / "b" / "train.csv").write_text("text\nrow\n", encoding = "utf-8")

    # Two csv files outvote one jsonl, so csv is settled and the csv sibling is offered.
    # Config a keeps a jsonl the csv builder chokes on, so it is not.
    assert local_options._snapshot_options(snapshot) == {("b", "train")}


def test_snapshot_options_reject_a_card_with_a_non_string_key(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, "1: x\n")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


@pytest.mark.parametrize("path", ["train.jsonl", "sub/train.jsonl"])
def test_snapshot_options_match_a_recursive_glob_at_any_depth(tmp_path, path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, "configs:\n- config_name: cfg\n  data_files: '**/train.jsonl'\n")
    target = snapshot / path
    target.parent.mkdir(parents = True, exist_ok = True)
    target.write_text('{"text":"row"}\n', encoding = "utf-8")

    # fsspec lets **/ stand for no directory at all.
    assert local_options._snapshot_options(snapshot) == {("cfg", "train")}


def test_snapshot_options_read_a_declared_path_that_starts_with_a_dot(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, "configs:\n- config_name: cfg\n  data_files: ./train.jsonl\n")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("cfg", "train")}


def test_snapshot_options_reject_a_declared_path_list_holding_a_non_string(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(
        snapshot,
        "configs:\n- config_name: cfg\n  data_files:\n  - split: train\n    path:\n    - a.jsonl\n    - 123\n",
    )
    (snapshot / "a.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_keep_a_config_whose_wildcard_alternative_misses(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(
        snapshot, "configs:\n- config_name: a\n  data_files:\n  - a.jsonl\n  - 'missing-*.jsonl'\n"
    )
    (snapshot / "a.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # The loader swallows FileNotFoundError for a pattern with magic in it.
    assert local_options._snapshot_options(snapshot) == {("a", "train")}


def test_snapshot_options_reject_a_config_whose_literal_alternative_misses(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, "configs:\n- config_name: a\n  data_files:\n  - a.jsonl\n  - nope.jsonl\n")
    (snapshot / "a.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_all_files_come_back_in_resolved_order(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / "z").mkdir(parents = True)
    (snapshot / "a").mkdir()
    for name in ("z/b.csv", "a/y.jsonl", ".hidden.jsonl"):
        (snapshot / name).write_text('{"text":"row"}\n', encoding = "utf-8")

    paths = [path.as_posix() for path in local_options._snapshot_all_files(snapshot)]
    assert paths == sorted(paths)
    assert ".hidden.jsonl" in paths


def test_snapshot_options_read_a_glob_class_with_a_literal_caret(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, "configs:\n- config_name: cfg\n  data_files: '[^b]rain.jsonl'\n")
    (snapshot / "brain.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # Only ! negates a glob class, so ^ is one of the characters it matches.
    assert local_options._snapshot_options(snapshot) == {("cfg", "train")}


def test_snapshot_options_survive_a_glob_the_engine_cannot_compile(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, "configs:\n- config_name: cfg\n  data_files: '[z-a].jsonl'\n")
    (snapshot / "a.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_treat_a_null_data_dir_as_the_root(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, "configs:\n- config_name: cfg\n  data_dir: null\n")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("cfg", "train")}


def test_snapshot_all_files_report_a_truncated_walk(tmp_path, monkeypatch):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    for index in range(3):
        (snapshot / f"train-{index}.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    monkeypatch.setattr(local_options, "_MAX_SNAPSHOT_DATA_FILES", 2)

    assert local_options._snapshot_all_files(snapshot) is None


def test_snapshot_options_read_a_data_dir_that_steps_back_inside(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, "configs:\n- config_name: cfg\n  data_dir: a/../b\n")
    (snapshot / "a").mkdir()
    (snapshot / "a" / "other.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")
    (snapshot / "b").mkdir()
    (snapshot / "b" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("cfg", "train")}


def test_snapshot_options_reject_a_data_dir_whose_first_step_is_missing(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, "configs:\n- config_name: cfg\n  data_dir: a/../b\n")
    (snapshot / "b").mkdir(parents = True)
    (snapshot / "b" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # The loader walks the string as written, so a missing a/ empties the dataset.
    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_skip_an_unshowable_config_without_its_siblings(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    long_name = "n" * (local_options._MAX_OPTION_LENGTH + 1)
    _card(
        snapshot,
        f"configs:\n- config_name: {long_name}\n  data_dir: a\n- config_name: short\n  data_dir: b\n",
    )
    for name in ("a", "b"):
        (snapshot / name).mkdir(parents = True)
        (snapshot / name / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("short", "train")}


def test_snapshot_options_drop_a_config_recorded_with_no_splits(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(
        snapshot,
        "configs:\n- config_name: cfg\n  data_dir: d\n- config_name: other\n  data_dir: e\ndataset_info:\n- config_name: cfg\n  splits: []\n",
    )
    for name in ("d", "e"):
        (snapshot / name).mkdir(parents = True)
        (snapshot / name / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("other", "train")}


def test_snapshot_options_reject_a_data_files_list_mixing_shapes(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, "configs:\n- config_name: cfg\n  data_files:\n  - train.jsonl\n  - split: test\n    path: test.jsonl\n")
    for name in ("train.jsonl", "test.jsonl"):
        (snapshot / name).write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_ignore_a_mixed_case_media_suffix(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    (snapshot / "train").mkdir(parents = True)
    for name in ("a.Jpg", "b.Jpg", "c.Jpg"):
        (snapshot / "train" / name).write_bytes(b"x")
    (snapshot / "train" / "d.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    # datasets registers folder suffixes in lower and upper case only, so .Jpg is not media.
    assert local_options._snapshot_options(snapshot) == {("default", "train")}


@pytest.mark.parametrize("name", ["train.jsonl.txt", "train.csv.backup"])
def test_snapshot_options_offer_a_compound_suffix_the_loader_keeps(tmp_path, name):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / name).write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("default", "train")}


def test_snapshot_options_still_hide_a_codec_studio_cannot_open(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    (snapshot / "train.jsonl.zst").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


@pytest.mark.parametrize("features", ["- dtype: string", "- name: t\n    dtype: nope"])
def test_snapshot_options_reject_features_the_loader_cannot_build(tmp_path, features):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, f"configs:\n- config_name: cfg\n  data_dir: d\n  features:\n  {features}\n")
    (snapshot / "d").mkdir()
    (snapshot / "d" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


@pytest.mark.parametrize("dtype", ["string", "image", "timestamp[s]", "decimal128(10,2)"])
def test_snapshot_options_accept_features_the_loader_builds(tmp_path, dtype):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, f"configs:\n- config_name: cfg\n  data_dir: d\n  features:\n  - name: text\n    dtype: {dtype}\n")
    (snapshot / "d").mkdir()
    (snapshot / "d" / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == {("cfg", "train")}


def test_snapshot_options_reject_an_absolute_declared_path(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, "configs:\n- config_name: cfg\n  data_files: /train.jsonl\n")
    (snapshot / "train.jsonl").write_text('{"text":"row"}\n', encoding = "utf-8")

    assert local_options._snapshot_options(snapshot) == set()


def test_snapshot_options_reject_a_mapping_data_files(tmp_path):
    snapshot = tmp_path / "datasets--org--data" / "snapshots" / "commit"
    _card(snapshot, "configs:\n- config_name: cfg\n  data_files:\n    train: train.jsonl\n    test: test.jsonl\n")
    for name in ("train.jsonl", "test.jsonl"):
        (snapshot / name).write_text('{"text":"row"}\n', encoding = "utf-8")

    # MetadataConfigs takes a string or a list here and raises on a mapping.
    assert local_options._snapshot_options(snapshot) == set()
