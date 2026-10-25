# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from pathlib import Path


def load_saved_dataset(path):
    """A `save_to_disk` folder as one Dataset (a DatasetDict's train split, else its first); None for any other path."""
    path = Path(path)
    if not _is_saved_dataset(path) and not _is_saved_dataset_dict(path):
        return None
    _require_manifest_inside(path)
    from datasets import DatasetDict, load_from_disk

    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"] if "train" in dataset else next(iter(dataset.values()))
    return dataset


def _is_saved_dataset(path: Path) -> bool:
    return (path / "dataset_info.json").is_file() and (path / "state.json").is_file()


def _is_saved_dataset_dict(path: Path) -> bool:
    # A data file can be named dataset_dict.json too, so the manifest must also list split folders that are saves.
    try:
        splits = json.loads((path / "dataset_dict.json").read_text(encoding = "utf-8")).get("splits")
    except (OSError, ValueError, AttributeError):
        return False
    return (
        isinstance(splits, list)
        and bool(splits)
        and all(isinstance(split, str) and _is_saved_dataset(path / split) for split in splits)
    )


def _require_manifest_inside(path: Path) -> None:
    # load_from_disk joins manifest paths onto the folder unchecked: a crafted manifest could read another account's files.
    root = path.resolve()

    def check(entry: Path) -> Path:
        resolved = entry.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"saved dataset at {path} refers to files outside its folder")
        return resolved

    folders = [path]
    if _is_saved_dataset_dict(path):
        splits = (
            json.loads((path / "dataset_dict.json").read_text(encoding = "utf-8")).get("splits") or []
        )
        folders = [check(path / str(split)) for split in splits]
    for folder in folders:
        state = folder / "state.json"
        if state.is_file():
            for data_file in json.loads(state.read_text(encoding = "utf-8")).get("_data_files") or []:
                check(folder / str(data_file.get("filename", "")))
