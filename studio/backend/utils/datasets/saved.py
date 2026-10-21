# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path


def load_saved_dataset(path):
    """A `save_to_disk` folder as one Dataset (a DatasetDict's train split, else its first); None for any other path."""
    path = Path(path)
    if not (path / "dataset_dict.json").is_file() and not (
        (path / "dataset_info.json").is_file() and (path / "state.json").is_file()
    ):
        return None
    from datasets import DatasetDict, load_from_disk

    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"] if "train" in dataset else next(iter(dataset.values()))
    return dataset
