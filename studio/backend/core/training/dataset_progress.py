# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small dependency-free helpers for reporting multi-dataset load progress."""


def training_dataset_entries_with_progress(trainer, entries):
    """Yield selections after publishing the dataset that is about to load."""
    total = len(entries)
    for index, entry in enumerate(entries, start = 1):
        trainer._update_progress(
            current_dataset_index = index,
            current_dataset_total = total,
            current_dataset_repository_id = entry.get("hf_dataset"),
            status_message = f"Loading dataset {index}/{total}...",
        )
        yield entry
