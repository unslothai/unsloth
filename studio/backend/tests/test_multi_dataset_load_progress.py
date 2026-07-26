# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib.util
from pathlib import Path


_path = Path(__file__).parents[1] / "core" / "training" / "dataset_progress.py"
_spec = importlib.util.spec_from_file_location("dataset_progress", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)
training_dataset_entries_with_progress = _module.training_dataset_entries_with_progress


def test_two_dataset_run_reports_each_dataset_before_loading():
    reports = []

    class Trainer:
        def _update_progress(self, **payload):
            reports.append(payload)

    entries = [{"hf_dataset": "org/one"}, {"hf_dataset": "org/two"}]
    assert list(training_dataset_entries_with_progress(Trainer(), entries)) == entries
    assert [
        (report["current_dataset_index"], report["current_dataset_total"])
        for report in reports
    ] == [(1, 2), (2, 2)]
    assert [report["current_dataset_repository_id"] for report in reports] == [
        "org/one", "org/two",
    ]
