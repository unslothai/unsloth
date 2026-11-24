# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""REAP calibration loads a Hub dataset inside the export worker, a spawned child: the dataset card
data_files guard (absolute paths, globs, loopback URLs) must be installed there too, before anything
else runs."""

import os
import warnings
from unittest.mock import patch

import pytest

from core.export import worker
from loggers.config import LogConfig
from utils.security import hub_data_files


class _Stop(Exception):
    pass


def test_export_worker_installs_the_dataset_card_guard(monkeypatch):
    installed = []
    monkeypatch.setattr(hub_data_files, "install", lambda: installed.append(True))
    monkeypatch.setattr(LogConfig, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(worker, "_setup_log_capture", lambda _: None)

    def stop(*_a, **_k):
        raise _Stop

    monkeypatch.setattr(worker, "apply_gpu_ids", stop)
    # run_export_process configures its own process; keep that from leaking into this one.
    with patch.dict(os.environ), warnings.catch_warnings(), pytest.raises(_Stop):
        worker.run_export_process(cmd_queue = None, resp_queue = None, config = {"checkpoint_path": "x"})
    assert installed == [True]
