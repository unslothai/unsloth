# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Completed image runs retain unlimited checkpoints without offering an implicit rollback."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from core.training import diffusion_checkpoint as dc


def _save(root, step, value):
    """Write a real checkpoint bundle containing a small CPU adapter tensor."""
    identity = dc.CheckpointIdentity(
        family = "sdxl",
        base_model = "example/model",
        lora_target_modules = ("to_q",),
        lora_rank = 4,
        lora_alpha = 4,
        precision = "bf16",
        base_precision = "nf4",
        resolution = 1024,
    )
    return Path(
        dc.save_checkpoint(
            output_dir = root,
            step = step,
            adapter_state = {"weight": torch.tensor([value])},
            identity = identity,
            target_steps = 30,
            save_total_limit = 0,
        )
    )


def test_completed_run_keeps_all_bundles_outside_automatic_resume(tmp_path):
    """Completed checkpoints remain readable but are not chosen as the latest resume point."""
    _save(tmp_path, 10, 10)
    _save(tmp_path, 20, 20)

    dc.retire_own_checkpoints(tmp_path, [], resumed_here = False, keep_completed = True)

    assert dc.list_checkpoints(tmp_path) == []
    assert dc.latest_valid_checkpoint(tmp_path) is None
    archived = sorted(tmp_path.glob("completed-checkpoints-*/checkpoint-*"))
    assert [p.name for p in archived] == ["checkpoint-10", "checkpoint-20"]
    for path, value in zip(archived, (10, 20)):
        manifest = json.loads((path / dc.TRAINER_STATE_FILENAME).read_text(encoding = "utf-8"))
        assert manifest["global_step"] == value
        assert dc.read_checkpoint(path) == manifest
        assert load_file(path / dc.ADAPTER_FILENAME)["weight"].item() == value


def test_resumed_run_restores_displaced_source_after_archiving(tmp_path):
    """A resumed run retains its new bundle and restores a source it overwrote."""
    original = _save(tmp_path, 10, 1)
    previous = dc.snapshot_checkpoints(tmp_path)
    _save(tmp_path, 10, 2)

    dc.retire_own_checkpoints(tmp_path, previous, resumed_here = True, keep_completed = True)

    assert load_file(str(original / dc.ADAPTER_FILENAME))["weight"].item() == 1
    archived = list(tmp_path.glob("completed-checkpoints-*/checkpoint-10"))
    assert len(archived) == 1
    assert load_file(archived[0] / dc.ADAPTER_FILENAME)["weight"].item() == 2


def test_repeated_completed_runs_do_not_overwrite_retained_bundles(tmp_path):
    """A later run with the same step numbers leaves the earlier archive intact."""
    _save(tmp_path, 10, 1)
    dc.retire_own_checkpoints(tmp_path, [], resumed_here = False, keep_completed = True)
    _save(tmp_path, 10, 2)
    dc.retire_own_checkpoints(tmp_path, [], resumed_here = False, keep_completed = True)

    archived = list(tmp_path.glob("completed-checkpoints-*/checkpoint-10"))
    assert len(archived) == 2
    assert sorted(load_file(path / dc.ADAPTER_FILENAME)["weight"].item() for path in archived) == [
        1,
        2,
    ]


def test_finite_retention_keeps_existing_completion_cleanup(tmp_path):
    """The default completion path still retires temporary resume bundles."""
    _save(tmp_path, 10, 1)
    dc.retire_own_checkpoints(tmp_path, [], resumed_here = False)

    assert dc.list_checkpoints(tmp_path) == []
    assert list(tmp_path.glob("completed-checkpoints-*")) == []


def test_archive_failure_restores_already_moved_bundles(tmp_path, monkeypatch):
    """A failed move leaves the original resume candidates in place."""
    _save(tmp_path, 10, 10)
    _save(tmp_path, 20, 20)
    replace = dc.os.replace

    def fail_second(source, destination):
        """Simulate an actual filesystem failure on the second checkpoint move."""
        if str(source).endswith("checkpoint-10"):
            raise OSError("checkpoint move failed")
        return replace(source, destination)

    monkeypatch.setattr(dc.os, "replace", fail_second)
    with pytest.raises(OSError, match = "checkpoint move failed"):
        dc.retire_own_checkpoints(tmp_path, [], resumed_here = False, keep_completed = True)

    assert (tmp_path / "checkpoint-10" / dc.ADAPTER_FILENAME).is_file()
    assert (tmp_path / "checkpoint-20" / dc.ADAPTER_FILENAME).is_file()
