# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A standalone custom task YAML is staged inside the temp dir whatever its task name says."""

from pathlib import Path

import pytest

pytest.importorskip("yaml")

from unsloth_cli.commands.eval import resolve_tasks


def test_task_name_cannot_escape_the_staging_dir(tmp_path):
    victim_dir = tmp_path / "victim"
    victim_dir.mkdir()
    victim = victim_dir / "settings.yaml"
    victim.write_text("keep: me\n")
    stage = tmp_path / "stage"
    stage.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    task = src / "evil.yaml"
    task.write_text("task: ../../victim/settings\ndataset_path: x\n")

    names, includes = resolve_tasks(str(task), "input", "target", stage)

    assert victim.read_text() == "keep: me\n"
    assert names == ["../../victim/settings"]
    staged = [Path(p) for p in includes]
    assert all(p.resolve().is_relative_to(stage.resolve()) for p in staged)
    assert len(list((stage / "custom").glob("*.yaml"))) == 1


def test_plain_task_is_staged_and_named_as_before(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    task = tmp_path / "mytask.yaml"
    task.write_text("task: mytask\ndataset_path: x\n")
    names, includes = resolve_tasks(str(task), "input", "target", stage)
    assert names == ["mytask"]
    assert includes == [str((stage / "custom").resolve())]
    (copied,) = (stage / "custom").glob("*.yaml")
    assert copied.read_text() == task.read_text()
