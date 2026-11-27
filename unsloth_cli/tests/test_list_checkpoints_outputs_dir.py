# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""unsloth list-checkpoints --outputs-dir: a path outside the outputs folder is a usage error, not a traceback."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _list_checkpoints(tmp_path: Path, outputs_dir: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    for key in ("STUDIO_HOME", "UNSLOTH_HOME", "UNSLOTH_PORTABLE"):
        env.pop(key, None)
    env.update(
        HOME = str(tmp_path / "home"),
        UNSLOTH_STUDIO_HOME = str(tmp_path / "studio"),
        PYTHONPATH = str(REPO_ROOT),
        NO_COLOR = "1",
        TERM = "dumb",
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "unsloth_cli",
            "list-checkpoints",
            "--outputs-dir",
            str(outputs_dir),
        ],
        cwd = str(tmp_path),
        env = env,
        capture_output = True,
        text = True,
        timeout = 300,
    )


def _make_run(outputs: Path) -> Path:
    checkpoint = outputs / "my_run" / "checkpoint-10"
    checkpoint.mkdir(parents = True)
    (checkpoint / "adapter_config.json").write_text("{}")
    return checkpoint


def test_outputs_dir_outside_the_outputs_folder_is_a_usage_error(tmp_path):
    outside = tmp_path / "project" / "outputs"
    _make_run(outside)

    result = _list_checkpoints(tmp_path, outside)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr
    errors = [line for line in result.stderr.splitlines() if line.startswith("Error: ")]
    assert errors, result.stderr
    assert str((tmp_path / "studio" / "outputs").resolve()) in errors[0]


def test_missing_outputs_dir_warning_names_the_folder_it_searched(tmp_path):
    result = _list_checkpoints(tmp_path, Path("runs"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "No checkpoints found." in result.stdout
    assert f"Outputs directory not found: {tmp_path / 'studio' / 'outputs' / 'runs'}" in (
        result.stdout + result.stderr
    )


def test_outputs_dir_inside_the_outputs_folder_lists_its_checkpoints(tmp_path):
    checkpoint = _make_run(tmp_path / "studio" / "outputs")

    result = _list_checkpoints(tmp_path, tmp_path / "studio" / "outputs")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "\nmy_run:\n" in result.stdout
    assert f"checkpoint-10: {checkpoint.resolve()}" in result.stdout
