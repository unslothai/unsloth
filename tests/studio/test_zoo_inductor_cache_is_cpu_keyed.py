# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Core zoo's Inductor cache must only be restored on the CPU target that built it.

Inductor compiles its CPU kernels with -march=native and its own cache key does not name the
CPU. When the Core zoo cache was keyed on the compiler stack alone, a cache saved by a main push
on one runner model was restored by PRs on another, and the compiled kernels crashed pytest
workers (free(): invalid pointer, buffer overflow detected) in every leg.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "consolidated-tests-ci.yml"
)
STACK_OUTPUT = "${{ steps.inductor-stack.outputs.stack }}"


def _zoo_steps() -> list[dict]:
    data = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    return data["jobs"]["consolidated-zoo"]["steps"]


def _step(predicate) -> dict:
    found = [step for step in _zoo_steps() if predicate(step)]
    assert len(found) == 1, found
    return found[0]


def _stack_step() -> dict:
    return _step(lambda step: step.get("id") == "inductor-stack")


def _restore_step() -> dict:
    return _step(lambda step: step.get("id") == "inductor-cache")


def _save_step() -> dict:
    return _step(
        lambda step: "actions/cache/save@" in step.get("uses", "")
        and "inductor-cache" in str(step.get("with", {}).get("path", ""))
    )


def test_restore_and_save_keys_carry_the_stack_hash():
    restore = _restore_step()["with"]
    save = _save_step()["with"]
    assert STACK_OUTPUT in restore["key"]
    assert STACK_OUTPUT in restore["restore-keys"]
    assert save["key"] == restore["key"]
    assert restore["key"].startswith(restore["restore-keys"].strip())
    # v1 entries were built before the CPU joined the key; none of them may be restored.
    assert "zoo-inductor-v1-" not in WORKFLOW.read_text(encoding = "utf-8")


def _run_stack_step(tmp_path: Path, march: str | None) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok = True)
    gcc = bin_dir / "gcc"
    if march is None:
        gcc.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
    else:
        gcc.write_text(
            "#!/bin/sh\n"
            'printf "The following options are target specific:\\n"\n'
            f'printf "  -march=                     \\t\\t{march}\\n"\n'
            f'printf "  -mtune=                     \\t\\t{march}\\n"\n',
            encoding = "utf-8",
        )
    gcc.chmod(0o755)
    # Fixed compiler stack versions, so only the CPU target can move the hash.
    python = bin_dir / "python"
    python.write_text("#!/bin/sh\necho 2.10.0 3.6.0 4.57.6\n", encoding = "utf-8")
    python.chmod(0o755)
    output = tmp_path / f"output-{march}"
    output.write_text("", encoding = "utf-8")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "GITHUB_OUTPUT": str(output),
        "GITHUB_ENV": str(tmp_path / "env"),
        "RUNNER_TEMP": str(tmp_path),
    }
    return subprocess.run(
        ["bash", "-e", "-c", _stack_step()["run"]],
        env = env,
        capture_output = True,
        text = True,
        timeout = 60,
    )


def _stack(tmp_path: Path, march: str) -> str:
    result = _run_stack_step(tmp_path, march)
    assert result.returncode == 0, result.stderr
    lines = (tmp_path / f"output-{march}").read_text(encoding = "utf-8").splitlines()
    stacks = [line.split("=", 1)[1] for line in lines if line.startswith("stack=")]
    assert len(stacks) == 1 and stacks[0], lines
    assert march in result.stdout, result.stdout
    return stacks[0]


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason = "runs the step's bash"
)
def test_a_different_cpu_target_gets_a_different_cache(tmp_path):
    granite = _stack(tmp_path, "graniterapids")
    zen4 = _stack(tmp_path, "znver4")
    assert granite != zen4
    assert _stack(tmp_path, "graniterapids") == granite


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason = "runs the step's bash"
)
def test_an_unknown_cpu_target_refuses_to_key_the_cache(tmp_path):
    result = _run_stack_step(tmp_path, None)
    assert result.returncode != 0
    assert "stack=" not in (tmp_path / "output-None").read_text(encoding = "utf-8")
