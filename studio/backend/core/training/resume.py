# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Helpers for validating resumable training outputs."""

from pathlib import Path
from typing import Optional

from utils.paths import outputs_root, resolve_output_dir


def _is_under_outputs(path: Path) -> bool:
    resolved = path.resolve(strict = False)
    root = outputs_root().resolve(strict = False)
    try:
        resolved.relative_to(root)
        return True
    except ValueError:
        return False


def has_resume_state(path_value: Optional[str]) -> bool:
    if not path_value:
        return False
    return get_resume_checkpoint_path(path_value) is not None


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        return -1


def get_resume_checkpoint_path(path_value: str) -> Optional[str]:
    path = resolve_output_dir(path_value)
    if not _is_under_outputs(path) or not path.is_dir():
        return None
    if (path / "trainer_state.json").is_file():
        return str(path)

    checkpoints = [
        child
        for child in path.glob("checkpoint-*")
        if child.is_dir() and (child / "trainer_state.json").is_file()
    ]
    if not checkpoints:
        return None
    return str(max(checkpoints, key = _checkpoint_step))


def normalize_resume_output_dir(path_value: str) -> str:
    path = resolve_output_dir(path_value)
    if not _is_under_outputs(path):
        raise ValueError("Resume checkpoint must be inside Studio outputs.")
    return str(path)


def can_resume_run(run: dict) -> bool:
    if run.get("resumed_later"):
        return False

    final_step = run.get("final_step")
    total_steps = run.get("total_steps")
    has_remaining_steps = (
        not isinstance(final_step, int)
        or not isinstance(total_steps, int)
        or total_steps <= 0
        or final_step < total_steps
    )
    return (
        run.get("status") == "stopped"
        and has_remaining_steps
        and has_resume_state(run.get("output_dir"))
    )
