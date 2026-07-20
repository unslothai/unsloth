# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Helpers for validating resumable training outputs."""

import json
import pickletools
import zipfile
from pathlib import Path
from typing import Optional

from safetensors import SafetensorError, safe_open

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


_MODEL_FILES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
)
_MODEL_INDEXES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


def _valid_state_file(path: Path, require_tensor: bool = True) -> bool:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        if path.suffix == ".safetensors":
            with safe_open(str(path), framework = "np") as state:
                return bool(state.keys())
        if path.suffix in {".bin", ".pt"}:
            with zipfile.ZipFile(path) as state:
                names = state.namelist()
                data_name = next(
                    (name for name in names if name == "data.pkl" or name.endswith("/data.pkl")),
                    None,
                )
                if data_name is None:
                    return False
                data_prefix = data_name.removesuffix("data.pkl") + "data/"
                operations = list(pickletools.genops(state.read(data_name)))
                return (
                    bool(operations)
                    and operations[-1][0].name == "STOP"
                    and (not require_tensor or any(name.startswith(data_prefix) for name in names))
                )
        return True
    except (OSError, ValueError, SafetensorError, zipfile.BadZipFile):
        return False


def _checkpoint_state(path: Path) -> Optional[int]:
    try:
        state = json.loads((path / "trainer_state.json").read_text(encoding = "utf-8"))
        step = state.get("global_step") if isinstance(state, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        return None
    directory_step = _checkpoint_step(path)
    return step if directory_step < 0 or step == directory_step else None


def _has_model_state(path: Path) -> bool:
    if any(_valid_state_file(path / name) for name in _MODEL_FILES):
        return True
    for name in _MODEL_INDEXES:
        try:
            index = json.loads((path / name).read_text(encoding = "utf-8"))
            shards = set(index["weight_map"].values())
        except (
            AttributeError,
            OSError,
            KeyError,
            TypeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            continue
        if shards and all(
            isinstance(shard, str) and _valid_state_file(path / shard) for shard in shards
        ):
            return True
    return False


def is_resume_checkpoint_valid(
    path: Path,
    expected_step: Optional[int] = None,
    backend: Optional[str] = None,
) -> bool:
    step = _checkpoint_state(path) if path.is_dir() else None
    step_valid = step is not None and (expected_step is None or step == expected_step)
    if backend == "mlx":
        valid_bundle = _valid_state_file(path / "adapters.safetensors") and _valid_state_file(
            path / "optimizer_state.safetensors"
        )
    else:
        valid_bundle = (
            _has_model_state(path)
            and _valid_state_file(path / "optimizer.pt")
            and _valid_state_file(path / "scheduler.pt", require_tensor = False)
        )
        if backend is None and not valid_bundle:
            valid_bundle = _valid_state_file(path / "adapters.safetensors") and _valid_state_file(
                path / "optimizer_state.safetensors"
            )
    return step_valid and valid_bundle


def get_resume_checkpoint_path(
    path_value: str, expected_step: Optional[int] = None
) -> Optional[str]:
    path = resolve_output_dir(path_value)
    if not _is_under_outputs(path) or not path.is_dir():
        return None
    if is_resume_checkpoint_valid(path, expected_step):
        return str(path)

    checkpoints = sorted(path.glob("checkpoint-*"), key = _checkpoint_step, reverse = True)
    return next(
        (
            str(checkpoint)
            for checkpoint in checkpoints
            if _checkpoint_step(checkpoint) >= 0
            and is_resume_checkpoint_valid(checkpoint, expected_step)
        ),
        None,
    )


def normalize_resume_output_dir(path_value: str) -> str:
    path = resolve_output_dir(path_value)
    if not _is_under_outputs(path):
        raise ValueError("Resume checkpoint must be inside Unsloth outputs.")
    return str(path)


def _run_config(run: dict) -> dict:
    raw_config = run.get("config_json")
    if isinstance(raw_config, dict):
        return raw_config
    if not isinstance(raw_config, str) or not raw_config.strip():
        return {}
    try:
        parsed = json.loads(raw_config)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _uses_s3_dataset(run: dict) -> bool:
    config = _run_config(run)
    return config.get("dataset_source") == "s3" or "s3_dataset" in config


def can_resume_run(run: dict) -> bool:
    if run.get("resumed_later"):
        return False
    # Set when a stop-and-save failed to write a current-step checkpoint.
    if run.get("resume_blocked"):
        return False
    if _uses_s3_dataset(run):
        return False

    status = run.get("status")
    if status == "error":
        # A save-time crash can report final_step == total_steps with no final
        # artifacts; checkpoint state alone decides resumability.
        return has_resume_state(run.get("output_dir"))

    final_step = run.get("final_step")
    total_steps = run.get("total_steps")
    has_remaining_steps = (
        not isinstance(final_step, int)
        or not isinstance(total_steps, int)
        or total_steps <= 0
        or final_step < total_steps
    )
    return status == "stopped" and has_remaining_steps and has_resume_state(run.get("output_dir"))
