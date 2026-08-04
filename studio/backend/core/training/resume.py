# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Helpers for validating resumable training outputs."""

import json
import pickletools
import time
import zipfile
from pathlib import Path
from typing import Optional

from utils.paths import outputs_root, resolve_output_dir

# Records a resume that rewound past newer checkpoints; see record_resume_rewind.
_REWIND_MARKER = "resume_rewind.json"


def _is_under_outputs(path: Path) -> bool:
    resolved = path.resolve(strict = False)
    root = outputs_root().resolve(strict = False)
    try:
        resolved.relative_to(root)
        return True
    except ValueError:
        return False


def current_training_backend() -> str:
    """Backend a new run trains with on this host (worker.py's MLX fast-path)."""
    from core.training.training import is_apple_silicon_training_platform
    return "mlx" if is_apple_silicon_training_platform() else "pt"


def has_resume_state(path_value: Optional[str]) -> bool:
    if not path_value:
        return False
    # Backend-scoped: a bundle the other backend wrote cannot resume here, so it
    # must not light up Resume in history either.
    return get_resume_checkpoint_path(path_value, backend = current_training_backend()) is not None


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
            try:
                from safetensors import SafetensorError, safe_open
            except ImportError:
                return False
            try:
                with safe_open(str(path), framework = "np") as state:
                    return bool(state.keys())
            except SafetensorError:
                return False
        if path.suffix in {".bin", ".pt"}:
            with zipfile.ZipFile(path) as state:
                infos = state.infolist()
                names = [info.filename for info in infos]
                data_name = next(
                    (name for name in names if name == "data.pkl" or name.endswith("/data.pkl")),
                    None,
                )
                if data_name is None:
                    return False
                data_prefix = data_name.removesuffix("data.pkl") + "data/"
                operations = list(pickletools.genops(state.read(data_name)))
                if not operations or operations[-1][0].name != "STOP":
                    return False
                if not require_tensor:
                    return True
                # Require a non-empty tensor record; a zero-byte one fails torch.load.
                return any(
                    info.filename.startswith(data_prefix)
                    and not info.is_dir()
                    and info.file_size > 0
                    for info in infos
                )
        # Unrecognized state-file formats are not usable resume state.
        return False
    except (OSError, ValueError, zipfile.BadZipFile):
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


_INDEX_SHARD_SUFFIX = {
    "model.safetensors.index.json": ".safetensors",
    "pytorch_model.bin.index.json": ".bin",
}


def _valid_indexed_shard(checkpoint: Path, shard: object, expected_suffix: str) -> bool:
    # Shard must be a relative, in-format path contained in the checkpoint dir.
    if not isinstance(shard, str) or not shard:
        return False
    if Path(shard).is_absolute() or Path(shard).suffix != expected_suffix:
        return False
    try:
        root = checkpoint.resolve(strict = True)
        candidate = (checkpoint / shard).resolve(strict = True)
        candidate.relative_to(root)
    except (OSError, ValueError):
        return False
    return _valid_state_file(candidate)


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
        expected_suffix = _INDEX_SHARD_SUFFIX[name]
        if shards and all(_valid_indexed_shard(path, shard, expected_suffix) for shard in shards):
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
            # optimizer/scheduler state can be validly tensor-free (e.g. SGD without
            # momentum); _has_model_state still requires real model tensors.
            and _valid_state_file(path / "optimizer.pt", require_tensor = False)
            and _valid_state_file(path / "scheduler.pt", require_tensor = False)
        )
        if backend is None and not valid_bundle:
            valid_bundle = _valid_state_file(path / "adapters.safetensors") and _valid_state_file(
                path / "optimizer_state.safetensors"
            )
    return step_valid and valid_bundle


def _checkpoint_written_at(path: Path) -> float:
    # trainer_state.json is rewritten on every save, so it dates the checkpoint even
    # when a resumed run rewrites a same-numbered directory.
    try:
        return (path / "trainer_state.json").stat().st_mtime
    except OSError:
        return -1.0


def resume_step_cap(run_dir: Path) -> Optional[int]:
    """Highest step a plain resume may select, from a recorded rewind (None: no cap)."""
    try:
        marker = json.loads((run_dir / _REWIND_MARKER).read_text(encoding = "utf-8"))
        step, recorded_at = marker["step"], float(marker["recorded_at"])
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        KeyError,
        ValueError,
    ):
        return None
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        return None
    # A checkpoint written after the rewind belongs to the new timeline and raises the
    # cap to itself; the abandoned siblings it did not reach stay out of selection.
    for checkpoint in run_dir.glob("checkpoint-*"):
        checkpoint_step = _checkpoint_step(checkpoint)
        if checkpoint_step > step and _checkpoint_written_at(checkpoint) > recorded_at:
            step = checkpoint_step
    return step


def record_resume_rewind(resume_checkpoint: str, backend: Optional[str] = None) -> None:
    """Remember that a resume rewound past newer checkpoints, or drop the record.

    Deleting the abandoned checkpoints would discard user data, so the rewind is
    recorded instead: until the new timeline writes past that step, a later plain
    resume must not jump forward onto the timeline this one abandoned.
    """
    path = Path(resume_checkpoint)
    step = _checkpoint_step(path)
    if step < 0:
        return
    run_dir = path.parent
    rewound = any(
        _checkpoint_step(sibling) > step and is_resume_checkpoint_valid(sibling, backend = backend)
        for sibling in run_dir.glob("checkpoint-*")
    )
    try:
        if rewound:
            (run_dir / _REWIND_MARKER).write_text(
                json.dumps({"step": step, "recorded_at": time.time()}), encoding = "utf-8"
            )
        else:
            # Nothing newer to keep out of reach: this resume adopts the newest timeline.
            (run_dir / _REWIND_MARKER).unlink(missing_ok = True)
    except OSError:
        pass


def get_resume_checkpoint_path(
    path_value: str,
    expected_step: Optional[int] = None,
    backend: Optional[str] = None,
) -> Optional[str]:
    path = resolve_output_dir(path_value)
    if not _is_under_outputs(path) or not path.is_dir():
        return None
    # An explicitly targeted checkpoint is the user's choice; only the sibling scan
    # below is capped by a recorded rewind.
    if is_resume_checkpoint_valid(path, expected_step, backend):
        return str(path)

    step_cap = resume_step_cap(path)
    checkpoints = sorted(path.glob("checkpoint-*"), key = _checkpoint_step, reverse = True)
    return next(
        (
            str(checkpoint)
            for checkpoint in checkpoints
            if _checkpoint_step(checkpoint) >= 0
            and (step_cap is None or _checkpoint_step(checkpoint) <= step_cap)
            and is_resume_checkpoint_valid(checkpoint, expected_step, backend)
        ),
        None,
    )


def normalize_resume_output_dir(path_value: str) -> str:
    path = resolve_output_dir(path_value)
    if not _is_under_outputs(path):
        raise ValueError("Resume checkpoint must be inside Unsloth outputs.")
    return str(path)


def resume_run_dir(resume_checkpoint: str) -> str:
    """Run directory a resumed run continues in; new checkpoints nest under it."""
    path = Path(resume_checkpoint)
    return str(path.parent) if path.name.startswith("checkpoint-") else str(path)


def find_resumable_run(resume_dir: str) -> Optional[dict]:
    """DB lookup for a resume target; a checkpoint-N path maps to its parent run dir."""
    from storage.studio_db import get_resumable_run_by_output_dir

    run = get_resumable_run_by_output_dir(resume_dir)
    if run is None:
        path = Path(resume_dir)
        if path.name.startswith("checkpoint-"):
            run = get_resumable_run_by_output_dir(str(path.parent))
    return run


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
        # A save-time crash can report final_step == total_steps with no artifacts; checkpoint state alone decides resumability.
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
