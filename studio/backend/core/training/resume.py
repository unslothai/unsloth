# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Helpers for validating resumable training outputs."""

import json
import os
import pickletools
import zipfile
from pathlib import Path
from typing import Any, Optional, Sequence

from utils.paths import outputs_root, resolve_output_dir
from core.training.manifest import MANIFEST_FILENAME, ManifestError, parse_manifest


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


def _checkpoint_max_steps(path: Path) -> Optional[int]:
    try:
        state = json.loads((path / "trainer_state.json").read_text(encoding = "utf-8"))
        max_steps = state.get("max_steps") if isinstance(state, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        return None
    return max_steps


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


class CheckpointImportError(ValueError):
    """A checkpoint selected through the file browser is unsafe or incomplete."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _detected_backend(path: Path) -> str:
    if (path / "adapters.safetensors").exists() or (path / "optimizer_state.safetensors").exists():
        return "mlx"
    return "transformers"


def _configuration_metadata(path: Path) -> dict[str, Any]:
    """Read safe JSON metadata only; never unpickle a user-selected file."""
    metadata: dict[str, Any] = {}
    for filename in ("training_config.json", "config.json", "adapter_config.json"):
        candidate = path / filename
        if not candidate.is_file() and path.name.startswith("checkpoint-"):
            candidate = path.parent / filename
        try:
            value = json.loads(candidate.read_text(encoding = "utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            metadata[filename] = value
    return metadata


def _contained_in(path: Path, roots: Sequence[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root.resolve(strict = True))
            return True
        except (OSError, ValueError):
            continue
    return False


def inspect_import_checkpoint(path_value: str, approved_roots: Sequence[Path]) -> dict[str, Any]:
    """Validate an external checkpoint independently of training history policy."""
    raw = Path(path_value).expanduser()
    if ".." in raw.parts:
        raise CheckpointImportError(["Path traversal ('..') is not allowed."])
    try:
        selected = raw.resolve(strict = True)
    except OSError as exc:
        raise CheckpointImportError([f"Checkpoint directory does not exist: {exc}"]) from exc
    if not selected.is_dir():
        raise CheckpointImportError(["Selected checkpoint path is not a directory."])
    if not _contained_in(selected, approved_roots):
        raise CheckpointImportError(
            ["Checkpoint directory resolves outside the approved browse roots."]
        )
    if not os.access(selected, os.R_OK | os.X_OK):
        raise CheckpointImportError(
            ["Selected checkpoint output directory is not readable."]
        )

    checkpoint = selected
    if _checkpoint_state(checkpoint) is None:
        candidates = sorted(selected.glob("checkpoint-*"), key = _checkpoint_step, reverse = True)
        checkpoint = next((p for p in candidates if _checkpoint_state(p) is not None), selected)
    # Containment is checked again after selecting a nested checkpoint and catches
    # checkpoint-* symlinks escaping the selected directory.
    try:
        checkpoint = checkpoint.resolve(strict = True)
        checkpoint.relative_to(selected)
    except (OSError, ValueError) as exc:
        raise CheckpointImportError(
            ["Selected checkpoint resolves outside its directory."]
        ) from exc

    errors: list[str] = []
    step = _checkpoint_state(checkpoint)
    if step is None:
        errors.append("trainer_state.json is missing, corrupt, or has a mismatched global_step.")
    backend = _detected_backend(checkpoint)
    if backend == "mlx":
        if not _valid_state_file(checkpoint / "adapters.safetensors"):
            errors.append("adapters.safetensors is missing or corrupt.")
        if not _valid_state_file(checkpoint / "optimizer_state.safetensors"):
            errors.append("optimizer_state.safetensors is missing or corrupt.")
    else:
        if not _has_model_state(checkpoint):
            errors.append("Model or adapter state is missing or corrupt.")
        if not _valid_state_file(checkpoint / "optimizer.pt", require_tensor = False):
            errors.append("optimizer.pt is missing or corrupt.")
        if not _valid_state_file(checkpoint / "scheduler.pt", require_tensor = False):
            errors.append("scheduler.pt is missing or corrupt.")
    if errors:
        raise CheckpointImportError(errors)
    manifest = None
    manifest_path = checkpoint / MANIFEST_FILENAME
    if not manifest_path.is_file() and checkpoint.name.startswith("checkpoint-"):
        manifest_path = checkpoint.parent / MANIFEST_FILENAME
    if manifest_path.is_file():
        try:
            manifest = parse_manifest(manifest_path)
        except ManifestError as exc:
            raise CheckpointImportError([str(exc)]) from exc
        if manifest.get("expected_checkpoint_step") != step:
            raise CheckpointImportError(
                [
                    "Training manifest expected checkpoint step does not match trainer_state.json and the checkpoint directory."
                ]
            )
        if manifest.get("training_backend") != backend:
            raise CheckpointImportError(
                ["Training manifest backend does not match the checkpoint state format."]
            )
        dataset_meta = manifest.get("datasets") or {}
        bundle_root = checkpoint.parent if checkpoint.name.startswith("checkpoint-") else checkpoint
        snapshot = dataset_meta.get("snapshot")
        if snapshot:
            from core.training.portable_data import PortableDatasetError, verify_snapshot

            try:
                verify_snapshot(bundle_root, snapshot)
            except PortableDatasetError as exc:
                raise CheckpointImportError([str(exc)]) from exc
        for source in dataset_meta.get("bundled_sources") or []:
            relative = source.get("relative_path")
            if relative and not (bundle_root / relative).is_file() and not (
                bundle_root / relative
            ).is_dir():
                errors.append(f"Bundled dataset is missing: {relative}")
        if errors:
            raise CheckpointImportError(errors)
        adapter = _configuration_metadata(checkpoint).get("adapter_config.json", {})
        adapter_model = (
            adapter.get("base_model_name_or_path") if isinstance(adapter, dict) else None
        )
        manifest_model = manifest.get("base_model")
        if adapter_model and isinstance(manifest_model, str) and adapter_model != manifest_model:
            raise CheckpointImportError(
                ["Training manifest base model does not match adapter_config.json."]
            )
    return {
        "selected_checkpoint": str(checkpoint),
        "output_dir": str(selected if checkpoint == selected else checkpoint.parent),
        "global_step": step,
        "max_steps": _checkpoint_max_steps(checkpoint),
        "backend_type": backend,
        "configuration": _configuration_metadata(checkpoint),
        "manifest": manifest,
        "compatibility_warnings": []
        if manifest
        else ["No portable training manifest was found; compatibility checks are limited."],
    }


def validate_import_compatibility(
    info: dict[str, Any], request: Any, backend_type: str
) -> list[str]:
    errors: list[str] = []
    if info["backend_type"] != backend_type:
        errors.append(
            f"Checkpoint backend {info['backend_type']!r} is incompatible with active backend {backend_type!r}."
        )
    merged: dict[str, Any] = {}
    for value in info.get("configuration", {}).values():
        if isinstance(value, dict):
            merged.update(value)
    manifest = info.get("manifest") or {}
    manifest_model = manifest.get("base_model")
    requested_model = getattr(request, "model_name", None)
    if isinstance(manifest_model, str) and requested_model and manifest_model != requested_model:
        errors.append(
            f"Checkpoint model {manifest_model!r} does not match requested {requested_model!r}."
        )
    checkpoint_step = int(info.get("global_step") or 0)
    requested_max_steps = int(getattr(request, "max_steps", 0) or 0)
    checkpoint_max_steps = int(info.get("max_steps") or 0)
    effective_max_steps = max(requested_max_steps, checkpoint_max_steps)
    if effective_max_steps > 0 and checkpoint_step >= effective_max_steps:
        errors.append(
            f"Checkpoint step {checkpoint_step} has already reached Max Steps "
            f"({effective_max_steps}). Choose an earlier checkpoint or set Max Steps "
            f"higher than {checkpoint_step} before resuming."
        )
    warnings_list = list(info.get("compatibility_warnings") or [])
    for key in ("training_type", "load_in_4bit", "max_seq_length"):
        expected = merged.get(key)
        actual = getattr(request, key, None)
        if expected is not None and actual is not None and expected != actual:
            warnings_list.append(
                f"Checkpoint {key}={expected!r} differs from requested {actual!r}."
            )
    if errors:
        raise CheckpointImportError(errors)
    if warnings_list and not bool(getattr(request, "confirm_import_differences", False)):
        raise CheckpointImportError(
            [
                "Confirmation is required for noncritical checkpoint differences.",
                *warnings_list,
            ]
        )
    return warnings_list


def preserve_checkpoint_training_target(info: dict[str, Any], request: Any) -> None:
    """Do not shorten a resumed run below the target saved by TrainerState."""
    checkpoint_max_steps = int(info.get("max_steps") or 0)
    if checkpoint_max_steps > int(getattr(request, "max_steps", 0) or 0):
        request.max_steps = checkpoint_max_steps


def get_resume_checkpoint_path(
    path_value: str, expected_step: Optional[int] = None
) -> Optional[str]:
    # Changing the configured checkpoint root can leave historical runs
    # pointing at the previous root. History must remain browseable; such a run
    # is simply no longer resumable from the active storage location.
    try:
        path = resolve_output_dir(path_value)
    except ValueError:
        return None
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


def continuation_output_name(source_name: str, timestamp: str) -> str:
    """Return a stable continuation name across repeated checkpoint imports."""
    base_name = source_name
    prefix = "continuation_"
    # Generated directories end in YYYYMMDD_HHMMSS. Remove every generated
    # wrapper so a continuation of a continuation keeps the original run name.
    while base_name.startswith(prefix):
        candidate = base_name[len(prefix):]
        dated_stem, separator, time_part = candidate.rpartition("_")
        stem, date_separator, date_part = dated_stem.rpartition("_")
        if not (separator and date_separator and len(date_part) == 8 and len(time_part) == 6):
            break
        if not (date_part.isdigit() and time_part.isdigit()):
            break
        base_name = stem
    return f"{prefix}{base_name}_{timestamp}"


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
