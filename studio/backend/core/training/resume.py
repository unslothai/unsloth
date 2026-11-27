# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Helpers for validating resumable training outputs."""

import json
import os
import pickletools
import stat
import threading
import time
import zipfile
from collections import OrderedDict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Optional

from utils.paths import outputs_root, resolve_output_dir
from utils.scan_memo import memoized, scan_memo_active

# Inside scan_memo, verdicts are cached per unchanged file; resume start paths always read. Files changed
# within the settle window are not cached: a coarse fs clock can give a same-size rewrite equal stamps.
_STATE_CACHE: "OrderedDict[tuple, tuple[tuple, object]]" = OrderedDict()
_STATE_CACHE_MAX = 4096
_STATE_CACHE_SETTLE_NS = 2_000_000_000
_STATE_CACHE_LOCK = threading.Lock()


def _stat_key(st: os.stat_result) -> tuple:
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _cached_state(cache_id: tuple, key: tuple):
    with _STATE_CACHE_LOCK:
        hit = _STATE_CACHE.get(cache_id)
        if hit is None or hit[0] != key:
            return False, None
        _STATE_CACHE.move_to_end(cache_id)
        return True, hit[1]


def _store_state(cache_id: tuple, key: tuple, value) -> None:
    if time.time_ns() - max(key[3], key[4]) <= _STATE_CACHE_SETTLE_NS:
        return
    with _STATE_CACHE_LOCK:
        _STATE_CACHE[cache_id] = (key, value)
        _STATE_CACHE.move_to_end(cache_id)
        while len(_STATE_CACHE) > _STATE_CACHE_MAX:
            _STATE_CACHE.popitem(last = False)


def _is_foreign_absolute_path(path_value: str) -> bool:
    native = Path(path_value)
    return not native.is_absolute() and (
        PureWindowsPath(path_value).is_absolute() or PurePosixPath(path_value).is_absolute()
    )


def _is_under_outputs(path: Path) -> bool:
    try:
        resolved = path.resolve(strict = False)
        root = outputs_root().resolve(strict = False)
        resolved.relative_to(root)
        return True
    except (OSError, RuntimeError, ValueError):
        # Unrecognized state-file formats are not usable resume state.
        return False


def has_resume_state(path_value: Optional[str]) -> bool:
    if not path_value:
        return False
    # can_resume_run and the blocked reason both ask per listing row.
    return memoized(
        ("has_resume_state", path_value),
        lambda: get_resume_checkpoint_path(path_value) is not None,
    )


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


def _read_state_file(path: Path, require_tensor: bool, io_error: Callable[[], None]) -> bool:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            # Caller saw a non-empty file, so False here is a swallowed stat error or a change: not a verdict.
            io_error()
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
        return False
    except OSError:
        io_error()
        return False
    except (ValueError, zipfile.BadZipFile):
        return False


def _valid_state_file(path: Path, require_tensor: bool = True) -> bool:
    if not scan_memo_active():
        return _read_state_file(path, require_tensor, lambda: None)
    try:
        before = os.stat(path)
    except (OSError, ValueError):
        return _read_state_file(path, require_tensor, lambda: None)
    if not stat.S_ISREG(before.st_mode) or before.st_size == 0:
        return _read_state_file(path, require_tensor, lambda: None)
    cache_id = ("state_file", str(path), require_tensor)
    key = _stat_key(before)
    hit, value = _cached_state(cache_id, key)
    if hit:
        return value
    io_errors: list = []
    value = _read_state_file(path, require_tensor, lambda: io_errors.append(True))
    # An IO failure may be transient, and a file swapped mid-read is not the one stat named.
    if not io_errors:
        try:
            if _stat_key(os.stat(path)) == key:
                _store_state(cache_id, key, value)
        except (OSError, ValueError):
            pass
    return value


def session_eta_seconds(
    elapsed_seconds: Optional[float], step: int, start_step: int, total_steps: int
) -> Optional[float]:
    steps_done = step - start_step
    steps_remaining = total_steps - step
    if elapsed_seconds is None or steps_done <= 0 or steps_remaining <= 0:
        return None
    return (elapsed_seconds / steps_done) * steps_remaining


def _trainer_state_step(trainer_state: Path) -> object:
    """global_step of a trainer_state.json; raises OSError / UnicodeDecodeError like read_text."""
    cache_id = ("trainer_state", str(trainer_state)) if scan_memo_active() else None
    with open(trainer_state, encoding = "utf-8") as f:
        if cache_id is not None:
            # fstat names the file this handle reads, even if the path is replaced meanwhile.
            key = _stat_key(os.fstat(f.fileno()))
            hit, step = _cached_state(cache_id, key)
            if hit:
                return step
        text = f.read()
    try:
        state = json.loads(text)
        step = state.get("global_step") if isinstance(state, dict) else None
    except json.JSONDecodeError:
        step = None
    if cache_id is not None and (step is None or type(step) in (int, bool)):
        _store_state(cache_id, key, step)
    return step


def _checkpoint_state(path: Path) -> Optional[int]:
    try:
        step = _trainer_state_step(path / "trainer_state.json")
    except (OSError, UnicodeDecodeError):
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
    if step is None or (expected_step is not None and step != expected_step):
        return False
    if backend == "mlx":
        valid_bundle = _valid_state_file(path / "adapters.safetensors") and _valid_state_file(
            path / "optimizer_state.safetensors"
        )
    else:
        valid_bundle = (
            _has_model_state(path)
            # optimizer/scheduler state can be validly tensor-free (e.g. SGD without momentum).
            and _valid_state_file(path / "optimizer.pt", require_tensor = False)
            and _valid_state_file(path / "scheduler.pt", require_tensor = False)
        )
        if backend is None and not valid_bundle:
            valid_bundle = _valid_state_file(path / "adapters.safetensors") and _valid_state_file(
                path / "optimizer_state.safetensors"
            )
    return valid_bundle


def artifacts_present(path_value: Optional[str]) -> bool:
    if not path_value:
        return False
    if _is_foreign_absolute_path(path_value):
        return False
    try:
        path = resolve_output_dir(path_value)
        return _is_under_outputs(path) and path.is_dir()
    except (OSError, RuntimeError, ValueError):
        return False


def get_resume_checkpoint_path(
    path_value: str, expected_step: Optional[int] = None
) -> Optional[str]:
    if _is_foreign_absolute_path(path_value):
        return None
    try:
        path = resolve_output_dir(path_value)
    except (OSError, RuntimeError, ValueError):
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
    if _is_foreign_absolute_path(path_value):
        raise ValueError("Resume checkpoint uses a path from a different operating system.")
    try:
        path = resolve_output_dir(path_value)
        path.resolve(strict = True)
    except (OSError, RuntimeError) as error:
        raise ValueError("Resume checkpoint path could not be resolved.") from error
    if not _is_under_outputs(path):
        raise ValueError("Resume checkpoint must be inside Unsloth outputs.")
    return str(path)


def training_run_config(run: dict) -> dict:
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
    config = training_run_config(run)
    return config.get("dataset_source") == "s3" or "s3_dataset" in config


def _resource_resume_cache_key(config: dict) -> Optional[str]:
    marker = config.get("resource_provenance")
    if isinstance(marker, dict):
        marker = {
            key: marker.get(key)
            for key in (
                "version",
                "status",
                "model_status",
                "model_load_mode",
                "dataset_status",
            )
        }
    values = {
        "resource_provenance": marker,
        "model_name": config.get("model_name"),
        "actual_model_repo_id": config.get("actual_model_repo_id"),
        "model_snapshot_path": config.get("model_snapshot_path"),
        "load_in_4bit": config.get("load_in_4bit"),
        "resume_model_load_mode": config.get("resume_model_load_mode"),
        "hf_dataset": config.get("hf_dataset"),
        "dataset_snapshot_path": config.get("dataset_snapshot_path"),
    }
    try:
        return json.dumps(
            values,
            sort_keys = True,
            separators = (",", ":"),
            ensure_ascii = False,
        )
    except (TypeError, ValueError):
        return None


def resource_resume_blocker(config: dict) -> Optional[str]:
    """resource_provenance_resume_blocker, answered once per scan_memo scope for runs with the same inputs."""
    from core.training.provenance import resource_provenance_resume_blocker

    key = _resource_resume_cache_key(config)
    if key is None:
        return resource_provenance_resume_blocker(config)
    return memoized(("resume_blocker", key), lambda: resource_provenance_resume_blocker(config))


def can_resume_run(run: dict, *, resource_cache: Optional[dict[str, bool]] = None) -> bool:
    if run.get("resumed_later"):
        return False
    # Set when a stop-and-save failed to write a current-step checkpoint.
    if run.get("resume_blocked"):
        return False
    if _uses_s3_dataset(run):
        return False

    status = run.get("status")
    if status == "error":
        # Save-time crash: final_step == total_steps, no artifacts; checkpoints alone decide.
        resume_state_available = has_resume_state(run.get("output_dir"))
    else:
        final_step = run.get("final_step")
        total_steps = run.get("total_steps")
        has_remaining_steps = (
            not isinstance(final_step, int)
            or not isinstance(total_steps, int)
            or total_steps <= 0
            or final_step < total_steps
        )
        resume_state_available = (
            status == "stopped" and has_remaining_steps and has_resume_state(run.get("output_dir"))
        )
    if not resume_state_available:
        return False

    from core.training.provenance import resource_provenance_allows_resume

    config = training_run_config(run)
    cache_key = _resource_resume_cache_key(config)
    if resource_cache is None or cache_key is None:
        return resource_provenance_allows_resume(config)
    if cache_key not in resource_cache:
        resource_cache[cache_key] = resource_provenance_allows_resume(config)
    return resource_cache[cache_key]


def checked_resume_checkpoint(path_value):
    """The checkpoint to hand trainer.train, re-checked right before it: transformers joins each
    weight index name onto the folder as is, and the folder may have been edited since start."""
    if path_value:
        from utils.security.file_security import escaping_weight_references
        if escaping_weight_references(path_value):
            raise ValueError("This model's weight index names files outside its folder.")
    return path_value
