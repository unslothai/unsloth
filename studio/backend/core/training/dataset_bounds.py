# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Row bound for a max_steps run.

TRL prepares the whole train_dataset in the SFTTrainer constructor and never looks
at max_steps, so a 30-step run over a large corpus tokenizes millions of rows to
read a few hundred. The count is known before any of that work happens.

This module holds no torch and no unsloth imports: both loaders use it, and the
MLX one runs on hosts where importing core.training.trainer would drag in a torch
stack that need not exist.
"""

import json
import os
import re
import tempfile
from typing import Any, Optional

# Loose on purpose: some rows never produce a step (eval split, train_on_responses_only drops).
MAX_STEPS_ROW_SLACK = 4
MIN_MAX_STEPS_ROWS = 1024
# Env, not torch.distributed: this module is torch-free and runs before any process group.
WORLD_SIZE_ENV_VARS = (
    "WORLD_SIZE",  # torchrun, accelerate launch, deepspeed; not set by any MPI
    "LOCAL_WORLD_SIZE",
    "MLX_WORLD_SIZE",  # only mlx.launch's NCCL backend
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",  # MPICH and Intel MPI via Hydra; srun only under --mpi=pmi2
    "PMIX_SIZE",
    "MPI_WORLD_SIZE",
    "MV2_COMM_WORLD_SIZE",  # MVAPICH2, only under mpirun_rsh
)
# mlx.launch ring and JACCL backends export a path to a JSON list with one entry per rank.
WORLD_SIZE_ENV_FILES = (
    "MLX_HOSTFILE",
    "MLX_IBV_DEVICES",
)
MAX_WORLD_SIZE_FILE_BYTES = 1 << 20
# Its absence means the checkpoint predates the bound.
ROW_BOUND_MARKER_FILE = "unsloth_row_bound.json"
_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-\d+$")


def _int_or(value: Any, default: int) -> int:
    """Coerce a config value to an int; a row bound must never be what raises."""
    try:
        # OverflowError: json accepts Infinity.
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _positive_int(value: Any, default: int) -> int:
    """_int_or for counts, where zero and negatives are unusable."""
    number = _int_or(value, default)
    return number if number > 0 else default


def _seed_int(value: Any, default: int) -> int:
    """_int_or for seeds, where 0 is legitimate but numpy rejects negatives."""
    number = _int_or(value, default)
    return number if number >= 0 else default


def world_size_from_rank_files(environ: Any = None) -> int:
    """Only regular files are opened: opening a fifo named by a variable could block a run forever."""
    source = os.environ if environ is None else environ
    sizes = [1]
    for name in WORLD_SIZE_ENV_FILES:
        try:
            value = source.get(name)
            if not value:
                continue
            if value.lstrip()[:1] in ("[", "{"):
                payload = json.loads(value[:MAX_WORLD_SIZE_FILE_BYTES])
            elif os.path.isfile(value):
                # Binary so the cap counts bytes, not characters.
                with open(value, "rb") as handle:
                    payload = json.loads(handle.read(MAX_WORLD_SIZE_FILE_BYTES))
            else:
                continue
        except (OSError, UnicodeError, ValueError, TypeError, AttributeError):
            continue
        if isinstance(payload, dict):
            payload = payload.get("hosts")
        if isinstance(payload, list):
            sizes.append(len(payload))
    return max(sizes)


def world_size_from_env(environ: Any = None) -> int:
    """Takes the largest advertised count: a multi-node torchrun must be sized by the global WORLD_SIZE."""
    source = os.environ if environ is None else environ
    numbers = max(_positive_int(source.get(name), 1) for name in WORLD_SIZE_ENV_VARS)
    return max(numbers, world_size_from_rank_files(source))


def world_size_env_report(environ: Any = None) -> str:
    """Lists launcher variables that are set; a stale one makes a single-machine run look multi-rank."""
    source = os.environ if environ is None else environ
    parts = []
    for name in WORLD_SIZE_ENV_VARS + WORLD_SIZE_ENV_FILES:
        try:
            value = source.get(name)
        except Exception:  # noqa: BLE001 - a log line must not be what fails a run
            continue
        if value:
            parts.append(f"{name}={str(value)[:64]}")
    return ", ".join(parts) or "no launcher variable set"


def max_steps_dataset_rows(
    max_steps: Any,
    batch_size: Any,
    gradient_accumulation_steps: Any,
    *,
    world_size: Any = None,
) -> Optional[int]:
    """Each step draws batch_size * gradient_accumulation_steps rows per replica, so scale by world_size."""
    steps = _positive_int(max_steps, 0)
    if steps <= 0:
        return None
    replicas = _positive_int(world_size, 0) or world_size_from_env()
    per_step = _positive_int(batch_size, 1) * _positive_int(gradient_accumulation_steps, 1)
    return max(MIN_MAX_STEPS_ROWS, steps * per_step * replicas * MAX_STEPS_ROW_SLACK)


def effective_packing(config: dict, branch_never_packs: bool = False) -> bool:
    """Pass the branch the model probe detected, not the client dataset flags, which match column names."""
    if not config.get("packing", False):
        return False
    return not branch_never_packs


def max_train_rows_for_config(
    config: dict,
    branch_never_packs: bool = False,
    *,
    world_size: Any = None,
) -> Optional[int]:
    """world_size is not read from the config; a stale one from a spawn would size the wrong machine."""
    if effective_packing(config, branch_never_packs = branch_never_packs):
        return None
    return max_steps_dataset_rows(
        config.get("max_steps", 0) or 0,
        config.get("batch_size", 2),
        config.get("gradient_accumulation_steps", 4),
        world_size = world_size,
    )


def run_dir_for_checkpoint(checkpoint_path: Any) -> Optional[str]:
    """Only checkpoint-<global_step> counts; a bare checkpoint prefix would also match run directories."""
    if not checkpoint_path:
        return None
    path = str(checkpoint_path).rstrip("/\\")
    if not path:
        return None
    head, tail = os.path.split(path)
    if _CHECKPOINT_DIR_RE.match(tail):
        return head or os.curdir
    return path


def record_row_bound(
    output_dir: Any,
    max_train_rows: Optional[int],
    seed: Any = 3407,
) -> bool:
    """Written via temp file and os.replace, so a full disk mid-rewrite cannot leave an empty marker."""
    run_dir = run_dir_for_checkpoint(output_dir)
    if not run_dir:
        return False
    marker = os.path.join(run_dir, ROW_BOUND_MARKER_FILE)
    tmp_path = None
    try:
        payload = json.dumps(
            {
                "max_train_rows": _positive_int(max_train_rows, 0) or None,
                "seed": _seed_int(seed, 3407),
            }
        )
        handle, tmp_path = tempfile.mkstemp(dir = run_dir, prefix = ".row_bound_", suffix = ".tmp")
        with os.fdopen(handle, "w", encoding = "utf-8") as tmp_file:
            tmp_file.write(payload)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, marker)
        tmp_path = None
    except (OSError, UnicodeError, TypeError, ValueError):
        return False
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return True


def row_bound_for_resume(
    checkpoint_path: Any,
    max_train_rows: Optional[int],
    seed: Any = 3407,
) -> tuple[Optional[int], int]:
    """A resume with no readable marker gets no bound, as both trainers resume by batch index."""
    fallback_seed = _seed_int(seed, 3407)
    if not checkpoint_path:
        return max_train_rows, fallback_seed
    run_dir = run_dir_for_checkpoint(checkpoint_path)
    if not run_dir:
        return max_train_rows, fallback_seed
    try:
        with open(os.path.join(run_dir, ROW_BOUND_MARKER_FILE), encoding = "utf-8") as handle:
            marker = json.load(handle)
        recorded = marker["max_train_rows"]
    except (OSError, UnicodeDecodeError, ValueError, TypeError, KeyError):
        return None, fallback_seed
    return _positive_int(recorded, 0) or None, _seed_int(marker.get("seed"), fallback_seed)


def bound_dataset_rows(
    dataset,
    max_train_rows: Optional[int],
    seed: Any = 3407,
    *,
    on_bound = None,
):
    """Shuffled, not the head, since an ordered corpus would train one slab; apply before formatting."""
    if not max_train_rows or max_train_rows <= 0:
        return dataset
    # A DatasetDict answers len() with its split count, so guard on ops, not type.
    if not hasattr(dataset, "shuffle") or not hasattr(dataset, "select"):
        return dataset
    try:
        total_rows = len(dataset)
    except TypeError:
        return dataset
    if total_rows <= max_train_rows:
        return dataset
    bounded = dataset.shuffle(seed = _seed_int(seed, 3407)).select(range(max_train_rows))
    if on_bound is not None:
        on_bound(max_train_rows, total_rows)
    return bounded
