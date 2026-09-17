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

# Deliberately loose: rows are consumed by things that never produce a step (the eval split, rows
# train_on_responses_only drops when the response template is missing), and running short only
# re-reads the subset; 4x is still orders of magnitude under the datasets this exists for.
MAX_STEPS_ROW_SLACK = 4
# Below this a subset is small enough to skew a run for no meaningful saving.
MIN_MAX_STEPS_ROWS = 1024
# Read as env because this module is torch-free and the bound is computed before any process group
# exists, so torch.distributed cannot be asked. A bare size is enough here, unlike
# routes/inference.py, which pairs each size with its rank partner because it refuses requests.
WORLD_SIZE_ENV_VARS = (
    "WORLD_SIZE",  # torchrun, accelerate launch, deepspeed. NOT set by any MPI
    "LOCAL_WORLD_SIZE",  # torchrun's --nproc-per-node; it sets WORLD_SIZE too
    "MLX_WORLD_SIZE",  # only mlx.launch's NCCL backend, which is CUDA-only
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",  # MPICH and Intel MPI via Hydra; srun only under --mpi=pmi2
    "PMIX_SIZE",  # nothing sets this: PMIx answers job size through PMIx_Get
    "MPI_WORLD_SIZE",  # likewise undocumented in every MPI checked
    "MV2_COMM_WORLD_SIZE",  # MVAPICH2, and only under its mpirun_rsh launcher
)
# Of mlx.launch's five backends only NCCL exports MLX_WORLD_SIZE; ring and JACCL export a path to
# a JSON file whose outer list has one entry per rank.
WORLD_SIZE_ENV_FILES = (
    "MLX_HOSTFILE",  # ring backend: a path to [["ip:port", ...], ...], one per rank
    "MLX_IBV_DEVICES",  # jaccl backend: a path to the N x N RDMA matrix, one row per rank
)
# Read a bounded prefix so a wrong path cannot pull an enormous file into memory; a truncated read
# fails to parse as JSON and is discarded.
MAX_WORLD_SIZE_FILE_BYTES = 1 << 20
# Its absence is the signal that a checkpoint predates the bound.
ROW_BOUND_MARKER_FILE = "unsloth_row_bound.json"
# transformers writes checkpoint-<global_step> and nothing else under that prefix.
_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-\d+$")


def _int_or(value: Any, default: int) -> int:
    """Coerce a config value to an int; a row bound must never be what raises."""
    try:
        # OverflowError: json accepts Infinity, so a config or request can carry one.
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
    """Ranks an mlx.launch listed in a hostfile, or 1 when there is no readable one.

    Either representation the rest of the repo accepts: the payload inline in the
    variable, or a path to a file holding it. `unsloth_cli/_inference.py`'s
    `_json_rank_count_from_env` reads the same two variables the same way, down to the
    {"hosts": [...]} object form, so the two must not disagree about how many ranks a
    launch has.

    Only a list of ranks counts, and its length is the count. Anything else -- no such
    file, a truncated or malformed payload, some other object, an empty ring hostfile
    (which is what mlx.launch writes for a single host) -- reads as 1, the count of
    Unsloth's own launch. Never raises: a row bound must not be what fails a run.

    A path must name a regular file. mlx.launch writes a temp file, and opening
    whatever else a variable happens to name could block a run forever on a fifo.
    """
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
                # Binary, so the cap really is bytes: a text read() counts CHARACTERS, and json.loads takes bytes
                # anyway.
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
    """Data-parallel processes the launcher advertises, or 1 when none does.

    The largest wins: a torchrun launch sets WORLD_SIZE and LOCAL_WORLD_SIZE, and on
    one node they agree, while a multi-node one must be sized by the global count.
    Anything unusable (unset, empty, a stray "auto", 0, negative) reads as 1, which
    is the count Unsloth's own single-process launch has.

    Some launchers advertise the count as a file rather than a number; see
    WORLD_SIZE_ENV_FILES.
    """
    source = os.environ if environ is None else environ
    numbers = max(_positive_int(source.get(name), 1) for name in WORLD_SIZE_ENV_VARS)
    return max(numbers, world_size_from_rank_files(source))


def world_size_env_report(environ: Any = None) -> str:
    """The launcher variables that are set, for a log line. Never raises.

    Which variable claimed the rank count is the only thing a user can act on when
    a run on one machine is told it makes several passes. mpirun, srun and some
    container images leave a size variable behind, and a stale one reads as a
    multi-rank launch here exactly as it does in the row bound.

    Values are truncated: MLX_HOSTFILE legitimately carries a whole JSON payload.
    """
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
    """Rows a max_steps run can reach, or None when it is unbounded.

    A step draws batch_size * gradient_accumulation_steps rows on every data-parallel
    replica, so world_size times that in total: DDP hands each rank its own shard of
    the step, and DataParallel splits the batch over the visible devices. Leaving the
    factor out spends the whole slack on rank count alone, and from four replicas up
    a run re-reads rows it has already trained on.

    world_size is what the caller established (the CUDA worker also counts visible
    CUDA devices, which env cannot report); anything unusable falls back to the
    launcher env, and that falls back to 1, which is Unsloth's own launch.
    """
    steps = _positive_int(max_steps, 0)
    if steps <= 0:
        return None
    replicas = _positive_int(world_size, 0) or world_size_from_env()
    per_step = _positive_int(batch_size, 1) * _positive_int(gradient_accumulation_steps, 1)
    return max(MIN_MAX_STEPS_ROWS, steps * per_step * replicas * MAX_STEPS_ROW_SLACK)


def effective_packing(config: dict, branch_never_packs: bool = False) -> bool:
    """Whether the trainer will actually pack, not merely what was requested.

    Packing opts the bound out, since one packed sample spans an unknown number of
    source rows. The requested value is the answer unless the caller establishes
    that this run's branch never packs: the vision and audio-VLM branches, and
    every audio codec, train on a Trainer with no packing argument.

    Do NOT pass the client-supplied dataset flags: `is_dataset_image` /
    `is_dataset_audio` are true on a column-NAME match, so a text model with an
    "audio" column carries the flag yet trains on the text path, which packs. Pass
    the branch the model probe detected. The branches differ on raw-text and CPT:
    vision is gated on `not raw_text_mode`, while audio preprocessing is chosen
    before the raw-text bypass and so holds either way.
    """
    if not config.get("packing", False):
        return False
    return not branch_never_packs


def max_train_rows_for_config(
    config: dict,
    branch_never_packs: bool = False,
    *,
    world_size: Any = None,
) -> Optional[int]:
    """The bound for a worker config, or None when the run is not bounded.

    Streaming and an explicit train-split range opt out further down, in the
    loaders, where those values live.

    world_size is not read from the config: it belongs to the launch, not to what
    the user configured, and a stale one carried across a spawn would size the
    subset for the wrong machine.
    """
    if effective_packing(config, branch_never_packs = branch_never_packs):
        return None
    return max_steps_dataset_rows(
        config.get("max_steps", 0) or 0,
        config.get("batch_size", 2),
        config.get("gradient_accumulation_steps", 4),
        world_size = world_size,
    )


def run_dir_for_checkpoint(checkpoint_path: Any) -> Optional[str]:
    """The run directory a checkpoint lives in, or None when there is none.

    Only ``<output_dir>/checkpoint-<global_step>`` counts. Matching the bare
    prefix would take the parent of a RUN directory that happens to start with it,
    writing the marker one level above where a resume looks. A caller that names
    the run directory itself gets it back unchanged.
    """
    if not checkpoint_path:
        return None
    path = str(checkpoint_path).rstrip("/\\")
    if not path:
        return None
    head, tail = os.path.split(path)
    if _CHECKPOINT_DIR_RE.match(tail):
        # A bare "checkpoint-30" splits to an empty head; its run dir is the cwd.
        return head or os.curdir
    return path


def record_row_bound(
    output_dir: Any,
    max_train_rows: Optional[int],
    seed: Any = 3407,
) -> bool:
    """Record the bound a run started with, beside its checkpoints.

    The subset is training state: fixed at the first start and read back on every
    resume, because both loaders fast-forward to a batch *index* and the ordering
    is a function of the bound. Re-deriving on resume cannot work, since the config
    is editable between runs and a pre-feature checkpoint is indistinguishable.

    Best effort, and it reports whether it succeeded so the caller can say so: a
    run must never fail over a marker, and the dataset is already bounded by now,
    so an unwritable marker only costs a later resume reading the run as unbounded.

    Written to a temp file and os.replace'd (atomic on POSIX and Windows) because a
    resume rewrites an already valid marker: truncating in place then failing (a
    full disk) would leave an empty file, read as "no marker".
    """
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
    """The (rows, seed) a resume must use, or the freshly computed pair.

    Not resuming: the caller's own values, which record_row_bound then pins.

    Resuming a marked run: that run's values, so the rows and their order match
    what it trained on, whatever the config now says.

    Resuming with no readable marker: no bound. Such a checkpoint trained on the
    whole corpus in its natural order, and both trainers resume by batch index
    rather than by remembering rows (HF Trainer replays the current dataloader,
    `ignore_data_skip` defaults to False; MLXTrainer jumps a cursor into a schedule
    rebuilt from the current dataset), so a subset would continue on unrelated rows.
    """
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
    """Cut a map-style dataset to max_train_rows rows, or return it untouched.

    Shuffled, not the head: a corpus ordered by source or difficulty would
    otherwise make a short run train on one homogeneous slab. shuffle() only builds
    an indices mapping.

    Callers apply this before the formatting, template and tokenization passes,
    which map over every row: that is the cost this avoids.
    """
    if not max_train_rows or max_train_rows <= 0:
        return dataset
    # A DatasetDict answers len() with its split count, so guard on ops, not type.
    if not hasattr(dataset, "shuffle") or not hasattr(dataset, "select"):
        return dataset
    try:
        total_rows = len(dataset)
    except TypeError:
        # No __len__ means streaming, which is bounded lazily instead.
        return dataset
    if total_rows <= max_train_rows:
        return dataset
    bounded = dataset.shuffle(seed = _seed_int(seed, 3407)).select(range(max_train_rows))
    if on_bound is not None:
        on_bound(max_train_rows, total_rows)
    return bounded
