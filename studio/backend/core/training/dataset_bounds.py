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

# Slack on the row bound. Rows are consumed by things that never produce a step:
# the eval split carved off the train set, and the rows train_on_responses_only
# drops when the response template is missing. Running short is not an error --
# max_steps just re-reads the subset -- but it trains on the same rows twice, so
# the bound is deliberately loose. Even 4x is three orders of magnitude under the
# datasets this exists for.
MAX_STEPS_ROW_SLACK = 4
# Below this a subset is small enough to skew a run for no meaningful saving.
MIN_MAX_STEPS_ROWS = 1024
# Written into a run's output directory at its first start; read back on resume.
# Its absence is the signal that a checkpoint predates the bound.
ROW_BOUND_MARKER_FILE = "unsloth_row_bound.json"
# transformers writes checkpoint-<global_step> and nothing else under that prefix.
_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-\d+$")


def _int_or(value: Any, default: int) -> int:
    """Coerce a config value to an int, falling back on anything unusable.

    Studio's request schema validates these, but the worker is also driven from
    the DB, from resumed-run records and by direct callers, any of which can hand
    over a None or a string. A row bound is an optimization; it must never be the
    thing that raises.
    """
    try:
        # OverflowError: json accepts Infinity without a flag, so a config column
        # or a request body can carry one.
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _positive_int(value: Any, default: int) -> int:
    """_int_or for counts, where zero and negatives are as unusable as None."""
    number = _int_or(value, default)
    return number if number > 0 else default


def _seed_int(value: Any, default: int) -> int:
    """_int_or for seeds, where 0 is legitimate but numpy rejects negatives."""
    number = _int_or(value, default)
    return number if number >= 0 else default


def max_steps_dataset_rows(
    max_steps: Any, batch_size: Any, gradient_accumulation_steps: Any
) -> Optional[int]:
    """Rows a max_steps run can reach, or None when it is unbounded.

    A step draws batch_size * gradient_accumulation_steps rows.
    """
    steps = _positive_int(max_steps, 0)
    if steps <= 0:
        return None
    per_step = _positive_int(batch_size, 1) * _positive_int(gradient_accumulation_steps, 1)
    return max(MIN_MAX_STEPS_ROWS, steps * per_step * MAX_STEPS_ROW_SLACK)


def effective_packing(config: dict, is_vlm: bool = False) -> bool:
    """Whether the trainer will actually pack, not merely what was requested.

    Packing opts the bound out because one packed sample spans an unknown number
    of source rows. The stored value alone overshoots: the frontend hides the
    packing control for image VLMs without resetting it, API clients can submit
    the combination directly, and the vision, audio-VLM and audio-codec branches
    all train without packing whatever the config says -- csm and snac on a plain
    HF Trainer, whisper on Seq2SeqTrainer, none of which take a packing argument,
    and bicodec/dac forced off on the SFTTrainer path.

    Raw-text and CPT are the exception to that: the vision/audio-VLM branch is
    gated on `not raw_text_mode`, so those runs take the text path and it honours
    the requested value however the dataset is flagged.
    """
    if not config.get("packing", False):
        return False
    raw_text_mode = (
        config.get("training_type") == "Continued Pretraining" or config.get("format_type") == "raw"
    )
    if raw_text_mode:
        return True
    if is_vlm or config.get("is_dataset_image", False) or config.get("is_dataset_audio", False):
        return False
    return True


def max_train_rows_for_config(config: dict, is_vlm: bool = False) -> Optional[int]:
    """The bound for a worker config, or None when the run is not bounded.

    Streaming and an explicit train-split range opt out further down, in the
    loaders, where those values live.
    """
    if effective_packing(config, is_vlm = is_vlm):
        return None
    return max_steps_dataset_rows(
        config.get("max_steps", 0) or 0,
        config.get("batch_size", 2),
        config.get("gradient_accumulation_steps", 4),
    )


def run_dir_for_checkpoint(checkpoint_path: Any) -> Optional[str]:
    """The run directory a checkpoint lives in, or None when there is none.

    Trainer writes ``<output_dir>/checkpoint-<global_step>`` (transformers'
    PREFIX_CHECKPOINT_DIR, always followed by the step number), so only that
    exact shape is a checkpoint. Matching the bare prefix would take the parent
    of a RUN directory that happens to start with it, and the marker would then
    be written one level above where a later resume looks for it. A caller that
    names the run directory itself gets it back unchanged.
    """
    if not checkpoint_path:
        return None
    path = str(checkpoint_path).rstrip("/\\")
    if not path:
        return None
    head, tail = os.path.split(path)
    if head and _CHECKPOINT_DIR_RE.match(tail):
        return head
    return path


def record_row_bound(
    output_dir: Any,
    max_train_rows: Optional[int],
    seed: Any = 3407,
) -> None:
    """Record the bound a run started with, beside its checkpoints.

    The subset a run trains on is training state: it has to be fixed at the first
    start and read back on every resume, because both loaders fast-forward to a
    batch *index* and the ordering is a function of the bound. Deriving it again
    on resume cannot work -- the config it is derived from is editable between
    runs, and a checkpoint written before this feature existed leaves no
    arithmetic that distinguishes it reliably.

    Best effort by design. A run must never fail over a marker; a marker that
    could not be written costs the optimization on the next resume and nothing
    else.

    Written through a temporary file and moved into place, because a resume
    rewrites a marker that is already valid: truncating in place and then failing
    -- a full disk is the ordinary way -- would leave an empty file, which reads
    as "no marker" and resumes the run over the whole dataset. os.replace is
    atomic on POSIX and on Windows.
    """
    run_dir = run_dir_for_checkpoint(output_dir)
    if not run_dir:
        return
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
        return
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def row_bound_for_resume(
    checkpoint_path: Any,
    max_train_rows: Optional[int],
    seed: Any = 3407,
) -> tuple[Optional[int], int]:
    """The (rows, seed) a resume must use, or the freshly computed pair.

    Not resuming: the caller's own values, which record_row_bound then pins.

    Resuming a run recorded by record_row_bound: that run's values, so the rows
    and their order are exactly the ones it was training on, whatever the config
    now says about max_steps, batch size or accumulation.

    Resuming anything with no marker -- a checkpoint written before the bound
    existed, or one whose marker is unreadable: no bound. Such a checkpoint
    trained on the whole corpus in its natural order, and both trainers resume by
    batch index rather than by remembering which rows they saw (HF Trainer
    replays the current dataloader, `ignore_data_skip` defaulting to False;
    unsloth_zoo's MLXTrainer jumps a cursor into a schedule rebuilt from the
    current dataset), so a shuffled subset would silently continue on unrelated
    rows.
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

    Shuffled, not the head. A corpus ordered by source or difficulty would
    otherwise make a short run train on one homogeneous slab. shuffle() builds an
    indices mapping; it does not rewrite the table.

    Callers apply this before the formatting, template and tokenization passes,
    all of which map over every row: that is the cost this avoids.
    """
    if not max_train_rows or max_train_rows <= 0:
        return dataset
    # A DatasetDict answers len() with its split count, so guard on the ops this
    # needs rather than on the type: anything else is left alone.
    if not hasattr(dataset, "shuffle") or not hasattr(dataset, "select"):
        return dataset
    try:
        total_rows = len(dataset)
    except TypeError:
        # No __len__ means a streaming dataset, which is bounded lazily instead.
        return dataset
    if total_rows <= max_train_rows:
        return dataset
    bounded = dataset.shuffle(seed = _seed_int(seed, 3407)).select(range(max_train_rows))
    if on_bound is not None:
        on_bound(max_train_rows, total_rows)
    return bounded
