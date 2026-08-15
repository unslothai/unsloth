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
    the combination directly, and the image, audio-codec and audio-VLM branches
    all train without packing whatever the config says.
    """
    if not config.get("packing", False):
        return False
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


def checkpoint_predates_row_bound(
    checkpoint_path: Any, max_train_rows: Optional[int], config: dict
) -> bool:
    """Whether a checkpoint was written against a much larger dataset.

    The subset is part of training state now. Resuming a run that was started
    with the same bound continues exactly, because the bound is a function of the
    seed, max_steps, batch size and accumulation. A checkpoint written before the
    bound existed saw the whole corpus, and Trainer fast-forwards by batch count
    over the *current* dataloader (ignore_data_skip defaults to False), so
    bounding it now would resume into unrelated rows.

    trainer_state.json records global_step and a fractional epoch, and epoch is
    rows_seen / dataset_rows, so the dataset the checkpoint trained on can be
    recovered. Anything unreadable answers False: an unresumable checkpoint is
    the resume path's problem, not this one's.
    """
    if not checkpoint_path or not max_train_rows:
        return False
    state_file = os.path.join(str(checkpoint_path), "trainer_state.json")
    try:
        with open(state_file, encoding = "utf-8") as handle:
            state = json.load(handle)
        step = float(state["global_step"])
        epoch = float(state["epoch"])
    except (OSError, UnicodeDecodeError, ValueError, TypeError, KeyError):
        return False
    if step <= 0 or epoch <= 0 or step != step or epoch != epoch:
        return False
    # The checkpoint's own batch size when it recorded one, since a resume may
    # carry a different one. Accumulation is not in the state, so the current
    # config answers for it; changing it on resume already makes the trainer's
    # own fast-forward arithmetic unreliable, and reading it wrong here only
    # drops the bound, which is the pre-existing behaviour.
    per_step = _positive_int(
        state.get("train_batch_size"), _positive_int(config.get("batch_size"), 1)
    ) * _positive_int(config.get("gradient_accumulation_steps"), 1)
    previous_rows = (step * per_step) / epoch
    # Twice the bound, so an eval carve or masked-out rows in the earlier leg
    # cannot read as a different dataset.
    return previous_rows > max_train_rows * 2


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
