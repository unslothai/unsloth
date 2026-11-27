# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stop a run once eval loss stops improving, and keep the weights from the best evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

MAX_EARLY_STOPPING_PATIENCE = 1000
_EARLY_STOPPING_DIR = "studio_early_stopping"
_EARLY_STOPPING_STATE = "studio_early_stopping.safetensors"
# Past this the CPU copy of the best weights costs more host RAM than it is worth (a full finetune of a 1B model).
MAX_BEST_WEIGHTS_BYTES = 2 << 30


def early_stopping_patience(value: Any) -> int:
    """Patience in evaluations; anything unusable is 0, meaning off."""
    if isinstance(value, bool) or value is None:
        return 0
    try:
        patience = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    if not math.isfinite(patience) or patience < 1:
        return 0
    return min(int(patience), MAX_EARLY_STOPPING_PATIENCE)


def _shard_signature(model, trainable):
    from unsloth.models.ep_native import _layout, is_expert_parallel
    from unsloth.models.portable_resume import _ep_layout, _is_dtensor
    from unsloth.models.tp_native import _param_layouts

    tensor_layouts = _param_layouts(model)
    shards = {}
    for name, param in trainable.items():
        if _is_dtensor(param):
            shards[name] = (
                tuple(param.shape),
                param.device_mesh.mesh.tolist(),
                tuple(repr(placement) for placement in param.placements),
                tensor_layouts.get(id(param)),
            )
        elif (layout := _ep_layout(param)) is not None:
            shards[name] = layout
    expert = _layout(model) if is_expert_parallel(model) else None
    if not shards and expert is None:
        return None
    encoded = json.dumps((expert, shards), sort_keys = True, separators = (",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def create_early_stopping_callback(
    patience: int,
    user_stopped = None,
    resume_from_checkpoint = None,
    *,
    pipeline_parallel: bool = False,
    sharded: bool = False,
):
    """HF TrainerCallback: counts evaluations whose eval_loss is not below the best so far, stops the run after
    ``patience`` of them in a row, and, like load_best_model_at_end (and zoo's MLX trainer), copies the best
    evaluation's trainable weights back when the run finishes, so the final save holds them.
    Its own bookkeeping, not EarlyStoppingCallback, which needs metric_for_best_model and restores nothing without
    load_best_model_at_end, and that needs checkpoints Studio does not write by default."""
    from transformers import TrainerCallback

    if pipeline_parallel or sharded:
        from unsloth.models.fsdp2_save import _local_tensor, _reshard_units, fsdp2_units
        from unsloth.models.portable_resume import _agree
    sharded = sharded or pipeline_parallel

    class EvalLossEarlyStopping(TrainerCallback):
        def __init__(self):
            self.patience = patience
            self.best_loss: Optional[float] = None
            self.best_step: Optional[int] = None
            self.bad_evals = 0
            self.stopped_at: Optional[int] = None
            self.restored = False
            self._best_weights: Optional[dict] = None
            self._best_layout: Optional[str] = None
            self._too_large = False

        def _rank_path(self, args):
            return f"rank-{args.process_index}-of-{args.world_size}.safetensors"

        def on_train_begin(
            self,
            args,
            state,
            control,
            model = None,
            **kwargs,
        ):
            if not resume_from_checkpoint:
                return
            folder = Path(resume_from_checkpoint)
            source = folder / _EARLY_STOPPING_DIR / _EARLY_STOPPING_STATE
            if not source.exists():
                source = folder / _EARLY_STOPPING_STATE
            snapshot = self._read_checkpoint(source, state.global_step, read_weights = not sharded)
            if sharded:
                rank_snapshot = self._read_checkpoint(
                    folder / _EARLY_STOPPING_DIR / self._rank_path(args), state.global_step
                )
                # Old pipeline checkpoints placed patience in the rank file only.
                if not _agree(snapshot is not None):
                    if not _agree(rank_snapshot is not None):
                        return
                    snapshot = rank_snapshot
            if snapshot is None:
                return

            progress, weights, _ = snapshot
            self.best_step, self.best_loss, self.bad_evals, self._too_large = progress
            if sharded:
                self._restore_rank_weights(model, rank_snapshot, progress)
            else:
                self._best_weights = weights

        def _restore_rank_weights(self, model, snapshot, progress):
            trainable = self._trainable_parameters(model)
            layout = _shard_signature(model, trainable)
            weights = None
            self._too_large = False
            if snapshot is not None:
                local_progress, local_weights, saved_layout = snapshot
                # Before layouts were recorded, only plain pipeline stages wrote ranked snapshots.
                same_layout = saved_layout == (layout or "whole-parameters") or (
                    saved_layout is None and layout is None and pipeline_parallel
                )
                if same_layout and local_progress[:3] == progress[:3]:
                    weights = local_weights if trainable else {}
                    self._too_large = local_progress[3]
            if _agree(
                weights is not None
                and {name: _local_tensor(param).shape for name, param in trainable.items()}
                == {name: value.shape for name, value in weights.items()}
            ):
                self._best_weights = weights
                self._best_layout = layout

        def _read_checkpoint(
            self,
            source,
            step,
            *,
            read_weights = True,
        ):
            if not source.is_file() or source.is_symlink():
                return None
            try:
                from safetensors import safe_open
                with safe_open(source, framework = "pt", device = "cpu") as saved:
                    meta = saved.metadata() or {}
                    if int(meta["checkpoint_step"]) != step:
                        return None
                    best_step = int(meta["best_step"]) if meta["best_step"] else None
                    best_loss = float(meta["best_loss"]) if meta["best_loss"] else None
                    bad_evals = int(meta["bad_evals"])
                    too_large = meta["too_large"] == "1"
                    if not 0 <= bad_evals or (best_step is not None and not 0 <= best_step <= step):
                        raise ValueError("invalid early-stopping progress")
                    if best_loss is not None and not math.isfinite(best_loss):
                        raise ValueError("invalid early-stopping loss")
                    # Bound restored tensors before allocating them.
                    weights = None
                    if (
                        read_weights
                        and source.stat().st_size <= MAX_BEST_WEIGHTS_BYTES + 16 * 1024 * 1024
                    ):
                        weights = {name: saved.get_tensor(name) for name in saved.keys()} or None
                return (best_step, best_loss, bad_evals, too_large), weights, meta.get("layout")
            except Exception as exc:
                logger.warning(f"Could not restore early-stopping progress: {exc}")
                return None

        def save_checkpoint(self, args, state):
            if not sharded and not state.is_world_process_zero:
                return
            step = state.global_step
            folder = Path(args.output_dir) / f"checkpoint-{step}"
            if not folder.is_dir():
                return
            directory = folder / _EARLY_STOPPING_DIR
            metadata = {
                "checkpoint_step": str(step),
                "best_step": "" if self.best_step is None else str(self.best_step),
                "best_loss": "" if self.best_loss is None else str(self.best_loss),
                "bad_evals": str(self.bad_evals),
                "too_large": "1" if self._too_large else "0",
            }
            if state.is_world_process_zero:
                self._write_checkpoint(
                    directory / _EARLY_STOPPING_STATE,
                    metadata,
                    None if sharded else self._best_weights,
                )
            if sharded:
                metadata["layout"] = self._best_layout or "whole-parameters"
                self._write_checkpoint(
                    directory / self._rank_path(args), metadata, self._best_weights
                )

        def _write_checkpoint(self, destination, metadata, weights):
            pending = None
            try:
                from safetensors.torch import save_file

                destination.parent.mkdir(exist_ok = True)
                with tempfile.NamedTemporaryFile(
                    dir = destination.parent, suffix = ".tmp", delete = False
                ) as file:
                    pending = Path(file.name)
                if weights:
                    try:
                        save_file(
                            {name: value.contiguous() for name, value in weights.items()},
                            pending,
                            metadata = metadata,
                        )
                    except Exception as exc:
                        logger.warning(
                            f"Could not save the best weights, saving progress only: {exc}"
                        )
                        save_file({}, pending, metadata = metadata)
                else:
                    save_file({}, pending, metadata = metadata)
                os.replace(pending, destination)
            except Exception as exc:
                logger.warning(f"Could not save early-stopping progress: {exc}")
                if pending is not None:
                    try:
                        pending.unlink(missing_ok = True)
                    except OSError:
                        pass

        def on_save(self, args, state, control, **kwargs):
            self.save_checkpoint(args, state)

        def on_evaluate(
            self,
            args,
            state,
            control,
            metrics = None,
            model = None,
            **kwargs,
        ):
            loss = (metrics or {}).get("eval_loss")
            if loss is None:
                return
            loss = float(loss)
            # A diverged (NaN / inf) eval counts against patience and never becomes the best.
            if math.isfinite(loss) and (self.best_loss is None or loss < self.best_loss):
                self.best_loss = loss
                self.best_step = state.global_step
                self.bad_evals = 0
                self._keep_best_weights(model)
                return
            self.bad_evals += 1
            # At the last step the run ends anyway; calling that an early stop would misreport it.
            max_steps = getattr(state, "max_steps", 0)
            steps_left = not max_steps or state.global_step < max_steps
            if self.bad_evals >= self.patience and self.stopped_at is None and steps_left:
                self.stopped_at = state.global_step
                control.should_training_stop = True
                logger.info(
                    f"Early stopping at step {state.global_step}: eval loss has not improved for "
                    f"{self.bad_evals} evaluations (best {self.best_loss} at step {self.best_step})\n"
                )

        def on_train_end(
            self,
            args,
            state,
            control,
            model = None,
            **kwargs,
        ):
            # User stops never restore, even at patience's limit; optimizer state is current.
            user_stop = user_stopped is not None and user_stopped()
            if sharded:
                user_stop = not _agree(not user_stop)
            if user_stop:
                # Studio's manual stop writes its resume checkpoint after on_train_end.
                return
            best, self._best_weights = self._best_weights, None
            finished = self.stopped_at is not None or state.global_step >= state.max_steps
            can_restore = (
                finished
                and self.best_step != state.global_step
                and best is not None
                and model is not None
            )
            if not (_agree(can_restore) if sharded else can_restore):
                return
            try:
                import torch

                # Check every shape before touching the model, so a mismatch leaves the final weights whole. copy_
                # casts and moves one tensor at a time, so nothing is staged on the GPU.
                pairs = [
                    (_local_tensor(param) if sharded else param, best[name])
                    for name, param in self._trainable_parameters(model).items()
                    if name in best
                ]
                fits = len(pairs) == len(best) and all(
                    param.shape == saved.shape for param, saved in pairs
                )
                if not (_agree(fits) if sharded else fits):
                    raise ValueError("a saved tensor no longer matches its parameter's shape")
                with torch.no_grad():
                    for param, saved in pairs:
                        param.copy_(saved)
                self.restored = True
                logger.info(
                    f"Restored the weights from step {self.best_step} (eval loss {self.best_loss:.4f})\n"
                )
            except Exception as exc:
                logger.warning(f"Could not restore the best weights, keeping the final ones: {exc}")

        def _trainable_parameters(self, model):
            if sharded:
                # FSDP2 can leave full parameters registered after evaluation.
                _reshard_units(fsdp2_units(model))
            return {name: param for name, param in model.named_parameters() if param.requires_grad}

        def _keep_best_weights(self, model):
            if model is None or self._too_large:
                return
            try:
                trainable = self._trainable_parameters(model)
                self._best_layout = _shard_signature(model, trainable) if sharded else None
                if sharded:
                    trainable = {name: _local_tensor(param) for name, param in trainable.items()}
                size = sum(param.numel() * param.element_size() for param in trainable.values())
                if size > MAX_BEST_WEIGHTS_BYTES:
                    self._too_large = True
                    self._best_weights = None
                    logger.info(
                        f"Early stopping keeps the final weights: {size / 2**30:.1f} GB of trainable "
                        "weights is too much to hold a copy of the best ones\n"
                    )
                    return
                # Refill the previous copy in place: building a new dict first would hold two copies at once.
                if self._best_weights is not None and self._best_weights.keys() == trainable.keys():
                    for n, p in trainable.items():
                        self._best_weights[n].copy_(p.detach())
                else:
                    self._best_weights = None
                    self._best_weights = {
                        n: p.detach().to("cpu", copy = True) for n, p in trainable.items()
                    }
            except Exception as exc:
                self._too_large = True
                self._best_weights = None
                logger.warning(f"Could not keep a copy of the best weights: {exc}")

    return EvalLossEarlyStopping()


def attach_early_stopping(
    trainer,
    value: Any,
    user_stopped = None,
    resume_from_checkpoint = None,
):
    patience = early_stopping_patience(value)
    if patience == 0:
        return None
    strategy = getattr(getattr(trainer, "args", None), "eval_strategy", "no")
    if (
        getattr(trainer, "eval_dataset", None) is None
        or str(getattr(strategy, "value", strategy)) == "no"
    ):
        logger.info("Early stopping is off: this run has no evaluation to watch\n")
        return None
    pipeline_parallel = getattr(trainer, "_unsloth_pipeline_parallel", None) is not None
    sharded = pipeline_parallel
    if not sharded and getattr(trainer.args, "world_size", 1) > 1:
        from unsloth.models.portable_resume import _sharded_trainer
        sharded = _sharded_trainer(trainer)
    callback = create_early_stopping_callback(
        patience,
        user_stopped = user_stopped,
        resume_from_checkpoint = resume_from_checkpoint,
        pipeline_parallel = pipeline_parallel,
        sharded = sharded,
    )
    trainer.add_callback(callback)
    logger.info(f"Early stopping on: patience of {patience} evaluations\n")
    return callback
