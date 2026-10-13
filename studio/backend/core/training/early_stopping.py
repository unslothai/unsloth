# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stop a run once eval loss stops improving, and keep the weights from the best evaluation."""

from __future__ import annotations

import math
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

MAX_EARLY_STOPPING_PATIENCE = 1000
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


def create_early_stopping_callback(patience: int):
    """HF TrainerCallback: counts evaluations whose eval_loss is not below the best so far, stops the run after
    ``patience`` of them in a row, and copies the best evaluation's trainable weights back at train end whenever
    the last evaluation was worse, so the final save holds the best ones.
    Its own bookkeeping, not EarlyStoppingCallback, which needs metric_for_best_model and restores nothing without
    load_best_model_at_end, and that needs checkpoints Studio does not write by default."""
    from transformers import TrainerCallback

    class EvalLossEarlyStopping(TrainerCallback):
        def __init__(self):
            self.patience = patience
            self.best_loss: Optional[float] = None
            self.best_step: Optional[int] = None
            self.bad_evals = 0
            self.stopped_at: Optional[int] = None
            self.restored = False
            self._best_weights: Optional[dict] = None
            self._too_large = False

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
            if self.bad_evals >= self.patience and self.stopped_at is None:
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
            # bad_evals == 0 means the last evaluation was the best, so the final weights already are. A run that
            # ends on its own after worse evaluations restores too, as MLX's load_best_model_at_end does. A user
            # stop does not: stop-and-save writes a resume checkpoint, whose optimizer state is the current step's.
            best, self._best_weights = self._best_weights, None
            finished = self.stopped_at is not None or state.global_step >= state.max_steps
            if not finished or self.bad_evals == 0 or best is None or model is None:
                return
            try:
                import torch

                # Convert every tensor before touching the model, so a failure leaves the final weights whole.
                pairs = [
                    (param, best[name].to(device = param.device, dtype = param.dtype))
                    for name, param in model.named_parameters()
                    if name in best
                ]
                with torch.no_grad():
                    for param, saved in pairs:
                        param.copy_(saved)
                self.restored = True
                logger.info(
                    f"Restored the weights from step {self.best_step} (eval loss {self.best_loss:.4f})\n"
                )
            except Exception as exc:
                logger.warning(f"Could not restore the best weights, keeping the final ones: {exc}")

        def _keep_best_weights(self, model):
            if model is None or self._too_large:
                return
            try:
                trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
                size = sum(p.numel() * p.element_size() for _, p in trainable)
                if size > MAX_BEST_WEIGHTS_BYTES:
                    self._too_large = True
                    self._best_weights = None
                    logger.info(
                        f"Early stopping keeps the final weights: {size / 2**30:.1f} GB of trainable "
                        "weights is too much to hold a copy of the best ones\n"
                    )
                    return
                self._best_weights = {n: p.detach().to("cpu", copy = True) for n, p in trainable}
            except Exception as exc:
                self._too_large = True
                self._best_weights = None
                logger.warning(f"Could not keep a copy of the best weights: {exc}")

    return EvalLossEarlyStopping()


def attach_early_stopping(trainer, value: Any):
    """Add the callback to an HF trainer that evaluates; returns it, or None when off or there is nothing to watch."""
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
    callback = create_early_stopping_callback(patience)
    trainer.add_callback(callback)
    logger.info(f"Early stopping on: patience of {patience} evaluations\n")
    return callback
