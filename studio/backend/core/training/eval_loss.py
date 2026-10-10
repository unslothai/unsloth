# SPDX-License-Identifier: AGPL-3.0-only
"""Keep evaluation loss independent of DDP gradient normalization."""

from functools import wraps
from types import MethodType

_LOCAL_EVAL_LOSS_WRAPPED = "_unsloth_local_eval_loss_wrapped"


def use_local_eval_loss(trainer):
    """Evaluate local mean losses; leave global token weighting enabled for training.

    Unsloth forces logits during prediction. Some compiled model forwards in that
    path return a local mean rather than consuming num_items_in_batch. Trainer's
    global-token compensation then multiplies that mean by the world size.
    Evaluation already gathers and averages losses and needs no DDP gradient
    compensation. Scope this change to prediction_step, including on exceptions.
    """
    if getattr(trainer, _LOCAL_EVAL_LOSS_WRAPPED, False):
        return

    original = trainer.prediction_step

    @wraps(original)
    def prediction_step(self, *args, **kwargs):
        previous = self.args.average_tokens_across_devices
        self.args.average_tokens_across_devices = False
        try:
            return original(*args, **kwargs)
        finally:
            self.args.average_tokens_across_devices = previous

    trainer.prediction_step = MethodType(prediction_step, trainer)
    setattr(trainer, _LOCAL_EVAL_LOSS_WRAPPED, True)
