# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
from transformers.models.llama.modeling_llama import logger
from packaging.version import Version

__all__ = [
    "causal_loss_function",
    "transformers_losses_patcher",
    "patch_loss_function",
]


def causal_loss_function(_fast_cross_entropy_loss):
    def UnslothForCausalLMLoss(
        logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
    ):
        shift_logits = logits
        shift_labels = torch.empty_like(labels)
        shift_labels[..., :-1] = labels[..., 1:]
        shift_labels[..., -1] = -100
        loss = _fast_cross_entropy_loss(
            logits  = shift_logits,
            labels  = shift_labels,
            n_items = num_items_in_batch,
        )
        return loss
    pass

    if (Version(torch.__version__) < Version("2.4.0")):
        UnslothForCausalLMLoss = torch._disable_dynamo(UnslothForCausalLMLoss)
    pass
    return UnslothForCausalLMLoss
pass


def transformers_losses_patcher(UnslothForCausalLMLoss):
    def _patch_transformers_losses():
        import re
        try:
            import transformers.loss.loss_utils
        except:
            logger.warning_once("Unsloth: Cannot patch loss functions - update transformers for faster modules!")
            return
        pass

        import transformers.modeling_utils
        LOSS_MAPPING = transformers.loss.loss_utils.LOSS_MAPPING
        LOSS_MAPPING["ForCausalLM"] = UnslothForCausalLMLoss

        # Remove @property and @lru_cache
        if hasattr(transformers.modeling_utils.PreTrainedModel.loss_function, "fget"):
            transformers.modeling_utils.PreTrainedModel.loss_function = \
                transformers.modeling_utils.PreTrainedModel.loss_function.fget.__wrapped__
        pass
    pass
    return _patch_transformers_losses
pass


def patch_loss_function(model):
    try:
        # model.loss_function starts as a dict to a loss fx
        # We invoke it to save it
        model.loss_function = model.loss_function()
    except:
        # Failed means we already invoked it, and we need args to the loss fx
        pass
    pass
    return model
pass
