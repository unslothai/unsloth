# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
from packaging.version import Version
import os
torch_nn_functional_cross_entropy = torch.nn.functional.cross_entropy
from triton import __version__ as triton_version
major, minor = torch.cuda.get_device_capability()

global HAS_CUT_CROSS_ENTROPY
if (Version(torch.__version__) >= Version("2.4.0")) and \
    (not ((major <= 7) and (minor < 5))) and \
    (not (Version(triton_version) < Version("3.0.0"))):
    try:
        from cut_cross_entropy import linear_cross_entropy
        HAS_CUT_CROSS_ENTROPY = True
    except:
        HAS_CUT_CROSS_ENTROPY = False
else:
    HAS_CUT_CROSS_ENTROPY = False
pass

__all__ = [
    "patch_loss_functions",
    "post_patch_loss_function",
    "HAS_CUT_CROSS_ENTROPY",
    "fused_linear_cross_entropy",
]


def patch_loss_functions(_fast_cross_entropy_loss, torch_compile = True):
    try:
        import transformers.loss.loss_utils
    except:
        print("Unsloth: Cannot patch loss functions - update transformers for faster modules!")
        return None
    pass

    # Generic cross entropy loss
    def unsloth_fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
        if ignore_index == -100:
            loss = _fast_cross_entropy_loss(
                logits  = source,
                labels  = target,
                n_items = num_items_in_batch,
            )
        else:
            reduction = "sum" if num_items_in_batch is not None else "mean"
            loss = torch_nn_functional_cross_entropy(
                source,
                target,
                ignore_index = ignore_index,
                reduction    = reduction,
            )
            if reduction == "sum": loss = loss / num_items_in_batch
        return loss
    pass
    
    # Causal LM loss
    def UnslothForCausalLMLoss(
        logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
    ):
        shift_logits = logits
        shift_labels = torch.empty_like(labels)
        shift_labels[..., :-1] = labels[..., 1:]
        shift_labels[..., -1] = ignore_index
        loss = unsloth_fixed_cross_entropy(shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
        return loss
    pass

    if (Version(torch.__version__) < Version("2.4.0")):
        UnslothForCausalLMLoss = torch._disable_dynamo(UnslothForCausalLMLoss)
    
    elif torch_compile:
        torch_compile_options = {
            "epilogue_fusion"   : True,
            "max_autotune"      : True,
            "shape_padding"     : True,
            "trace.enabled"     : os.environ.get("UNSLOTH_COMPILE_DEBUG", "0") == "1",
            "triton.cudagraphs" : False,
        }
        UnslothForCausalLMLoss = torch.compile(
            UnslothForCausalLMLoss,
            dynamic = True,
            fullgraph = False,
            options = torch_compile_options,
        )
    pass

    # Now patch the losses!
    import transformers.modeling_utils
    LOSS_MAPPING = transformers.loss.loss_utils.LOSS_MAPPING
    LOSS_MAPPING["ForCausalLM"] = UnslothForCausalLMLoss

    # Remove @property and @lru_cache
    if hasattr(transformers.modeling_utils.PreTrainedModel.loss_function, "fget") and \
        hasattr(transformers.modeling_utils.PreTrainedModel.loss_function.fget, "__wrapped__"):
        transformers.modeling_utils.PreTrainedModel.loss_function = \
            transformers.modeling_utils.PreTrainedModel.loss_function.fget.__wrapped__
    pass
    print("Unsloth: Patched cross entropy losses.")
    os.environ["UNSLOTH_PATCHED"] = "1"
pass


def post_patch_loss_function(model):
    current_model = model
    while hasattr(current_model, "model"):
        try:
            # model.loss_function starts as a dict to a loss fx
            # We invoke it to save it
            current_model.loss_function = current_model.loss_function()
        except:
            # Failed means we already invoked it, and we need args to the loss fx
            pass
        pass
        current_model = current_model.model
    pass
    try: current_model.loss_function = current_model.loss_function()
    except: pass
    return model
pass


def fused_linear_cross_entropy(
    hidden_states      : torch.Tensor,
    lm_weight          : torch.Tensor,
    labels             : torch.Tensor,
    num_items_in_batch : int = None,
    ignore_index       : int = -100,
    reduction          : str = "mean",
    logit_softcapping  : float = 0,
    accuracy_threshold : str = "auto",
):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    if logit_softcapping == 0: logit_softcapping = None
    loss = linear_cross_entropy(
        hidden_states,
        lm_weight,
        targets      = labels,
        ignore_index = ignore_index,
        softcap      = logit_softcapping,
        reduction    = reduction,
        shift        = True,
        filter_eps   = accuracy_threshold,
    )
    if num_items_in_batch is not None: loss = loss / num_items_in_batch
    return loss
pass

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
