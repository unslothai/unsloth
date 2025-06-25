# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
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

__all__ = [
    "RL_REPLACEMENTS"
]

import torch
import inspect
import os
import numpy as np
from typing import Union, Callable, Optional, List, Dict

RL_REPLACEMENTS = dict()

torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : False, # Disable Triton mm kernels
    "shape_padding"     : True,
    "trace.enabled"     : False,
    "triton.cudagraphs" : False,
}

# https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L1674
@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def selective_log_softmax(logits, index):
    logits = logits.to(torch.float32)
    selected_logits = torch.gather(logits, dim = -1, index = index.unsqueeze(-1)).squeeze(-1)
    # loop to reduce peak mem consumption
    # logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
    logsumexp_values = torch.logsumexp(logits, dim = -1)
    per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    return per_token_logps
pass
RL_REPLACEMENTS["selective_log_softmax"] = selective_log_softmax


# Custom compiled GRPO loss - creates 3 Triton kernels
def grpo_compute_loss(
    ref_logits,
    new_logits,
    old_logits,
    input_ids,
    mask,
    beta,
    advantages,
    **kwargs
):
    # All Unsloth Zoo code licensed under LGPLv3
    # Set defaults for optional arguments
    loss_type = kwargs.get("loss_type", "grpo")
    epsilon_low = kwargs.get("epsilon_low", 0.2)
    epsilon_high = kwargs.get("epsilon_high", 0.2)
    max_completion_length = kwargs.get("max_completion_length", 8192)
    delta = kwargs.get("delta", None)
    temperature = kwargs.get("temperature", 1.0)

    new_logits = new_logits.to(torch.float32)
    # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
    if temperature != 1.0:
        new_logits = new_logits / temperature
    input_ids  = input_ids.unsqueeze(-1)

    # x_i - logsumexp(x_i)
    with torch.no_grad():
        if beta != 0.0:
            assert ref_logits is not None, "ref_logits should not be None when beta != 0.0"
            ref_logits = ref_logits.to(torch.float32)
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            if temperature != 1.0:
                ref_logits = ref_logits / temperature
            ref_x = torch.gather(ref_logits, dim = -1, index = input_ids).squeeze(-1)
            ref = ref_x - torch.logsumexp(ref_logits, dim = -1)
        if old_logits is not None:
            old_logits = old_logits.to(torch.float32)
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            if temperature != 1.0:
                old_logits = old_logits / temperature
            old_x = torch.gather(old_logits, dim = -1, index = input_ids).squeeze(-1)
            old = old_x - torch.logsumexp(old_logits, dim = -1)


    new_x = torch.gather(new_logits, dim = -1, index = input_ids).squeeze(-1)
    new = new_x - torch.logsumexp(new_logits, dim = -1)

    # Reverse KL
    # Note that this is a low variance low bias estimator for the KL divergence as used in GRPO paper
    if beta != 0.0:
        kl_i = torch.exp(ref - new) - (ref - new) - 1.0

    else:
        kl_i = 0.0 # set it to 0 to not effect the downstream computation
    # Full correct reverse KL divergence?? Missing term maybe?
    # kl_i = torch.exp(new) * kl_i

    # Below is forward KL (normal KL)
    # kl_i = torch.exp(old) * (old - new)
    if old_logits is not None: 
        coef_1 = torch.exp(new - old)
    else:
        coef_1 = torch.exp(new - new.detach())
    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)

    if delta is not None:
        loss_1 = torch.clamp(coef_1, max=delta) * advantages.unsqueeze(1)
    else:
        loss_1 = coef_1 * advantages.unsqueeze(1)

    
    # Must detach - otherwise gradients are not propagated correctly!
    # exp(x - x) == 1
    # loss_i = torch.exp(new - new.detach()) * advantages.unsqueeze(1)
    

    loss_2 = coef_2 * advantages.unsqueeze(1)
    loss_i = -torch.min(loss_1, loss_2)
    if beta != 0.0:
        loss_i = loss_i + beta * kl_i

    mask = mask.to(torch.float32)
    n_mask_per_reward = mask.sum(1)

    # https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L1363-L1370
    if loss_type == "grpo":
        loss = ((loss_i * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    elif loss_type == "bnpo":
        loss = (loss_i * mask).sum() / mask.sum().clamp(min=1.0)
    elif loss_type == "dr_grpo":
        loss = (loss_i * mask).sum() / (loss_i.size(0) * max_completion_length)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # loss = (loss_i * mask).sum() / mask.sum()
    
    # Get metrics as well which are folded
    with torch.inference_mode():
        completion_length = n_mask_per_reward.mean()
        mean_kl_per_reward = (kl_i * mask).sum(1) / n_mask_per_reward
        mean_kl = mean_kl_per_reward.mean()
    pass

    return loss, completion_length, mean_kl
pass
RL_REPLACEMENTS["grpo_compute_loss"]      = grpo_compute_loss
RL_REPLACEMENTS["grpo_compute_loss_slow"] = \
    f"@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options)\n"\
    f"{inspect.getsource(grpo_compute_loss)}"
RL_REPLACEMENTS["grpo_compute_loss_slow"] = \
    RL_REPLACEMENTS["grpo_compute_loss_slow"].replace(
        "def grpo_compute_loss",
        "def grpo_compute_loss_slow",
)

# Unsloth's memory efficient GRPO implementation
class UnslothEfficientGRPO(torch.autograd.Function):
    # All Unsloth Zoo code licensed under LGPLv3
    @staticmethod
    def forward(ctx, _new_hidden_states, _old_hidden_states, _ref_hidden_states, lm_head, _input_ids, _mask, _advantages, beta, scaler = None, n_chunks = 1, extra_kwargs=None):
        if extra_kwargs is None:
            extra_kwargs = {}
        def compute_loss(new_hidden_states, old_hidden_states, ref_hidden_states, input_ids, mask, advantages, scaling):
            new_logits = torch.matmul(new_hidden_states, lm_head.t())
            new_logits = new_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred
            with torch.no_grad():
                if beta != 0.0:
                    ref_logits = torch.matmul(ref_hidden_states, lm_head.t())
                    ref_logits = ref_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred 
                else:
                    ref_logits = None
                if old_hidden_states is not None:
                    old_logits = torch.matmul(old_hidden_states, lm_head.t())
                    old_logits = old_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred 
                else: 
                    old_logits = None
            # if old_hidden_states is not None: 
            #     old_logits = torch.matmul(old_hidden_states, lm_head.t()) #last logit already excluded
            #     old_logits = old_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred 
            # else:
            #     old_logits = None
            # unsloth_zoo/rl_replacements.py
            loss, completion_length, mean_kl = grpo_compute_loss(
                ref_logits,
                new_logits,
                old_logits,
                input_ids,
                mask,
                beta,
                advantages,
                **extra_kwargs,
            )

            # Scale loss if needed for mixed precision training
            scaled_loss = loss * scaling
            # Must add .loss.detach otherwise autograd uses 2x VRAM
            return scaled_loss, (loss.detach(), completion_length, mean_kl,)
        pass

        device =_new_hidden_states.device
        grad_inputs = torch.empty_like(_new_hidden_states)
        accumulated_loss              = torch.zeros(1, device = device)
        accumulated_completion_length = torch.zeros(1, device = device)
        accumulated_mean_kl           = torch.zeros(1, device = device)

        def accumulate_chunk(new_hidden_states_j, old_hidden_states_j, ref_hidden_states_j, input_ids_j, mask_j, advantages_j, scaling):
            (chunk_grad_input,), (chunk_loss, (unscaled_loss, chunk_completion_length, chunk_mean_kl,)) = torch.func.grad_and_value(
                compute_loss,
                argnums = (0,),
                has_aux = True,
            )(new_hidden_states_j, old_hidden_states_j, ref_hidden_states_j, input_ids_j, mask_j, advantages_j, scaling)
            accumulated_loss             .add_(unscaled_loss)
            accumulated_completion_length.add_(chunk_completion_length)
            accumulated_mean_kl          .add_(chunk_mean_kl)
            return chunk_grad_input
        pass

        accumulate_chunk = torch.compile(
            accumulate_chunk,
            fullgraph = True,
            options = torch_compile_options,
        )

        grad_inputs_chunks = torch.chunk(grad_inputs,        chunks = n_chunks, dim = 0)
        new_hidden_states  = torch.chunk(_new_hidden_states, chunks = n_chunks, dim = 0)
        if _old_hidden_states is not None: 
            old_hidden_states  = torch.chunk(_old_hidden_states, chunks = n_chunks, dim = 0)
        else: 
            old_hidden_states = [None] * n_chunks
        ref_hidden_states  = torch.chunk(_ref_hidden_states, chunks = n_chunks, dim = 0)
        input_ids          = torch.chunk(_input_ids,         chunks = n_chunks, dim = 0)
        mask               = torch.chunk(_mask,              chunks = n_chunks, dim = 0)
        advantages         = torch.chunk(_advantages,        chunks = n_chunks, dim = 0)

        # Get mixed precision scaling if seen
        scaling = scaler.get_scale() if scaler is not None else 1.0

        # Force torch.compile to use dynamic shapes for seqlen dim
        mark_dynamic = lambda x: torch._dynamo.mark_dynamic(x, 1)

        for (grad_inputs_j, new_hidden_states_j, old_hidden_states_j, ref_hidden_states_j,  input_ids_j, mask_j, advantages_j,) in \
            zip(grad_inputs_chunks, new_hidden_states, old_hidden_states, ref_hidden_states, input_ids, mask, advantages):

            mark_dynamic(new_hidden_states_j)
            mark_dynamic(ref_hidden_states_j)
            if old_hidden_states_j is not None: 
                mark_dynamic(old_hidden_states_j)
            mark_dynamic(input_ids_j)
            mark_dynamic(mask_j)

            
            grad_inputs_j.copy_(accumulate_chunk(new_hidden_states_j, old_hidden_states_j,ref_hidden_states_j,  input_ids_j, mask_j, advantages_j, scaling))
        pass

        grad_inputs                  .div_(n_chunks)
        accumulated_loss             .div_(n_chunks)
        accumulated_completion_length.div_(n_chunks)
        accumulated_mean_kl          .div_(n_chunks)
        ctx.save_for_backward(grad_inputs)
        return (
            accumulated_loss,
            accumulated_completion_length,
            accumulated_mean_kl,
        )
    pass

    @staticmethod
    def backward(ctx, grad_output, dcompletion_length, dmean_kl):
        (grad_input,) = ctx.saved_tensors
        return (grad_input, None, None, None, None, None, None, None, None, None, None)
    pass
pass
RL_REPLACEMENTS["UnslothEfficientGRPO"] = UnslothEfficientGRPO


def grpo_accumulated_loss(
    trainer,
    input_ids,
    logits_to_keep,
    completion_mask,
    advantages,
    old_hidden_states,
    n_chunks = -1,
    **kwargs,
):
    # All Unsloth Zoo code licensed under LGPLv3
    bsz, qlen = input_ids.shape
    
    # Find closest multiple
    factors = [i for i in range(1, bsz + 1) if bsz % i == 0]
    if n_chunks == -1: n_chunks = bsz
    n_chunks = factors[min(np.searchsorted(factors, n_chunks), len(factors)-1)]

    mixed_dtype = torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

    completion_input_ids = input_ids[:, -logits_to_keep:]
    lm_head = trainer.model.get_output_embeddings().weight

    with torch.amp.autocast(device_type = "cuda", dtype = mixed_dtype):
        # breakpoint()
        with torch.inference_mode(), trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper = False).disable_adapter():
            ref_hidden_states = trainer.model(input_ids = input_ids, logits_to_keep = logits_to_keep + 1).logits
        pass
        
        new_hidden_states = trainer.model(input_ids = input_ids, logits_to_keep = logits_to_keep + 1).logits
        
        loss, completion_length, mean_kl = UnslothEfficientGRPO.apply(
            new_hidden_states,
            old_hidden_states,
            ref_hidden_states,
            lm_head,
            completion_input_ids,
            completion_mask,
            advantages,
            trainer.beta,
            trainer.accelerator.scaler,
            n_chunks,
            kwargs # pass kwargs as a dict
        )

        return loss, completion_length, mean_kl

        # Old non efficient code path
        new_logits = torch.matmul(new_hidden_states, lm_head.t())
        new_logits = new_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred
        old_logits = torch.matmul(old_hidden_states, lm_head.t())
        old_logits = old_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred
        loss, completion_length, mean_kl = grpo_compute_loss(
            old_logits,
            new_logits,
            completion_input_ids,
            completion_mask,
            trainer.beta,
            advantages,
        )
        return loss, completion_length, mean_kl
    pass
pass
RL_REPLACEMENTS["grpo_accumulated_loss"] = grpo_accumulated_loss

from .dataset_utils import sft_prepare_dataset
RL_REPLACEMENTS["sft_prepare_dataset"] = sft_prepare_dataset

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
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
