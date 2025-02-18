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

RL_REPLACEMENTS = dict()

torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : True,
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
@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options,)
def grpo_compute_loss(old_logits, new_logits, input_ids, mask, beta, advantages, bsz):
    old_logits = old_logits.to(torch.float32)
    new_logits = new_logits.to(torch.float32)
    input_ids  = input_ids.unsqueeze(-1)
    
    # x_i - logsumexp(x_i)
    old_x = torch.gather(old_logits, dim = -1, index = input_ids).squeeze(-1)
    new_x = torch.gather(new_logits, dim = -1, index = input_ids).squeeze(-1)
    old = old_x - torch.logsumexp(old_logits, dim = -1)
    new = new_x - torch.logsumexp(new_logits, dim = -1)

    kl_i = torch.exp(old - new) - (old - new) - 1.0
    loss_i = torch.exp(new - new.detach()) * advantages.unsqueeze(1)
    loss_i = -(loss_i - beta * kl_i)

    mask = mask.to(torch.float32)
    n_mask_per_reward = mask.sum(1)
    loss_per_reward = (loss_i * mask).sum(1) / n_mask_per_reward
    loss = loss_per_reward / bsz #.mean()
    
    # Get metrics as well which are folded
    with torch.inference_mode():
        completion_length = n_mask_per_reward / bsz #.mean()
        mean_kl_per_reward = (kl_i * mask).sum(1) / n_mask_per_reward
        mean_kl = mean_kl_per_reward / bsz #.mean()
    pass
    return loss, completion_length, mean_kl
pass
RL_REPLACEMENTS["grpo_compute_loss"] = grpo_compute_loss


def grpo_accumulated_loss(
    trainer,
    input_ids,
    logits_to_keep,
    completion_mask,
    advantages,
    n_chunks = 1,
):
    bsz, qlen = input_ids.shape
    ga = trainer.args.gradient_accumulation_steps
    n_chunks = max(min(bsz, n_chunks), 1)
    batch_ids = np.array_split(torch.arange(bsz), n_chunks)
    # for param in trainer.model.parameters(): param.grad = None
    loss              = torch.zeros(1, dtype = torch.float32, device = "cuda")
    completion_length = torch.zeros(1, dtype = torch.float32, device = "cuda")
    mean_kl           = torch.zeros(1, dtype = torch.float32, device = "cuda")
    mixed_dtype = torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16
    
    for batch_id in batch_ids:
        _completion_mask = completion_mask[batch_id]
        _input_ids = input_ids[batch_id]
        _completion_input_ids = _input_ids[:, -logits_to_keep:]
        _advantages = advantages[batch_id]

        with torch.inference_mode(), trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper = False).disable_adapter():
            old_logits = trainer.model(input_ids = _input_ids, logits_to_keep = logits_to_keep + 1)
            old_logits = old_logits.logits[:, :-1, :]
        pass

        with torch.enable_grad(), torch.amp.autocast(device_type = "cuda", dtype = mixed_dtype):
            new_logits = trainer.model(input_ids = _input_ids, logits_to_keep = logits_to_keep + 1)
            new_logits = new_logits.logits[:, :-1, :]
    
            _loss, _completion_length, _mean_kl = grpo_compute_loss(
                old_logits, new_logits, _completion_input_ids, _completion_mask, trainer.beta, _advantages, bsz,
            )
        pass
        loss              += _loss.detach()
        completion_length += _completion_length.detach()
        mean_kl           += _mean_kl.detach()
        if ga > 1: _loss = _loss / ga
        trainer.accelerator.backward(_loss)
    pass
    completion_length = completion_length.item()
    mean_kl           = mean_kl.item()
    loss              = loss.item()

    # Dummy loss to trick downstream gradients
    dummy_loss = torch.tensor(loss, dtype = torch.float32, device = "cuda")
    dummy_loss.requires_grad_(True)
    return dummy_loss, completion_length, mean_kl
pass
RL_REPLACEMENTS["grpo_accumulated_loss"] = grpo_accumulated_loss

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
