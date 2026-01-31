# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


class MPSCrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        logit_softcapping: float = 0,
        logit_scaling: float = 0,
    ):
        # Save input shape
        batch, seq_len, vocab_size = logits.shape
        logits_reshaped = logits.reshape(-1, vocab_size)
        labels_reshaped = labels.reshape(-1)

        # Use float32 for intermediate calculations for numerical stability
        # to match Triton kernels' behavior
        logits_f32 = logits_reshaped.to(torch.float32)

        if logit_scaling > 0:
            logits_f32 = logits_f32 * logit_scaling

        if logit_softcapping > 0:
            logits_f32 = logit_softcapping * torch.tanh(logits_f32 / logit_softcapping)

        # logsumexp = c + log(sum(exp(x - c)))
        logits_max = logits_f32.max(dim = -1, keepdim = True).values
        logits_shifted = logits_f32 - logits_max
        logsumexp = logits_max.squeeze(-1) + torch.log(
            torch.exp(logits_shifted).sum(dim = -1)
        )

        # Loss calculation: logsumexp - logits[label]
        mask = labels_reshaped != -100
        safe_labels = labels_reshaped.clone()
        safe_labels[~mask] = 0

        logits_label = torch.gather(logits_f32, 1, safe_labels.unsqueeze(1)).squeeze(1)

        loss = logsumexp - logits_label
        loss[~mask] = 0.0

        ctx.save_for_backward(logits_f32, labels_reshaped, logsumexp)
        ctx.logit_softcapping = logit_softcapping
        ctx.logit_scaling = logit_scaling
        ctx.logits_shape = (batch * seq_len, vocab_size)
        ctx.original_shape = logits.shape
        ctx.dtype = logits.dtype

        return loss

    @staticmethod
    def backward(ctx, dloss: torch.Tensor):
        logits_f32, labels, logsumexp = ctx.saved_tensors

        dloss_f32 = dloss.to(torch.float32).unsqueeze(1)

        # Softmax(logits) = exp(logits - logsumexp)
        probs = torch.exp(logits_f32 - logsumexp.unsqueeze(1))

        # dL/dlogits = (probs - 1[label]) * dloss
        mask = labels != -100
        safe_labels = labels.clone()
        safe_labels[~mask] = 0

        dlogits = probs
        dlogits.scatter_add_(
            1,
            safe_labels.unsqueeze(1),
            -torch.ones_like(safe_labels.unsqueeze(1), dtype = dlogits.dtype),
        )

        dlogits = dlogits * dloss_f32
        dlogits[~mask] = 0.0

        # Chain rule for softcapping
        if ctx.logit_softcapping > 0:
            # d/dx [C * tanh(x/C)] = 1 - tanh^2(x/C)
            # Since softcapped_logits = C * tanh(x/C)
            tanh_val = logits_f32 / ctx.logit_softcapping
            dlogits = dlogits * (1.0 - tanh_val * tanh_val)

        # Chain rule for scaling
        if ctx.logit_scaling > 0:
            dlogits = dlogits * ctx.logit_scaling

        return dlogits.reshape(ctx.original_shape).to(ctx.dtype), None, None, None


def mps_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    logit_softcapping: float = 0,
    logit_scaling: float = 0,
):
    losses = MPSCrossEntropyLoss.apply(logits, labels, logit_softcapping, logit_scaling)
    n_items = torch.count_nonzero(labels != -100)
    return losses.sum() / n_items
