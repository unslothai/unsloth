# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Chunked contrastive loss (InfoNCE / NTXent) that avoids materializing the
full similarity matrix.  Drop-in replacement for
`sentence_transformers.losses.MultipleNegativesRankingLoss`.

Supports non-square matrices (B_a != B_b) for multi-positive setups.
"""

import torch
import torch.nn.functional as F
from .utils import torch_amp_custom_fwd, torch_amp_custom_bwd


class FusedContrastiveLoss(torch.autograd.Function):
    """
    Chunked forward + backward for contrastive (InfoNCE) loss.

    embeddings_a: (B_a, D) — anchors
    embeddings_b: (B_b, D) — positives (+ extra negatives when B_b > B_a)

    The positive pair for row i is at column i (diagonal).
    Columns beyond B_a are additional negatives.

    The forward is a single streaming pass that never allocates a full
    (B_a, B_b) tensor: each chunk updates a running (max, sum) pair via the
    online log-sum-exp recurrence (Welford-style rescale when the max grows),
    extracting the positive logits on the diagonal in the same loop. This
    replaces the older two-pass approach (separate max pass + lse pass) with
    one matmul pass.
    """

    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, embeddings_a, embeddings_b, scale = 20.0):
        B_a, _dim = embeddings_a.shape
        B_b = embeddings_b.shape[0]

        if B_a == 0 or B_b == 0:
            # Save context so backward returns zero grads instead of crashing on
            # an empty ctx.saved_tensors unpack.
            ctx.empty = True
            ctx.save_for_backward(embeddings_a, embeddings_b)
            return embeddings_a.new_zeros(())

        assert (
            B_a <= B_b
        ), f"FusedContrastiveLoss requires B_a <= B_b, got {B_a} and {B_b}"

        CHUNK = min(64, B_b)

        # Online log-sum-exp: one streaming pass instead of two. Keep a running
        # max + sum per row so we don't recompute every chunk's matmul twice.
        running_max = torch.full(
            (B_a,),
            float("-inf"),
            device = embeddings_a.device,
            dtype = embeddings_a.dtype,
        )
        running_sum = torch.zeros(
            B_a, device = embeddings_a.device, dtype = embeddings_a.dtype
        )
        pos_logits = torch.zeros(
            B_a, device = embeddings_a.device, dtype = embeddings_a.dtype
        )

        for j0 in range(0, B_b, CHUNK):
            j1 = min(j0 + CHUNK, B_b)
            sim = embeddings_a @ embeddings_b[j0:j1].t() * scale  # (B_a, chunk)

            chunk_max = sim.max(dim = 1).values
            new_max = torch.maximum(running_max, chunk_max)
            # First chunk: exp(-inf - finite) == 0, so running_sum starts clean.
            rescale = torch.exp(running_max - new_max)
            running_sum = running_sum * rescale + torch.exp(
                sim - new_max.unsqueeze(1)
            ).sum(dim = 1)
            running_max = new_max

            # Gather diagonal positives sim[i, i] in one shot (no per-row loop).
            # These are RAW (unshifted) logits; the loss adds row_max back below.
            diag_hi = min(j1, B_a)
            if diag_hi > j0:
                rows = torch.arange(j0, diag_hi, device = sim.device)
                pos_logits[j0:diag_hi] = sim[rows, rows - j0]

        # row_lse is the SHIFTED lse (relative to row_max): log(sum exp(sim - row_max)).
        # backward expects that form. The full log-sum-exp is (row_max + row_lse),
        # which is why the loss adds row_max back to the raw pos_logits here.
        row_max = running_max
        row_lse = running_sum.log()
        loss = (-pos_logits + (row_max + row_lse)).mean()

        ctx.save_for_backward(embeddings_a, embeddings_b, row_max, row_lse)
        ctx.scale = scale

        return loss

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, grad_output):
        if getattr(ctx, "empty", False):
            embeddings_a, embeddings_b = ctx.saved_tensors
            return torch.zeros_like(embeddings_a), torch.zeros_like(embeddings_b), None
        embeddings_a, embeddings_b, row_max, row_lse = ctx.saved_tensors
        scale = ctx.scale

        B_a = embeddings_a.shape[0]
        B_b = embeddings_b.shape[0]
        CHUNK = min(64, B_b)

        grad_a = torch.zeros_like(embeddings_a)
        grad_b = torch.zeros_like(embeddings_b)

        for j0 in range(0, B_b, CHUNK):
            j1 = min(j0 + CHUNK, B_b)
            b_chunk = embeddings_b[j0:j1]

            sim = embeddings_a @ b_chunk.t() * scale
            prob = (sim - row_max.unsqueeze(1) - row_lse.unsqueeze(1)).exp()

            # subtract 1 on the diagonal, vectorized (was a row-by-row loop)
            diag_hi = min(j1, B_a)
            if diag_hi > j0:
                rows = torch.arange(j0, diag_hi, device = prob.device)
                prob[rows, rows - j0] -= 1.0

            prob = prob * (grad_output * scale / B_a)

            grad_a += prob @ b_chunk
            grad_b[j0:j1] += prob.t() @ embeddings_a

        return grad_a, grad_b, None


class FastMultipleNegativesRankingLoss(torch.nn.Module):
    """
    Drop-in replacement for
    ``sentence_transformers.losses.MultipleNegativesRankingLoss``
    that uses :class:`FusedContrastiveLoss` under the hood.
    """

    def __init__(self, model, scale = 20.0, similarity_fct = None):
        super().__init__()
        if similarity_fct is not None:
            import warnings

            warnings.warn(
                "Unsloth: similarity_fct is ignored by FusedContrastiveLoss (cosine similarity is hardcoded).",
                stacklevel = 2,
            )
        self.model = model
        self.scale = scale

    def forward(self, sentence_features, labels = None):
        if labels is not None:
            import warnings

            warnings.warn(
                "Unsloth: labels is ignored by FusedContrastiveLoss (positive pairs are diagonal).",
                stacklevel = 2,
            )
        reps = [self.model(sf)["sentence_embedding"] for sf in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:], dim = 0)

        embeddings_a = F.normalize(embeddings_a, p = 2, dim = 1)
        embeddings_b = F.normalize(embeddings_b, p = 2, dim = 1)

        return FusedContrastiveLoss.apply(embeddings_a, embeddings_b, self.scale)
